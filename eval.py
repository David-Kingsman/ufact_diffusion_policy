import argparse
import json
import time
import numpy as np
import zmq
import cv2
from collections import deque

from devices.robot.xarm6 import XARM6
from devices.gripper.robotiq import Robotiq2FGripper
from devices.camera.realsense import RealSenseRGBDCamera

def main(cfg):
    name = cfg['name']
    robot_addr = [cfg['robot_addr1'], cfg['robot_addr2']]
    port = cfg['port']
    obs_horizon = cfg['obs_horizon']
    act_horizon = cfg['act_horizon']
    gripper_port = cfg.get('gripper_port', '/dev/ttyUSB0')
    cam = RealSenseRGBDCamera(serial=cfg.get('camera_serial', None))

    # 初始化 zmq 客户端
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://127.0.0.1:{port}")
    socket.RCVTIMEO = 3000
    socket.SNDTIMEO = 3000

    # 初始化机器人
    robot = XARM6(interface=robot_addr[0])
    robot.home()
    robot.set_zero_ft()

    # 初始化夹爪
    gripper = Robotiq2FGripper(port=gripper_port)
    gripper.open_gripper()

    # 环境观测缓存
    robot_eef_pose_hist = deque(maxlen=obs_horizon)
    robot_joint_hist = deque(maxlen=obs_horizon)
    robot_joint_vel_hist = deque(maxlen=obs_horizon)
    label_hist = deque(maxlen=obs_horizon)
    force_hist = deque(maxlen=obs_horizon)

    action_deque = deque(maxlen=100)

    finish = False
    while not finish:
        start_time = time.time()
        # 采集观测
        eef_pose = robot.get_tcp_pose(flatten=True)  # shape: (7,)
        joint = robot.get_joints()                   # shape: (6,) or (7,)
        joint_vel = robot.get_joint_vels()           # shape: (6,) or (7,)
        label = 0                                    # 可根据实际任务设置
        color_image, depth_image = cam.get_data()  # shape: (H, W, 3) and (H, W)
        # resize 到训练分辨率
        color_image = cv2.resize(color_image, (224, 224))
        depth_image = cv2.resize(depth_image, (224, 224))
        if color_image.dtype != np.float32:
            color_image = color_image.astype(np.float32) / 255.0
        if depth_image.dtype != np.float32:
            depth_image = depth_image.astype(np.float32)
        # 过滤异常
        color_image = np.nan_to_num(color_image)
        depth_image = np.nan_to_num(depth_image)
        # 点云采集健壮性：点云可能采集失败，判空或 try/except，避免 .tolist() 报错。可对点云数量做下采样（如 voxel downsample 或随机采样）。
        try:
            pointcloud = cam.get_pointcloud()  # shape: (N, 3)
            if pointcloud is None or len(pointcloud) == 0:
                pointcloud = np.zeros((2048, 3), dtype=np.float32)
            else:
                pointcloud = pointcloud.astype(np.float32)
                # 过滤 nan/inf
                pointcloud = pointcloud[np.isfinite(pointcloud).all(axis=1)]
                # 体素下采样（如有 voxel_down_sample 方法）
                # pointcloud = voxel_down_sample(pointcloud, voxel_size=0.005)
                # 随机采样最多 2048 个点
                if pointcloud.shape[0] > 2048:
                    idx = np.random.choice(pointcloud.shape[0], 2048, replace=False)
                    pointcloud = pointcloud[idx]
                elif pointcloud.shape[0] < 2048:
                    pad = np.zeros((2048 - pointcloud.shape[0], 3), dtype=np.float32)
                    pointcloud = np.concatenate([pointcloud, pad], axis=0)
        except Exception as e:
            print("Pointcloud error:", e)
            pointcloud = np.zeros((2048, 3), dtype=np.float32)
        try:
            force = robot.get_ft_sensor_data()
            force = force.astype(np.float32)
            if not np.all(np.isfinite(force)):
                force = np.zeros(6, dtype=np.float32)
        except Exception:
            force = np.zeros(6, dtype=np.float32)
        force_hist.append(force)
        if len(force_hist) == obs_horizon:
            force_smooth = np.mean(np.array(force_hist), axis=0)
        else:
            force_smooth = force

        robot_eef_pose_hist.append(eef_pose)
        robot_joint_hist.append(joint)
        robot_joint_vel_hist.append(joint_vel)
        label_hist.append(label)
        force_hist.append(force)
        if len(force_hist) == obs_horizon:
            force_array = np.array(force_hist)
            force_smooth = force_array.mean(axis=0)
        else:
            force_smooth = force
        if len(robot_eef_pose_hist) < obs_horizon:
            time.sleep(0.01)
            continue

        # 构造 obs，字段名与训练/推理完全一致
        obs = {
            "name": name,
            "robot_eef_pose": np.array(robot_eef_pose_hist).tolist(),
            "robot_joint": np.array(robot_joint_hist).tolist(),
            "robot_joint_vel": np.array(robot_joint_vel_hist).tolist(),
            "label": np.array(label_hist).tolist(),
            "force": force_smooth.tolist(),
            "camera_rgb": color_image.tolist(),
            "camera_depth": depth_image.tolist(),
            "pointcloud": pointcloud.tolist(),
        }

        # 发送 obs 到 node.py 服务端
        data_json = json.dumps(obs)
        try:
            socket.send_string(data_json)
            print("已发送 obs，等待动作返回...")
            received_data_json = socket.recv_string()
        except zmq.error.Again:
            print("ZMQ 超时，推理服务未响应，跳过本轮")
            continue

        received_data = json.loads(received_data_json)
        pred_action = np.array(received_data["action"])  # shape: (T, D)
        print("收到动作序列:", pred_action.shape)

        # 将动作加入队列
        for act in pred_action:
            action_deque.append(act)

        # 执行动作（假设策略输出为关节角度+夹爪开合度）
        if action_deque:
            act = action_deque.popleft()
            # 机械臂动作
            robot.move_joints([act[:6]])  # 若为7轴，改为[:7]
            # 夹爪动作（假设act[-1]为0~255的夹爪位置，需根据实际策略输出调整）
            if len(act) > 6:
                gripper_pos = int(np.clip(act[-1], 0, 255))
                gripper.action(gripper_pos)

        # 控制频率限制（如 20Hz）
        elapsed = time.time() - start_time
        time.sleep(max(0, 0.05 - elapsed))

        # 退出条件（可自定义，比如按键退出）
        # if ...:
        #     finish = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--name", type=str)
    parser.add_argument("--robot_addr1", type=str)
    parser.add_argument("--robot_addr2", type=str)
    parser.add_argument("--port", type=int)
    parser.add_argument("--obs_horizon", type=int, default=1)
    parser.add_argument("--act_horizon", type=int, default=10)
    parser.add_argument("--gripper_port", type=str, default="/dev/ttyUSB0")

    args = parser.parse_args()
    if args.config is not None:
        cfg = json.load(open(args.config, 'r'))
    else:
        cfg = {
            "name": args.name,
            "robot_addr1": args.robot_addr1,
            "robot_addr2": args.robot_addr2,
            "port": args.port,
            "obs_horizon": args.obs_horizon,
            "act_horizon": args.act_horizon,
            "gripper_port": args.gripper_port,
        }
    main(cfg)