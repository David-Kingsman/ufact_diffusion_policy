import argparse
import json
import time
import numpy as np
import zmq
from collections import deque

from eval_agent import Agent

def main(cfg):
    name = cfg['name']
    robot_addr = [cfg['robot_addr1'], cfg['robot_addr2']]
    port = cfg['port']
    obs_horizon = cfg['obs_horizon']
    gripper_port = cfg.get('gripper_port', '/dev/ttyUSB0')
    camera_serial = cfg.get('camera_serial', None)
    ft_sensor_ip = cfg.get('ft_sensor_ip', None)

    # 用Agent统一初始化所有设备
    agent = Agent(
        robot_ip=robot_addr[0],
        gripper_port=gripper_port,
        camera_serial=camera_serial,
        ft_sensor_ip=ft_sensor_ip,
        obs_horizon=obs_horizon
    )

    # 初始化 zmq 客户端
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://127.0.0.1:{port}")
    socket.RCVTIMEO = 3000
    socket.SNDTIMEO = 3000

    action_deque = deque(maxlen=100)
    finish = False
    while not finish:
        start_time = time.time()
        try:
            # 采集观测
            obs_dict = agent.get_observation()
            # 构造 obs，字段名与训练/推理完全一致
            obs = {
                "name": name,
                "robot_eef_pose": obs_dict["eef_pose"].tolist(),
                "robot_joint": obs_dict["joint"].tolist() if obs_dict["joint"] is not None else [],
                "robot_joint_vel": obs_dict["joint_vel"].tolist() if obs_dict["joint_vel"] is not None else [],
                "label": 0,
                "force": obs_dict["force"].tolist(),
                "camera_rgb": obs_dict["color"].tolist(),
                "camera_depth": obs_dict["depth"].tolist(),
                # 如有点云可加上
                # "pointcloud": obs_dict.get("pointcloud", np.zeros((2048, 3), dtype=np.float32)).tolist(),
            }

            # 发送 obs 到 node.py 服务端
            data_json = json.dumps(obs)
            socket.send_string(data_json)
            print("已发送 obs，等待动作返回...")
            received_data_json = socket.recv_string()

            received_data = json.loads(received_data_json)
            pred_action = np.array(received_data["action"])  # shape: (T, D)
            print("收到动作序列:", pred_action.shape)

            # 将动作加入队列
            for act in pred_action:
                action_deque.append(act)

            # 执行动作（假设策略输出为关节角度+夹爪开合度）
            if action_deque:
                act = action_deque.popleft()
                # 推荐用 Agent 封装接口
                if len(act) >= 6:
                    agent.robot._arm.set_servo_angle(angle=act[:6], is_radian=False, wait=True)
                if len(act) > 6:
                    gripper_pos = int(np.clip(act[-1], 0, 255))
                    agent.gripper.action(gripper_pos)

            # 控制频率限制（如 20Hz）
            elapsed = time.time() - start_time
            time.sleep(max(0, 0.05 - elapsed))
        except zmq.error.Again:
            print("ZMQ 超时，推理服务未响应，跳过本轮")
            continue
        except Exception as e:
            print("主循环异常:", e)
            continue

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
    parser.add_argument("--camera_serial", type=str, default=None)
    parser.add_argument("--ft_sensor_ip", type=str, default=None)

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
            "camera_serial": args.camera_serial,
            "ft_sensor_ip": args.ft_sensor_ip,
        }
    main(cfg)