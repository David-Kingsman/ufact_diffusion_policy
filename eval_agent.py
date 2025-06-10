"""
Evaluation Agent for xArm6 + Robotiq2F + RealSense + xArm FT Sensor
"""

import time
import numpy as np
from devices.robot.xarm6 import XARM6
from devices.gripper.robotiq import Robotiq2FGripper
from devices.camera.realsense import RealSenseRGBDCamera
from devices.sensors.ft_sensor import XArmFTSensor
from utils.transformation import xyz_rot_transform
from collections import deque 

class Agent:
    """
    统一封装xArm6机械臂、Robotiq夹爪、RealSense相机、xArm力/力矩传感器
    """
    def __init__(
        self, 
        robot_ip, 
        gripper_port, 
        camera_serial, 
        ft_sensor_ip=None,
        obs_horizon=100
    ):
        self.camera_serial = camera_serial
        print("Init robot, gripper, camera, ft sensor.")  
        # 初始化机器人
        self.robot = XARM6(interface=robot_ip)
        self.robot.reset() 
        time.sleep(1.5)
        # 初始化夹爪
        self.gripper = Robotiq2FGripper(port=gripper_port)
        self.gripper.open_gripper()    # 打开夹爪
        time.sleep(0.5)
        self.gripper.close_gripper()   # 关闭夹爪（可选，确保激活）
        self.gripper.open_gripper()    # 再次打开，准备 ready
        time.sleep(0.5)
        # 初始化力/力矩传感器
        self.ft_sensor = XArmFTSensor(ft_sensor_ip) if ft_sensor_ip else None
        self.force_history = deque(maxlen=obs_horizon)
        if self.ft_sensor:
            code, force = self.ft_sensor.read_force()
            self.force_history.append(force)
        self.camera = RealSenseRGBDCamera(serial=camera_serial)
        for _ in range(30):
            self.camera.get_rgbd_image()
        print("Initialization Finished.")

    @property
    def intrinsics(self):
        # 实际相机内参
        return np.array([
            [922.37457275, 0, 637.55419922],
            [0, 922.46069336, 368.37557983],
            [0, 0, 1]
        ])

    @property
    def ready_pose(self):
        return np.array([0, -60, -30, 0, 90, 0])

    @property
    def ready_rot_6d(self):
        # 你可以根据实际 ready pose 的旋转部分调整
        return np.array([-1, 0, 0, 0, 1, 0])

    def get_observation(self):
        color, depth = self.camera.get_rgbd_image()
        eef_pose = self.robot.get_current_pose()
        joint = getattr(self.robot._arm, "angles", None)
        joint_vel = getattr(self.robot._arm, "velocities", None)
        if self.ft_sensor:
            try:
                code, force = self.ft_sensor.read_force()
                if code == 0:
                    force = np.array(force)
                else:
                    force = np.zeros(6)
            except Exception as e:
                print("FT sensor read error:", e)
                force = np.zeros(6)
        else:
            force = np.zeros(6)
        self.force_history.append(force)
        return dict(
            color=color,
            depth=depth,
            eef_pose=np.array(eef_pose),
            joint=np.array(joint) if joint is not None else None,
            joint_vel=np.array(joint_vel) if joint_vel is not None else None,
            force=force
        )

    def get_force_torque_history(self):
        return np.stack(self.force_history) if len(self.force_history) > 0 else np.zeros((0, 6))

    def get_force_torque(self):
        if len(self.force_history) > 0:
            return self.force_history[-1]
        else:
            return np.zeros(6)

    def get_force(self):
        return self.get_force_torque()[:3]

    def get_torque(self):
        return self.get_force_torque()[3:]

    def get_force_torque_value(self):
        ft = self.get_force_torque()
        return np.linalg.norm(ft[:3]), np.linalg.norm(ft[3:])

    def get_force_value(self):
        return np.linalg.norm(self.get_force())

    def get_torque_value(self):
        return np.linalg.norm(self.get_torque())

    def get_tcp_pose(self):
        return self.robot.get_current_pose()

    def set_tcp_pose(self, pose, rotation_rep, rotation_rep_convention=None, blocking=False):
        tcp_pose = xyz_rot_transform(
            pose,
            from_rep=rotation_rep,
            to_rep="euler",  # xArm SDK常用欧拉角
            from_convention=rotation_rep_convention
        )
        self.robot.move_to_pose(tcp_pose, wait=blocking)

    def set_gripper_width(self, width, blocking=False):
        # Robotiq2F 夹爪最大宽度 0.085m
        width = float(np.clip(width, 0, 0.085))
        pos = int(width / 0.085 * 255)
        self.gripper.action(pos)
        if blocking:
            time.sleep(0.5)

    def stop(self):
        if self.ft_sensor:
            self.ft_sensor.disconnect()
        if hasattr(self.robot, "_arm"):
            self.robot._arm.disconnect()
        # 夹爪和相机如有关闭接口也可加上