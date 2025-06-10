建python3 conda环境（使用Python 3.10测试）并运行以下命令：
1. 创建python 3.10 conda环境： conda create --name xarm_robomimic_env python=3.10





单独测试每个设备的 Python 控制接口
在 devices 目录下，分别运行如下测试代码，确保每个设备都能正常工作
[1] RealSense 相机
from devices.camera.realsense import RealSenseRGBDCamera
import cv2

cam = RealSenseRGBDCamera()
color, depth = cam.get_data()
print("Color shape:", color.shape, "Depth shape:", depth.shape)
cv2.imshow("color", color)
cv2.imshow("depth", depth / depth.max())
cv2.waitKey(0)
[2] xArm6 机械臂
from devices.robot.xarm6 import XARM6

robot = XARM6(addr=["192.168.1.XXX", "192.168.1.XXX"], urdf_path="xxx.urdf")
print("Joints:", robot.get_joints())
print("TCP pose:", robot.get_tcp_pose(flatten=True))
robot.move_joints([[0, 0, 0, 0, 0, 0]])  # 测试运动
[3]  ufactory 力传感器
from devices.robot.xarm6 import XARM6

robot = XARM6(addr=["192.168.1.XXX", "192.168.1.XXX"], urdf_path="xxx.urdf")
print("FT sensor:", robot.get_ft_sensor_data())
[4] Robotiq 2F85 夹爪
from devices.gripper.robotiq import Robotiq2FGripper

gripper = Robotiq2FGripper(port="/dev/ttyUSB0")
gripper.open_gripper()
gripper.action(100)  # 夹爪闭合到100





























