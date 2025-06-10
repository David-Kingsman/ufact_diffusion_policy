import time
from xarm.wrapper import XArmAPI

class XArmFTSensor:
    """
    xArm 力/力矩（F/T）传感器一站式控制类，涵盖标定、零点、阻抗、力控、数据读取、配置等全部常用功能。
    """

    def __init__(self, ip):
        self.arm = XArmAPI(ip)
        self.arm.motion_enable(True)
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.set_mode(0)
        self.arm.set_state(0)

    # 1. 负载标定
    def identify_load(self, callback=None):
        self.arm.ft_sensor_enable(1)
        if callback:
            self.arm.register_iden_progress_changed_callback(callback)
        code, result = self.arm.ft_sensor_iden_load()
        if callback:
            self.arm.release_iden_progress_changed_callback(callback)
        return code, result

    # 2. 负载标定（当前负载）
    def identify_tcp_load(self, callback=None):
        if hasattr(self.arm, "iden_tcp_load"):
            if callback:
                self.arm.register_iden_progress_changed_callback(callback)
            code, result = self.arm.iden_tcp_load()
            if callback:
                self.arm.release_iden_progress_changed_callback(callback)
            return code, result
        else:
            raise NotImplementedError("当前SDK不支持 iden_tcp_load")

    # 3. 标定结果写入
    def calibrate_load(self, result):
        self.arm.ft_sensor_cali_load(result)

    # 4. 保存零点
    def save_zero(self):
        self.arm.ft_sensor_enable(1)
        time.sleep(0.1)
        self.arm.ft_sensor_set_zero()
        time.sleep(0.2)
        self.arm.save_conf()

    # 5. 设置阻抗控制参数
    def set_impedance(self, M, K, B, ref_frame=1, c_axis=None):
        self.arm.set_impedance_mbk(M, K, B)
        if c_axis is None:
            c_axis = [1, 1, 1, 1, 1, 1]
        self.arm.set_impedance_config(ref_frame, c_axis)

    # 6. 进入阻抗控制模式
    def start_impedance_mode(self, M, K, B, ref_frame=1, c_axis=None, duration=10):
        self.set_impedance(M, K, B, ref_frame, c_axis)
        self.arm.ft_sensor_enable(1)
        time.sleep(0.2)
        self.arm.ft_sensor_app_set(1)
        self.arm.set_state(0)
        print("已进入阻抗控制模式")
        time.sleep(duration)
        self.arm.ft_sensor_app_set(0)
        self.arm.ft_sensor_enable(0)

    # 7. 设置力控参数
    def set_force_control(self, Kp, Ki, Kd, linear_v_max, rot_v_max, ref_frame, force_axis, force_ref):
        self.arm.set_force_control_pid([Kp]*6, [Ki]*6, [Kd]*6, [linear_v_max]*3 + [rot_v_max]*3)
        self.arm.config_force_control(ref_frame, force_axis, force_ref, [0]*6)

    # 8. 进入力控模式
    def start_force_control_mode(self, Kp, Ki, Kd, linear_v_max, rot_v_max, ref_frame, force_axis, force_ref, duration=5):
        self.set_force_control(Kp, Ki, Kd, linear_v_max, rot_v_max, ref_frame, force_axis, force_ref)
        self.arm.ft_sensor_enable(1)
        self.arm.ft_sensor_set_zero()
        time.sleep(0.2)
        self.arm.ft_sensor_app_set(2)
        self.arm.set_state(0)
        print("已进入力控模式")
        time.sleep(duration)
        self.arm.ft_sensor_app_set(0)
        self.arm.ft_sensor_enable(0)

    # 9. 读取力/力矩数据
    def read_force(self):
        code, ext_force = self.arm.get_ft_sensor_data()
        return code, ext_force

    # 10. 读取原始力/力矩数据
    def read_raw_force(self):
        return getattr(self.arm, 'ft_raw_force', None)

    # 11. 读取外部力/力矩数据
    def read_ext_force(self):
        return getattr(self.arm, 'ft_ext_force', None)

    # 12. 获取传感器配置
    def get_sensor_config(self):
        code, config = self.arm.get_ft_senfor_config()
        return code, config

    # 13. 使能/关闭传感器
    def enable_sensor(self, enable=True):
        self.arm.ft_sensor_enable(1 if enable else 0)
        time.sleep(0.1)

    # 14. 设置零点
    def set_zero(self):
        self.enable_sensor(True)
        self.arm.ft_sensor_set_zero()
        time.sleep(0.2)

    # 15. 断开连接
    def disconnect(self):
        self.arm.disconnect()

if __name__ == '__main__':
    ip = input("请输入xArm IP地址：")
    sensor = XArmFTSensor(ip)
    menu = """
1. 负载标定
2. 当前负载标定
3. 写入标定结果
4. 保存零点
5. 设置阻抗参数并进入阻抗模式
6. 设置力控参数并进入力控模式
7. 读取力/力矩数据
8. 读取原始力/力矩数据
9. 读取外部力/力矩数据
10. 获取传感器配置
11. 断开连接并退出
"""
    print(menu)
    result = None
    while True:
        cmd = input("请选择操作：")
        if cmd == '1':
            def progress(item):
                print('progress: {}'.format(item['progress']))
            code, result = sensor.identify_load(callback=progress)
            print("标定结果:", result)
        elif cmd == '2':
            def progress(item):
                print('progress: {}'.format(item['progress']))
            code, result = sensor.identify_tcp_load(callback=progress)
            print("当前负载标定结果:", result)
        elif cmd == '3':
            if result is not None:
                sensor.calibrate_load(result)
                print("已写入标定结果")
            else:
                print("请先进行标定")
        elif cmd == '4':
            sensor.save_zero()
            print("已保存零点")
        elif cmd == '5':
            # 示例参数，可根据实际需求调整
            M = [0.06]*3 + [0.0006]*3
            K = [300]*3 + [4]*3
            B = [0]*6
            c_axis = [0,0,1,0,0,0]
            ref_frame = 0
            sensor.start_impedance_mode(M, K, B, ref_frame, c_axis, duration=10)
        elif cmd == '6':
            # 示例参数，可根据实际需求调整
            Kp = 0.005
            Ki = 0.00005
            Kd = 0.05
            linear_v_max = 200.0
            rot_v_max = 0.35
            ref_frame = 1
            force_axis = [0,0,1,0,0,0]
            force_ref = [0,0,5.0,0,0,0]
            sensor.start_force_control_mode(Kp, Ki, Kd, linear_v_max, rot_v_max, ref_frame, force_axis, force_ref, duration=5)
        elif cmd == '7':
            code, force = sensor.read_force()
            print("力/力矩数据:", force)
        elif cmd == '8':
            print("原始力/力矩数据:", sensor.read_raw_force())
        elif cmd == '9':
            print("外部力/力矩数据:", sensor.read_ext_force())
        elif cmd == '10':
            code, config = sensor.get_sensor_config()
            if code == 0:
                print('ft_app_status:', config[0])
                print('ft_is_started:', config[1])
                print('ft_type:', config[2])
                print('ft_id:', config[3])
                print('ft_freq:', config[4])
                print('ft_mass:', config[5])
                print('ft_dir_bias:', config[6])
                print('ft_centroid:', config[7])
                print('ft_zero:', config[8])
                print('imp_coord:', config[9])
                print('imp_c_axis:', config[10])
                print('M:', config[11])
                print('K:', config[12])
                print('B:', config[13])
                print('f_coord:', config[14])
                print('f_c_axis:', config[15])
                print('f_ref:', config[16])
                print('f_limits:', config[17])
                print('kp:', config[18])
                print('ki:', config[19])
                print('kd:', config[20])
                print('xe_limit:', config[21])
            else:
                print("获取配置失败，code:", code)
        elif cmd == '11':
            sensor.disconnect()
            print("已断开连接")
            break
        else:
            print("无效输入")