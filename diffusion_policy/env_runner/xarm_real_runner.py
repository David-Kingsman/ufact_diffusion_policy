import threading
import time
import os
import os.path as osp
import numpy as np
import torch
import tqdm
from loguru import logger
from typing import Dict, Tuple, Union, Optional, Any, List  
import transforms3d as t3d
from omegaconf import DictConfig, ListConfig
from copy import deepcopy

from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

# 添加XARM相关的导入
try:
    # 假设你有XARM的Python SDK
    from xarm.wrapper import XArmAPI
    XARM_AVAILABLE = True
except ImportError:
    logger.warning("XARM SDK not available, using simulation mode")
    XARM_AVAILABLE = False

import cv2
cv2.setNumThreads(4)

class XarmRealRunner(BaseImageRunner):
    """
    完整的XARM真实机器人评估器
    支持真实机器人控制、相机采集、动作推理
    """
    
    def __init__(self,
                 output_dir: str,
                 shape_meta: DictConfig,
                 # XARM配置
                 xarm_ip: str = "192.168.1.235",  # XARM机器人IP
                 camera_indices: List[int] = [0],  # 相机设备索引
                 # 评估参数
                 eval_episodes: int = 10,
                 max_duration_time: float = 30.0,
                 max_steps: int = 400,
                 # 控制参数
                 control_fps: float = 20.0,
                 inference_fps: float = 10.0,
                 n_obs_steps: int = 2,
                 n_action_steps: int = 8,
                 # 动作参数
                 action_horizon: int = 16,
                 use_relative_action: bool = False,
                 # 安全参数
                 position_limits: List[List[float]] = None,
                 velocity_limit: float = 100.0,  # mm/s
                 # 可视化
                 enable_video_recording: bool = True,
                 save_obs: bool = True,
                 **kwargs):
        
        super().__init__(output_dir)
        
        self.shape_meta = dict(shape_meta)
        self.eval_episodes = eval_episodes
        self.max_duration_time = max_duration_time
        self.max_steps = max_steps
        
        # 控制频率
        self.control_fps = control_fps
        self.inference_fps = inference_fps
        self.control_interval = 1.0 / control_fps
        self.inference_interval = 1.0 / inference_fps
        assert control_fps % inference_fps == 0
        
        # 观测和动作参数
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.action_horizon = action_horizon
        self.use_relative_action = use_relative_action
        
        # 安全限制
        self.position_limits = position_limits or [
            [200, -400, 100],   # xyz min (mm)
            [700, 400, 600]     # xyz max (mm)  
        ]
        self.velocity_limit = velocity_limit
        
        # 可视化
        self.enable_video_recording = enable_video_recording
        self.save_obs = save_obs
        self.video_dir = osp.join(output_dir, 'videos')
        self.obs_dir = osp.join(output_dir, 'observations')
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.obs_dir, exist_ok=True)
        
        # 初始化XARM
        self.xarm_ip = xarm_ip
        self.xarm = None
        self.cameras = []
        self.camera_indices = camera_indices
        
        # 线程控制
        self.stop_event = threading.Event()
        self.action_queue = []
        self.action_lock = threading.Lock()
        
        # 初始化硬件
        self._init_xarm()
        self._init_cameras()
        
        logger.info(f"XarmRealRunner initialized: {eval_episodes} episodes, "
                   f"control_fps={control_fps}, inference_fps={inference_fps}")
    
    def _init_xarm(self):
        """初始化XARM机器人"""
        if not XARM_AVAILABLE:
            logger.warning("XARM SDK not available, skipping robot initialization")
            return
            
        try:
            self.xarm = XArmAPI(self.xarm_ip)
            self.xarm.connect()
            
            # 使能机器人
            self.xarm.motion_enable(enable=True)
            self.xarm.set_mode(0)  # 位置控制模式
            self.xarm.set_state(state=0)  # 运动状态
            
            # 设置安全参数
            self.xarm.set_tcp_maxacc(1000)  # 最大加速度
            self.xarm.set_tcp_jerk(10000)   # 最大jerk
            
            logger.info(f"XARM connected successfully: {self.xarm_ip}")
            
        except Exception as e:
            logger.error(f"Failed to initialize XARM: {e}")
            self.xarm = None
    
    def _init_cameras(self):
        """初始化相机"""
        for idx in self.camera_indices:
            try:
                cap = cv2.VideoCapture(idx)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                if cap.isOpened():
                    self.cameras.append(cap)
                    logger.info(f"Camera {idx} initialized successfully")
                else:
                    logger.warning(f"Failed to open camera {idx}")
                    
            except Exception as e:
                logger.error(f"Error initializing camera {idx}: {e}")
    
    def _get_robot_state(self) -> Dict[str, np.ndarray]:
        """获取机器人当前状态"""
        if self.xarm is None:
            # 模拟状态
            return {
                'robot_eef_pose': np.random.randn(6),  # xyz + rpy
                'robot_joint': np.random.randn(7),     # 7个关节角度
                'robot_joint_vel': np.random.randn(7), # 7个关节速度
                'gripper': np.random.rand(1) * 850     # 夹爪开度
            }
        
        try:
            # 获取TCP位姿
            ret, tcp_pose = self.xarm.get_position()
            if ret != 0:
                tcp_pose = [0, 0, 0, 0, 0, 0]
            
            # 获取关节角度
            ret, joint_angles = self.xarm.get_servo_angle()
            if ret != 0:
                joint_angles = [0] * 7
            
            # 获取关节速度 (如果支持)
            joint_velocities = [0] * 7  # XARM可能不直接提供速度
            
            # 获取夹爪状态
            ret, gripper_pos = self.xarm.get_gripper_position()
            if ret != 0:
                gripper_pos = 0
            
            return {
                'robot_eef_pose': np.array(tcp_pose, dtype=np.float32),
                'robot_joint': np.array(joint_angles, dtype=np.float32) * np.pi / 180,  # 转换为弧度
                'robot_joint_vel': np.array(joint_velocities, dtype=np.float32),
                'gripper': np.array([gripper_pos], dtype=np.float32)
            }
            
        except Exception as e:
            logger.error(f"Error getting robot state: {e}")
            return self._get_robot_state()  # 返回模拟状态
    
    def _capture_images(self) -> Dict[str, np.ndarray]:
        """采集相机图像"""
        images = {}
        
        for i, cap in enumerate(self.cameras):
            try:
                ret, frame = cap.read()
                if ret:
                    # 调整尺寸并归一化
                    frame = cv2.resize(frame, (224, 224))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = frame.astype(np.float32) / 255.0
                    frame = frame.transpose(2, 0, 1)  # HWC -> CHW
                    
                    # 使用配置中的相机名称
                    if i == 0:
                        images['agentview_image'] = frame
                    else:
                        images[f'camera_{i}'] = frame
                else:
                    logger.warning(f"Failed to capture from camera {i}")
                    # 使用黑色图像作为fallback
                    images['agentview_image'] = np.zeros((3, 224, 224), dtype=np.float32)
                    
            except Exception as e:
                logger.error(f"Error capturing from camera {i}: {e}")
                images['agentview_image'] = np.zeros((3, 224, 224), dtype=np.float32)
        
        return images
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """获取完整观测"""
        # 获取机器人状态
        robot_state = self._get_robot_state()
        
        # 获取图像
        images = self._capture_images()
        
        # 合并观测
        obs = {**robot_state, **images}
        return obs
    
    def _execute_action(self, action: np.ndarray):
        """执行动作"""
        if self.xarm is None:
            logger.debug(f"Simulated action: {action}")
            time.sleep(0.01)  # 模拟执行时间
            return
        
        try:
            # 解析动作 (假设是7维: xyz + rpy + gripper)
            if len(action) >= 7:
                xyz = action[:3] * 1000  # 转换为mm
                rpy = action[3:6] * 180 / np.pi  # 转换为度
                gripper = action[6] if len(action) > 6 else 0
                
                # 安全检查
                xyz = np.clip(xyz, self.position_limits[0], self.position_limits[1])
                
                # 执行TCP运动
                pose = list(xyz) + list(rpy)
                ret = self.xarm.set_position(*pose, 
                                           speed=self.velocity_limit, 
                                           wait=False)
                
                if ret != 0:
                    logger.warning(f"XARM motion command failed: {ret}")
                
                # 执行夹爪动作
                if len(action) > 6:
                    gripper_pos = np.clip(gripper, 0, 850)
                    self.xarm.set_gripper_position(gripper_pos, wait=False)
                    
        except Exception as e:
            logger.error(f"Error executing action: {e}")
    
    def _action_control_thread(self, policy):
        """动作控制线程"""
        step_count = 0
        
        while not self.stop_event.is_set():
            start_time = time.time()
            
            # 检查是否有新动作
            with self.action_lock:
                if self.action_queue:
                    action = self.action_queue.pop(0)
                    self._execute_action(action)
            
            # 保持控制频率
            elapsed = time.time() - start_time
            sleep_time = max(0, self.control_interval - elapsed)
            time.sleep(sleep_time)
            step_count += 1
    
    def _collect_observation_sequence(self) -> Dict[str, np.ndarray]:
        """收集观测序列"""
        obs_sequence = []
        
        for _ in range(self.n_obs_steps):
            obs = self._get_observation()
            obs_sequence.append(obs)
            time.sleep(0.1)  # 短暂间隔
        
        # 转换为批次格式
        batch_obs = {}
        for key in obs_sequence[0].keys():
            values = [obs[key] for obs in obs_sequence]
            batch_obs[key] = np.stack(values, axis=0)
        
        return batch_obs
    
    def run(self, policy: DiffusionUnetImagePolicy) -> Dict[str, Any]:
        """运行完整评估"""
        logger.info(f"Starting XARM evaluation with {self.eval_episodes} episodes")
        
        policy.eval()
        device = policy.device
        
        all_episode_results = []
        
        try:
            for episode_idx in range(self.eval_episodes):
                logger.info(f"Episode {episode_idx + 1}/{self.eval_episodes}")
                
                # 重置环境 (手动或自动)
                self._reset_episode()
                
                # 启动动作控制线程
                self.stop_event.clear()
                self.action_queue = []
                
                action_thread = threading.Thread(
                    target=self._action_control_thread, 
                    args=(policy,), 
                    daemon=True
                )
                action_thread.start()
                
                # 运行单个episode
                episode_result = self._run_episode(policy, device, episode_idx)
                all_episode_results.append(episode_result)
                
                # 停止控制线程
                self.stop_event.set()
                action_thread.join(timeout=2.0)
                
                logger.info(f"Episode {episode_idx + 1} completed: "
                           f"reward={episode_result['reward']:.3f}, "
                           f"success={episode_result['success']}")
        
        finally:
            self._cleanup()
        
        # 计算统计结果
        results = self._compute_statistics(all_episode_results)
        logger.info("Evaluation completed!")
        self._log_results(results)
        
        return results
    
    def _reset_episode(self):
        """重置episode环境"""
        if self.xarm is not None:
            try:
                # 移动到安全位置
                self.xarm.set_position(400, 0, 300, 0, 0, 0, speed=100, wait=True)
                # 打开夹爪
                self.xarm.set_gripper_position(850, wait=True)
                logger.info("XARM reset to initial position")
            except Exception as e:
                logger.error(f"Error resetting XARM: {e}")
        
        # 用户确认
        input("Press Enter when environment is ready...")
    
    def _run_episode(self, policy, device, episode_idx) -> Dict[str, Any]:
        """运行单个episode"""
        episode_reward = 0.0
        success = False
        step_count = 0
        
        episode_start_time = time.time()
        last_inference_time = episode_start_time
        
        episode_obs = []
        episode_actions = []
        
        try:
            while step_count < self.max_steps:
                current_time = time.time()
                
                # 检查时间限制
                if current_time - episode_start_time > self.max_duration_time:
                    logger.info("Episode timeout")
                    break
                
                # 推理频率控制
                if current_time - last_inference_time >= self.inference_interval:
                    # 收集观测
                    obs_dict = self._collect_observation_sequence()
                    
                    # 转换为tensor
                    obs_tensor = {}
                    for key, value in obs_dict.items():
                        obs_tensor[key] = torch.from_numpy(value).float().unsqueeze(0).to(device)
                    
                    # 策略推理
                    with torch.no_grad():
                        action_dict = policy.predict_action(obs_tensor)
                    
                    # 提取动作序列
                    actions = action_dict['action'][0].cpu().numpy()  # (action_horizon, action_dim)
                    
                    # 添加动作到队列
                    with self.action_lock:
                        # 只取前几个动作步
                        for i in range(min(self.n_action_steps, len(actions))):
                            self.action_queue.append(actions[i])
                    
                    # 保存数据
                    if self.save_obs:
                        episode_obs.append(obs_dict)
                        episode_actions.append(actions)
                    
                    # 简单的奖励计算 (可以根据任务定制)
                    step_reward = self._compute_reward(obs_dict)
                    episode_reward += step_reward
                    
                    # 简单的成功判断 (可以根据任务定制)
                    if step_reward > 0.8:
                        success = True
                        break
                    
                    last_inference_time = current_time
                    step_count += 1
                
                time.sleep(0.01)  # 短暂休眠
                
        except KeyboardInterrupt:
            logger.warning("Episode interrupted by user")
        
        # 保存episode数据
        if self.save_obs:
            self._save_episode_data(episode_idx, episode_obs, episode_actions)
        
        return {
            'reward': episode_reward,
            'success': success,
            'steps': step_count,
            'duration': time.time() - episode_start_time
        }
    
    def _compute_reward(self, obs_dict: Dict[str, np.ndarray]) -> float:
        """计算奖励 (需要根据具体任务定制)"""
        # 这里是一个简单的示例
        # 可以基于任务完成度、距离目标的距离等计算
        
        # 示例: 基于末端执行器位置计算奖励
        eef_pose = obs_dict['robot_eef_pose'][-1]  # 最新的位姿
        target_position = np.array([0.5, 0.0, 0.3])  # 示例目标位置
        
        distance = np.linalg.norm(eef_pose[:3] - target_position)
        reward = max(0, 1.0 - distance)  # 距离越近奖励越高
        
        return reward
    
    def _save_episode_data(self, episode_idx: int, obs_list: list, action_list: list):
        """保存episode数据"""
        try:
            episode_data = {
                'observations': obs_list,
                'actions': action_list,
                'episode_idx': episode_idx
            }
            
            save_path = osp.join(self.obs_dir, f'episode_{episode_idx}.npz')
            np.savez_compressed(save_path, **episode_data)
            logger.debug(f"Episode data saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving episode data: {e}")
    
    def _compute_statistics(self, episode_results: list) -> Dict[str, float]:
        """计算统计结果"""
        if not episode_results:
            return {}
        
        rewards = [r['reward'] for r in episode_results]
        successes = [r['success'] for r in episode_results]
        durations = [r['duration'] for r in episode_results]
        
        return {
            'train_mean_score': np.mean(rewards),
            'train_std_score': np.std(rewards),
            'test_mean_score': np.mean(rewards),  # 同训练集
            'test_std_score': np.std(rewards),
            'train_success_rate': np.mean(successes),
            'test_success_rate': np.mean(successes),
            'mean_duration': np.mean(durations),
            'total_episodes': len(episode_results)
        }
    
    def _log_results(self, results: Dict[str, float]):
        """记录结果"""
        logger.info("\n📊 XARM Evaluation Results:")
        for key, value in results.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.3f}")
            else:
                logger.info(f"  {key}: {value}")
    
    def _cleanup(self):
        """清理资源"""
        # 停止线程
        self.stop_event.set()
        
        # 关闭相机
        for cap in self.cameras:
            cap.release()
        
        # 断开XARM连接
        if self.xarm is not None:
            try:
                self.xarm.disconnect()
                logger.info("XARM disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting XARM: {e}")
