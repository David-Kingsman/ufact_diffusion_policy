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

# Add XARM related imports
try:
    # Assuming you have XARM Python SDK
    from xarm.wrapper import XArmAPI
    XARM_AVAILABLE = True
except ImportError:
    logger.warning("XARM SDK not available, using simulation mode")
    XARM_AVAILABLE = False

import cv2
cv2.setNumThreads(4)

class XarmRealRunner(BaseImageRunner):
    """
    Complete XARM real robot evaluator
    Supports real robot control, camera capture, and action inference
    """
    
    def __init__(self,
                 output_dir: str,
                 shape_meta: DictConfig,
                 # XARM configuration
                 xarm_ip: str = "192.168.1.235",  # XARM机器人IP
                 camera_indices: List[int] = [0],  # Camera device indices
                 # Evaluation parameters
                 eval_episodes: int = 10,
                 max_duration_time: float = 30.0,
                 max_steps: int = 400,
                 # Control parameters
                 control_fps: float = 20.0,
                 inference_fps: float = 10.0,
                 n_obs_steps: int = 2,
                 n_action_steps: int = 8,
                 # Action parameters
                 action_horizon: int = 16,
                 use_relative_action: bool = False,
                 # Safety parameters
                 position_limits: List[List[float]] = None,
                 velocity_limit: float = 100.0,  # mm/s
                 # Visualization
                 enable_video_recording: bool = True,
                 save_obs: bool = True,
                 **kwargs):
        
        super().__init__(output_dir)
        
        self.shape_meta = dict(shape_meta)
        self.eval_episodes = eval_episodes
        self.max_duration_time = max_duration_time
        self.max_steps = max_steps
        
        # Control frequency
        self.control_fps = control_fps
        self.inference_fps = inference_fps
        self.control_interval = 1.0 / control_fps
        self.inference_interval = 1.0 / inference_fps
        assert control_fps % inference_fps == 0
        
        # Observation and action parameters
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.action_horizon = action_horizon
        self.use_relative_action = use_relative_action
        
        # Safety limits
        self.position_limits = position_limits or [
            [200, -400, 100],   # xyz min (mm)
            [700, 400, 600]     # xyz max (mm)  
        ]
        self.velocity_limit = velocity_limit
        
        # Visualization
        self.enable_video_recording = enable_video_recording
        self.save_obs = save_obs
        self.video_dir = osp.join(output_dir, 'videos')
        self.obs_dir = osp.join(output_dir, 'observations')
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.obs_dir, exist_ok=True)
        
        # Initialize XARM
        self.xarm_ip = xarm_ip
        self.xarm = None
        self.cameras = []
        self.camera_indices = camera_indices
        
        # Thread control
        self.stop_event = threading.Event()
        self.action_queue = []
        self.action_lock = threading.Lock()
        
        # Initialize hardware
        self._init_xarm()
        self._init_cameras()
        
        logger.info(f"XarmRealRunner initialized: {eval_episodes} episodes, "
                   f"control_fps={control_fps}, inference_fps={inference_fps}")
    
    def _init_xarm(self):
        """Initialize XARM robot"""
        if not XARM_AVAILABLE:
            logger.warning("XARM SDK not available, skipping robot initialization")
            return
            
        try:
            self.xarm = XArmAPI(self.xarm_ip)
            self.xarm.connect()
            
            # Enable robot
            self.xarm.motion_enable(enable=True)
            self.xarm.set_mode(0)  # Position control mode
            self.xarm.set_state(state=0)  # Motion state
            
            # Set safety parameters
            self.xarm.set_tcp_maxacc(1000)  # Maximum acceleration
            self.xarm.set_tcp_jerk(10000)   # Maximum jerk
            
            logger.info(f"XARM connected successfully: {self.xarm_ip}")
            
        except Exception as e:
            logger.error(f"Failed to initialize XARM: {e}")
            self.xarm = None
    
    def _init_cameras(self):
        """Initialize cameras"""
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
        """Get current robot state"""
        if self.xarm is None:
            # Simulation state
            return {
                'robot_eef_pose': np.random.randn(6),  # xyz + rpy
                'robot_joint': np.random.randn(7),     # 7 joint angles
                'robot_joint_vel': np.random.randn(7), # 7 joint velocities
                'gripper': np.random.rand(1) * 850     # Gripper opening
            }
        
        try:
            # Get TCP pose
            ret, tcp_pose = self.xarm.get_position()
            if ret != 0:
                tcp_pose = [0, 0, 0, 0, 0, 0]
            
            # Get joint angles
            ret, joint_angles = self.xarm.get_servo_angle()
            if ret != 0:
                joint_angles = [0] * 7
            
            # Get joint velocities (if supported)
            joint_velocities = [0] * 7  # XARM may not directly provide velocity
            
            # Get gripper state
            ret, gripper_pos = self.xarm.get_gripper_position()
            if ret != 0:
                gripper_pos = 0
            
            return {
                'robot_eef_pose': np.array(tcp_pose, dtype=np.float32),
                'robot_joint': np.array(joint_angles, dtype=np.float32) * np.pi / 180,  # Convert to radians
                'robot_joint_vel': np.array(joint_velocities, dtype=np.float32),
                'gripper': np.array([gripper_pos], dtype=np.float32)
            }
            
        except Exception as e:
            logger.error(f"Error getting robot state: {e}")
            return self._get_robot_state()  # Return simulation state
    
    def _capture_images(self) -> Dict[str, np.ndarray]:
        """Capture camera images"""
        images = {}
        
        for i, cap in enumerate(self.cameras):
            try:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (224, 224))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = frame.astype(np.float32) / 255.0
                    frame = frame.transpose(2, 0, 1)  # HWC -> CHW

                    if i == 0:
                        images['agentview_image'] = frame
                        images['agentview0_image'] = frame  # 可选：主视角也叫 agentview0_image
                    elif i == 1:
                        images['agentview1_image'] = frame
                    else:
                        images[f'camera_{i}'] = frame
                else:
                    logger.warning(f"Failed to capture from camera {i}")
                    images['agentview_image'] = np.zeros((3, 224, 224), dtype=np.float32)
            except Exception as e:
                logger.error(f"Error capturing from camera {i}: {e}")
                images['agentview_image'] = np.zeros((3, 224, 224), dtype=np.float32)
        
        return images
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get complete observation"""
        # Get robot state
        robot_state = self._get_robot_state()
        
        # Get images
        images = self._capture_images()
        
        # Merge observations
        obs = {**robot_state, **images}
        return obs
    
    def _execute_action(self, action: np.ndarray):
        """Execute action"""
        if self.xarm is None:
            logger.debug(f"Simulated action: {action}")
            time.sleep(0.01)  # Simulate execution time
            return
        
        try:
            # Parse action (assuming 7D: xyz + rpy + gripper)
            if len(action) >= 7:
                xyz = action[:3] * 1000  # Convert to mm
                rpy = action[3:6] * 180 / np.pi  # Convert to degrees
                gripper = action[6] if len(action) > 6 else 0
                
                # Safety check
                xyz = np.clip(xyz, self.position_limits[0], self.position_limits[1])
                
                # Execute TCP motion
                pose = list(xyz) + list(rpy)
                ret = self.xarm.set_position(*pose, 
                                           speed=self.velocity_limit, 
                                           wait=False)
                
                if ret != 0:
                    logger.warning(f"XARM motion command failed: {ret}")
                
                # Execute gripper action
                if len(action) > 6:
                    gripper_pos = np.clip(gripper, 0, 850)
                    self.xarm.set_gripper_position(gripper_pos, wait=False)
                    
        except Exception as e:
            logger.error(f"Error executing action: {e}")
    
    def _action_control_thread(self, policy):
        """Action control thread"""
        step_count = 0
        
        while not self.stop_event.is_set():
            start_time = time.time()
            
            # Check if there are new actions
            with self.action_lock:
                if self.action_queue:
                    action = self.action_queue.pop(0)
                    self._execute_action(action)
            
            # Maintain control frequency
            elapsed = time.time() - start_time
            sleep_time = max(0, self.control_interval - elapsed)
            time.sleep(sleep_time)
            step_count += 1
    
    def _collect_observation_sequence(self) -> Dict[str, np.ndarray]:
        """Collect observation sequence"""
        obs_sequence = []
        
        for _ in range(self.n_obs_steps):
            obs = self._get_observation()
            obs_sequence.append(obs)
            time.sleep(0.1)  # Brief interval
        
        # Convert to batch format
        batch_obs = {}
        for key in obs_sequence[0].keys():
            values = [obs[key] for obs in obs_sequence]
            batch_obs[key] = np.stack(values, axis=0)
        
        return batch_obs
    
    def run(self, policy: DiffusionUnetImagePolicy) -> Dict[str, Any]:
        """Run complete evaluation"""
        logger.info(f"Starting XARM evaluation with {self.eval_episodes} episodes")
        
        policy.eval()
        device = policy.device
        
        all_episode_results = []
        
        try:
            for episode_idx in range(self.eval_episodes):
                logger.info(f"Episode {episode_idx + 1}/{self.eval_episodes}")
                
                # Reset environment (manual or automatic)
                self._reset_episode()
                
                # 启动Action control thread
                self.stop_event.clear()
                self.action_queue = []
                
                action_thread = threading.Thread(
                    target=self._action_control_thread, 
                    args=(policy,), 
                    daemon=True
                )
                action_thread.start()
                
                # Run single episode
                episode_result = self._run_episode(policy, device, episode_idx)
                all_episode_results.append(episode_result)
                
                # Stop control thread
                self.stop_event.set()
                action_thread.join(timeout=2.0)
                
                logger.info(f"Episode {episode_idx + 1} completed: "
                           f"reward={episode_result['reward']:.3f}, "
                           f"success={episode_result['success']}")
        
        finally:
            self._cleanup()
        
        # Calculate statistics
        results = self._compute_statistics(all_episode_results)
        logger.info("Evaluation completed!")
        self._log_results(results)
        
        return results
    
    def _reset_episode(self):
        """Reset episode environment"""
        if self.xarm is not None:
            try:
                # Move to safe position
                self.xarm.set_position(400, 0, 300, 0, 0, 0, speed=100, wait=True)
                # Open gripper
                self.xarm.set_gripper_position(850, wait=True)
                logger.info("XARM reset to initial position")
            except Exception as e:
                logger.error(f"Error resetting XARM: {e}")
        
        # User confirmation
        input("Press Enter when environment is ready...")
    
    def _run_episode(self, policy, device, episode_idx) -> Dict[str, Any]:
        """Run single episode"""
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
                
                # Check time limit
                if current_time - episode_start_time > self.max_duration_time:
                    logger.info("Episode timeout")
                    break
                
                # Inference frequency control
                if current_time - last_inference_time >= self.inference_interval:
                    # Collect observations
                    obs_dict = self._collect_observation_sequence()
                    
                    # Convert to tensor
                    obs_tensor = {}
                    for key, value in obs_dict.items():
                        obs_tensor[key] = torch.from_numpy(value).float().unsqueeze(0).to(device)
                    
                    # Policy inference
                    with torch.no_grad():
                        action_dict = policy.predict_action(obs_tensor)
                    
                    # Extract action sequence
                    actions = action_dict['action'][0].cpu().numpy()  # (action_horizon, action_dim)
                    
                    # Add actions to queue
                    with self.action_lock:
                        # Only take first few action steps
                        for i in range(min(self.n_action_steps, len(actions))):
                            self.action_queue.append(actions[i])
                    
                    # Save data
                    if self.save_obs:
                        episode_obs.append(obs_dict)
                        episode_actions.append(actions)
                    
                    # Simple reward calculation (can be customized for tasks)
                    step_reward = self._compute_reward(obs_dict)
                    episode_reward += step_reward
                    
                    # Simple success judgment (can be customized for tasks)
                    if step_reward > 0.8:
                        success = True
                        break
                    
                    last_inference_time = current_time
                    step_count += 1
                
                time.sleep(0.01)  # Brief sleep
                
        except KeyboardInterrupt:
            logger.warning("Episode interrupted by user")
        
        # Save episode data
        if self.save_obs:
            self._save_episode_data(episode_idx, episode_obs, episode_actions)
        
        return {
            'reward': episode_reward,
            'success': success,
            'steps': step_count,
            'duration': time.time() - episode_start_time
        }
    
    def _compute_reward(self, obs_dict: Dict[str, np.ndarray]) -> float:
        """Calculate reward (needs to be customized for specific tasks)"""
        # This is a simple example
        # Can be calculated based on task completion, distance to target, etc.
        
        # Example: Calculate reward based on end-effector position
        eef_pose = obs_dict['robot_eef_pose'][-1]  # Latest pose
        target_position = np.array([0.5, 0.0, 0.3])  # Example target position
        
        distance = np.linalg.norm(eef_pose[:3] - target_position)
        reward = max(0, 1.0 - distance)  # Higher reward for closer distance
        
        return reward
    
    def _save_episode_data(self, episode_idx: int, obs_list: list, action_list: list):
        """Save episode data"""
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
        """Calculate statistics"""
        if not episode_results:
            return {}
        
        rewards = [r['reward'] for r in episode_results]
        successes = [r['success'] for r in episode_results]
        durations = [r['duration'] for r in episode_results]
        
        return {
            'train_mean_score': np.mean(rewards),
            'train_std_score': np.std(rewards),
            'test_mean_score': np.mean(rewards),  # Same as training set
            'test_std_score': np.std(rewards),
            'train_success_rate': np.mean(successes),
            'test_success_rate': np.mean(successes),
            'mean_duration': np.mean(durations),
            'total_episodes': len(episode_results)
        }
    
    def _log_results(self, results: Dict[str, float]):
        """Log results"""
        logger.info("\n📊 XARM Evaluation Results:")
        for key, value in results.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.3f}")
            else:
                logger.info(f"  {key}: {value}")
    
    def _cleanup(self):
        """Clean up resources"""
        # Stop threads
        self.stop_event.set()
        
        # Close cameras
        for cap in self.cameras:
            cap.release()
        
        # Disconnect XARM
        if self.xarm is not None:
            try:
                self.xarm.disconnect()
                logger.info("XARM disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting XARM: {e}")
