import h5py
import torch
import numpy as np
from typing import Dict, List
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer

class XarmHDF5Dataset(BaseImageDataset):
    def __init__(self, 
                 dataset_path: str,
                 horizon: int = 16,
                 pad_before: int = 1,
                 pad_after: int = 7,
                 n_obs_steps: int = 2,
                 abs_action: bool = True,
                 rotation_rep: str = 'axis_angle',
                 use_legacy_normalizer: bool = False,
                 min_episode_length: int = 10,
                 image_keys: List[str] = None,
                 val_ratio: float = 0.15,
                 **kwargs):
        
        self.dataset_path = dataset_path
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.abs_action = abs_action
        self.rotation_rep = rotation_rep
        self.use_legacy_normalizer = use_legacy_normalizer
        self.min_episode_length = min_episode_length
        self.image_keys = image_keys or ['agentview_image']
        self.val_ratio = val_ratio
        
        # 加载真实的HDF5数据
        print(f"Loading real XARM data from: {dataset_path}")
        self.episodes = []
        self.episode_ends = []
        
        with h5py.File(dataset_path, 'r') as f:
            demo_keys = [k for k in f['data'].keys() if k.startswith('demo_')]
            demo_keys.sort()  # 确保顺序
            
            cumulative_length = 0
            for demo_key in demo_keys:
                demo = f['data'][demo_key]
                episode_length = demo['actions'].shape[0]
                
                if episode_length >= min_episode_length:
                    # 加载图像数据 (T, H, W, C) -> (T, C, H, W)
                    images = demo['obs']['agentview_image'][:]
                    images = torch.from_numpy(images).float() / 255.0  # 归一化到[0,1]
                    images = images.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
                    
                    # 加载低维观测数据
                    robot_eef_pose = torch.from_numpy(demo['obs']['robot_eef_pose'][:]).float()
                    robot_joint = torch.from_numpy(demo['obs']['robot_joint'][:]).float()
                    robot_joint_vel = torch.from_numpy(demo['obs']['robot_joint_vel'][:]).float()
                    gripper = torch.from_numpy(demo['obs']['gripper'][:]).float().unsqueeze(-1)
                    
                    # 加载动作数据
                    actions = torch.from_numpy(demo['actions'][:]).float()
                    
                    episode_data = {
                        'agentview_image': images,
                        'robot_eef_pose': robot_eef_pose,
                        'robot_joint': robot_joint,
                        'robot_joint_vel': robot_joint_vel,
                        'gripper': gripper,
                        'actions': actions
                    }
                    
                    self.episodes.append(episode_data)
                    cumulative_length += episode_length - horizon + 1
                    self.episode_ends.append(cumulative_length)
                    
                    print(f"  {demo_key}: {episode_length} steps")
        
        print(f"Loaded {len(self.episodes)} episodes, {cumulative_length} training samples")
        
        # 创建索引映射
        self.episode_ends = np.array(self.episode_ends)
        self.total_samples = cumulative_length
        
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        print("Computing normalizer from real data...")
        
        # 收集所有动作和观测数据
        all_actions = []
        all_obs = {
            'robot_eef_pose': [],
            'robot_joint': [],
            'robot_joint_vel': [],
            'gripper': []
        }
        
        for episode in self.episodes:
            all_actions.append(episode['actions'])
            for key in all_obs.keys():
                all_obs[key].append(episode[key])
        
        # 合并数据
        all_actions = torch.cat(all_actions, dim=0)
        for key in all_obs.keys():
            all_obs[key] = torch.cat(all_obs[key], dim=0)
        
        # 创建归一化器
        normalizer = LinearNormalizer()
        
        # 动作归一化参数
        action_min = all_actions.min(dim=0)[0]
        action_max = all_actions.max(dim=0)[0]
        action_range = action_max - action_min
        action_range = torch.clamp(action_range, min=1e-8)
        
        normalizer.params_dict['action'] = torch.nn.ParameterDict({
            'min': torch.nn.Parameter(action_min, requires_grad=False),
            'max': torch.nn.Parameter(action_max, requires_grad=False),
            'scale': torch.nn.Parameter(2.0 / action_range, requires_grad=False),
            'offset': torch.nn.Parameter(-(action_max + action_min) / action_range, requires_grad=False)
        })
        
        # 图像观测（不需要归一化，已经在[0,1]范围）
        normalizer.params_dict['agentview_image'] = torch.nn.ParameterDict({
            'min': torch.nn.Parameter(torch.zeros(1), requires_grad=False),
            'max': torch.nn.Parameter(torch.ones(1), requires_grad=False),
            'scale': torch.nn.Parameter(torch.ones(1), requires_grad=False),
            'offset': torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        })
        
        # 低维观测归一化参数
        for key, obs_data in all_obs.items():
            obs_min = obs_data.min(dim=0)[0]
            obs_max = obs_data.max(dim=0)[0]
            obs_range = obs_max - obs_min
            obs_range = torch.clamp(obs_range, min=1e-8)
            
            normalizer.params_dict[key] = torch.nn.ParameterDict({
                'min': torch.nn.Parameter(obs_min, requires_grad=False),
                'max': torch.nn.Parameter(obs_max, requires_grad=False),
                'scale': torch.nn.Parameter(2.0 / obs_range, requires_grad=False),
                'offset': torch.nn.Parameter(-(obs_max + obs_min) / obs_range, requires_grad=False)
            })
        
        print("Normalizer computed successfully!")
        return normalizer
        
    def get_all_actions(self) -> torch.Tensor:
        all_actions = []
        for episode in self.episodes:
            # 创建滑动窗口的动作序列
            actions = episode['actions']
            for start_idx in range(len(actions) - self.horizon + 1):
                action_sequence = actions[start_idx:start_idx + self.horizon]
                all_actions.append(action_sequence)
        return torch.stack(all_actions)
        
    def _get_episode_index(self, idx):
        """根据全局索引获取对应的episode和episode内索引"""
        episode_idx = np.searchsorted(self.episode_ends, idx + 1)
        if episode_idx == 0:
            start_idx = idx
        else:
            start_idx = idx - self.episode_ends[episode_idx - 1]
        return episode_idx, start_idx
        
    def __len__(self):
        return self.total_samples
        
    def __getitem__(self, idx):
        episode_idx, start_idx = self._get_episode_index(idx)
        episode = self.episodes[episode_idx]
        
        # 获取观测序列 (n_obs_steps)
        obs_start = start_idx
        obs_end = start_idx + self.n_obs_steps
        
        # 获取动作序列 (horizon)
        action_start = start_idx
        action_end = start_idx + self.horizon
        
        batch = {
            'obs': {
                'agentview_image': episode['agentview_image'][obs_start:obs_end],
                'robot_eef_pose': episode['robot_eef_pose'][obs_start:obs_end],
                'robot_joint': episode['robot_joint'][obs_start:obs_end],
                'robot_joint_vel': episode['robot_joint_vel'][obs_start:obs_end],
                'gripper': episode['gripper'][obs_start:obs_end]
            },
            'action': episode['actions'][action_start:action_end]
        }
        return batch
        
    def get_validation_dataset(self):
        # 创建验证数据集（使用最后的episodes）
        val_dataset = XarmHDF5Dataset(
            dataset_path=self.dataset_path,
            horizon=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            n_obs_steps=self.n_obs_steps,
            abs_action=self.abs_action,
            rotation_rep=self.rotation_rep,
            use_legacy_normalizer=self.use_legacy_normalizer,
            min_episode_length=self.min_episode_length,
            image_keys=self.image_keys,
            val_ratio=self.val_ratio
        )
        
        # 只保留一部分作为验证集
        num_val_episodes = max(1, int(len(self.episodes) * self.val_ratio))
        val_dataset.episodes = self.episodes[-num_val_episodes:]
        
        # 重新计算验证集的索引
        cumulative_length = 0
        val_dataset.episode_ends = []
        for episode in val_dataset.episodes:
            episode_length = len(episode['actions'])
            cumulative_length += episode_length - self.horizon + 1
            val_dataset.episode_ends.append(cumulative_length)
        
        val_dataset.episode_ends = np.array(val_dataset.episode_ends)
        val_dataset.total_samples = cumulative_length
        
        print(f"Validation dataset: {len(val_dataset.episodes)} episodes, {cumulative_length} samples")
        return val_dataset
