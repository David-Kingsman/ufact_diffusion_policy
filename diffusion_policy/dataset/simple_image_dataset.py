import torch
import numpy as np
from typing import Dict, List
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer

class SimpleImageDataset(BaseImageDataset):
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
        
        # 生成模拟数据
        self.dummy_length = 100
        print(f"SimpleImageDataset initialized with {self.dummy_length} episodes")
        
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        # 创建所有数据
        all_data = []
        for i in range(min(10, self.dummy_length)):  # 只用少量数据避免内存问题
            batch = self.__getitem__(i)
            all_data.append(batch)
        
        # 合并所有动作数据
        all_actions = torch.stack([data['action'] for data in all_data])
        
        # 合并所有观测数据
        obs_dict = {}
        for key in all_data[0]['obs'].keys():
            obs_dict[key] = torch.stack([data['obs'][key] for data in all_data])
        
        # 创建归一化器
        normalizer = LinearNormalizer()
        
        # 为动作数据添加归一化参数
        action_flat = all_actions.reshape(-1, all_actions.shape[-1])
        action_min = action_flat.min(dim=0)[0]
        action_max = action_flat.max(dim=0)[0]
        action_range = action_max - action_min
        action_range = torch.clamp(action_range, min=1e-8)  # 避免除零
        
        normalizer.params_dict['action'] = torch.nn.ParameterDict({
            'min': torch.nn.Parameter(action_min, requires_grad=False),
            'max': torch.nn.Parameter(action_max, requires_grad=False),
            'scale': torch.nn.Parameter(2.0 / action_range, requires_grad=False),
            'offset': torch.nn.Parameter(-(action_max + action_min) / action_range, requires_grad=False)
        })
        
        # 为观测数据添加正确格式的归一化参数
        for key, obs_data in obs_dict.items():
            if key in self.image_keys:
                # 图像数据不需要归一化，设置恒等变换参数
                normalizer.params_dict[key] = torch.nn.ParameterDict({
                    'min': torch.nn.Parameter(torch.zeros(1), requires_grad=False),
                    'max': torch.nn.Parameter(torch.ones(1), requires_grad=False),
                    'scale': torch.nn.Parameter(torch.ones(1), requires_grad=False),
                    'offset': torch.nn.Parameter(torch.zeros(1), requires_grad=False)
                })
            else:
                # 低维观测数据需要真实的归一化参数
                obs_flat = obs_data.reshape(-1, obs_data.shape[-1])
                obs_min = obs_flat.min(dim=0)[0]
                obs_max = obs_flat.max(dim=0)[0]
                obs_range = obs_max - obs_min
                obs_range = torch.clamp(obs_range, min=1e-8)  # 避免除零
                
                normalizer.params_dict[key] = torch.nn.ParameterDict({
                    'min': torch.nn.Parameter(obs_min, requires_grad=False),
                    'max': torch.nn.Parameter(obs_max, requires_grad=False),
                    'scale': torch.nn.Parameter(2.0 / obs_range, requires_grad=False),
                    'offset': torch.nn.Parameter(-(obs_max + obs_min) / obs_range, requires_grad=False)
                })
        
        return normalizer
        
    def get_all_actions(self) -> torch.Tensor:
        # 生成模拟动作数据 [N, horizon, action_dim]
        return torch.randn(self.dummy_length, self.horizon, 7)
        
    def __len__(self):
        return self.dummy_length
        
    def __getitem__(self, idx):
        # 返回模拟数据批次
        batch = {
            'obs': {
                'agentview_image': torch.randn(self.n_obs_steps, 3, 224, 224),
                'robot_eef_pose': torch.randn(self.n_obs_steps, 6),
                'robot_joint': torch.randn(self.n_obs_steps, 6),
                'robot_joint_vel': torch.randn(self.n_obs_steps, 6),
                'gripper': torch.randn(self.n_obs_steps, 1)
            },
            'action': torch.randn(self.horizon, 7)
        }
        return batch
        
    def get_validation_dataset(self):
        # 返回一个较小的验证数据集
        val_dataset = SimpleImageDataset(
            dataset_path=self.dataset_path,
            horizon=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            n_obs_steps=self.n_obs_steps,
            abs_action=self.abs_action,
            rotation_rep=self.rotation_rep,
            use_legacy_normalizer=self.use_legacy_normalizer,
            min_episode_length=self.min_episode_length,
            image_keys=self.image_keys
        )
        val_dataset.dummy_length = 20  # 验证集更小
        return val_dataset
