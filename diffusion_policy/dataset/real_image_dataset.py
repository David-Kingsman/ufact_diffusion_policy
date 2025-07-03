import zarr
import torch
import numpy as np
from typing import Dict, List
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer

class RealImageDataset(BaseImageDataset):
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
                 # Add relative action parameters
                 use_relative_action: bool = False,
                 relative_type: str = "6D",
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
        
        # Relative action configuration
        self.use_relative_action = use_relative_action
        self.relative_type = relative_type
        
        # Disable absolute action if using relative action
        if use_relative_action:
            self.abs_action = False

        # Load Zarr data
        print(f"Loading real XARM data from Zarr: {dataset_path}")
        if use_relative_action:
            print(f"Use relative action, type: {relative_type}")
        else:
            print(f"Using absolute action mode")

        self.episodes = []
        self.episode_ends = []

        # Open Zarr store
        self.store = zarr.DirectoryStore(dataset_path)
        self.root = zarr.group(store=self.store)
        
        if 'data' not in self.root:
            raise ValueError(f"No 'data' group found in Zarr file: {dataset_path}")
        
        data_group = self.root['data']
        demo_keys = [k for k in data_group.keys() if k.startswith('demo_')]
        demo_keys.sort()  # Ensure order

        cumulative_length = 0
        for demo_key in demo_keys:
            demo = data_group[demo_key]
            episode_length = demo['actions'].shape[0]
            
            if episode_length >= min_episode_length:
                # Load image data (T, H, W, C) -> (T, C, H, W)
                images = demo['obs']['agentview_image'][:]
                images = torch.from_numpy(images).float() / 255.0  # Normalize to [0,1]
                if images.ndim == 4:  # (T, H, W, C)
                    images = images.permute(0, 3, 1, 2)  # -> (T, C, H, W)

                # Load low-dimensional observation data
                robot_eef_pose = torch.from_numpy(demo['obs']['robot_eef_pose'][:]).float()
                robot_joint = torch.from_numpy(demo['obs']['robot_joint'][:]).float()
                robot_joint_vel = torch.from_numpy(demo['obs']['robot_joint_vel'][:]).float()
                gripper = torch.from_numpy(demo['obs']['gripper'][:]).float()

                # Ensure gripper is 2D tensor
                if gripper.ndim == 1:
                    gripper = gripper.unsqueeze(-1)

                # Load action data
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

        # Create index mapping
        self.episode_ends = np.array(self.episode_ends)
        self.total_samples = cumulative_length
    
    def abs2relative(self, ee_gripper_data, type="6D"):
        """
        Convert absolute actions to relative actions
        Args:
            ee_gripper_data: numpy array [T, 7] (x,y,z,rx,ry,rz,gripper)
            type: "6D" (position+rotation) or "pos" (position only)
        Returns:
            relative_pose: numpy array [T, 7] relative action
        """
        from scipy.spatial.transform import Rotation
        
        if type == "6D":
            # Use first action as reference
            initial_xyz = ee_gripper_data[0, :3]
            initial_axis_angle = ee_gripper_data[0, 3:6]
            initial_quat = Rotation.from_rotvec(initial_axis_angle)
            
            # Calculate relative position
            relative_pose = ee_gripper_data.copy()
            relative_pose[:, :3] -= initial_xyz
            
            # Calculate relative rotation
            for i in range(relative_pose.shape[0]):
                abs_axis_angle = ee_gripper_data[i, 3:6]
                abs_quat = Rotation.from_rotvec(abs_axis_angle)
                
                quat_diff = abs_quat * initial_quat.inv()
                relative_pose[i, 3:6] = quat_diff.as_rotvec()
            
            # Gripper state remains unchanged
            
        elif type == "pos":
            # Only process position
            initial_xyz = ee_gripper_data[0, :3]
            relative_pose = ee_gripper_data.copy()
            relative_pose[:, :3] -= initial_xyz
        else:
            raise NotImplementedError(f"Relative action type {type} not implemented")

        return relative_pose
    
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        print("Computing normalizer from real Zarr data...")
        
        # collect all actions and observations
        all_actions = []
        all_obs = {
            'robot_eef_pose': [],
            'robot_joint': [],
            'robot_joint_vel': [],
            'gripper': []
        }
        
        # if use_relative_action:
        if self.use_relative_action:
            print(f"Computing normalization statistics for relative actions, type: {self.relative_type}")

            # Sample relative actions for statistics
            for episode in self.episodes:
                episode_actions = episode['actions'].numpy()
                episode_length = len(episode_actions)
                
                # Sample multiple times per episode
                for _ in range(min(50, episode_length - self.horizon)):
                    start_idx = np.random.randint(0, episode_length - self.horizon + 1)
                    action_chunk = episode_actions[start_idx:start_idx + self.horizon]
                    
                    # Convert to relative action
                    relative_actions = self.abs2relative(action_chunk, type=self.relative_type)
                    all_actions.append(torch.from_numpy(relative_actions).float())
                
                # Observation data unchanged
                for key in all_obs.keys():
                    all_obs[key].append(episode[key])
        else:
            # Absolute action processing
            for episode in self.episodes:
                all_actions.append(episode['actions'])
                for key in all_obs.keys():
                    all_obs[key].append(episode[key])
        
        # Merge data
        all_actions = torch.cat(all_actions, dim=0)
        for key in all_obs.keys():
            all_obs[key] = torch.cat(all_obs[key], dim=0)
        
        # Create normalizer
        normalizer = LinearNormalizer()

        # Action normalization parameters (considering different statistical properties of relative actions)
        if self.use_relative_action:
            # Relative actions usually have smaller ranges, use mean and std normalization
            action_mean = all_actions.mean(dim=0)
            action_std = all_actions.std(dim=0)
            action_std = torch.clamp(action_std, min=1e-8)
            
            normalizer.params_dict['action'] = torch.nn.ParameterDict({
                'min': torch.nn.Parameter(action_mean - 3 * action_std, requires_grad=False),
                'max': torch.nn.Parameter(action_mean + 3 * action_std, requires_grad=False),
                'scale': torch.nn.Parameter(1.0 / action_std, requires_grad=False),
                'offset': torch.nn.Parameter(-action_mean / action_std, requires_grad=False)
            })
        else:
            # Absolute actions use min-max normalization
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
        
        # Image observations (no normalization needed, already in [0,1] range)
        normalizer.params_dict['agentview_image'] = torch.nn.ParameterDict({
            'min': torch.nn.Parameter(torch.zeros(1), requires_grad=False),
            'max': torch.nn.Parameter(torch.ones(1), requires_grad=False),
            'scale': torch.nn.Parameter(torch.ones(1), requires_grad=False),
            'offset': torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        })
        
        # Low-dimensional observation normalization parameters
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

        action_type = "Relative Action" if self.use_relative_action else "Absolute Action"
        print(f"Zarr {action_type} Normalizer computation completed!")
        return normalizer
        
    def get_all_actions(self) -> torch.Tensor:
        all_actions = []
        for episode in self.episodes:
            # Create sliding window action sequences
            actions = episode['actions']
            for start_idx in range(len(actions) - self.horizon + 1):
                action_sequence = actions[start_idx:start_idx + self.horizon]
                
                # Convert if using relative action
                if self.use_relative_action:
                    action_sequence = torch.from_numpy(
                        self.abs2relative(action_sequence.numpy(), type=self.relative_type)
                    ).float()
                
                all_actions.append(action_sequence)
        return torch.stack(all_actions)
        
    def _get_episode_index(self, idx):
        """Get the corresponding episode and episode index based on the global index"""
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
        
        # Get observation sequence (n_obs_steps)
        obs_start = start_idx
        obs_end = start_idx + self.n_obs_steps
        
        # Get action sequence (horizon)
        action_start = start_idx
        action_end = start_idx + self.horizon
        
        # Get action data
        actions = episode['actions'][action_start:action_end]
        
        # Convert if using relative action
        if self.use_relative_action:
            actions = torch.from_numpy(
                self.abs2relative(actions.numpy(), type=self.relative_type)
            ).float()
        
        batch = {
            'obs': {
                'agentview_image': episode['agentview_image'][obs_start:obs_end],
                'robot_eef_pose': episode['robot_eef_pose'][obs_start:obs_end],
                'robot_joint': episode['robot_joint'][obs_start:obs_end],
                'robot_joint_vel': episode['robot_joint_vel'][obs_start:obs_end],
                'gripper': episode['gripper'][obs_start:obs_end]
            },
            'action': actions
        }
        return batch
        
    def get_validation_dataset(self):
        # Create validation dataset (using last episodes)
        val_dataset = RealImageDataset(
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
            val_ratio=self.val_ratio,
            # Pass relative action parameters
            use_relative_action=self.use_relative_action,
            relative_type=self.relative_type
        )
        
        # Keep only part as validation set
        num_val_episodes = max(1, int(len(self.episodes) * self.val_ratio))
        val_dataset.episodes = self.episodes[-num_val_episodes:]
        
        # Recalculate validation set indices
        cumulative_length = 0
        val_dataset.episode_ends = []
        for episode in val_dataset.episodes:
            episode_length = len(episode['actions'])
            cumulative_length += episode_length - self.horizon + 1
            val_dataset.episode_ends.append(cumulative_length)
        
        val_dataset.episode_ends = np.array(val_dataset.episode_ends)
        val_dataset.total_samples = cumulative_length

        action_type = "Relative Action" if self.use_relative_action else "Absolute Action"
        print(f"Validation Dataset ({action_type}): {len(val_dataset.episodes)} episodes, {cumulative_length} samples")
        return val_dataset
        
    def __del__(self):
        # Clean up Zarr resources
        if hasattr(self, 'store'):
            try:
                self.store.close()
            except:
                pass