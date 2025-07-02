import os
import json
import numpy as np
import h5py
from pathlib import Path
import cv2
from PIL import Image
import argparse
from tqdm import tqdm
import zarr
from loguru import logger
from scipy.spatial.transform import Rotation

class MetaQuestDataProcessor:
    def __init__(self, input_dir="demos_collected", output_format="hdf5", output_file="data/metaquest_dataset.hdf5"):
        self.input_dir = Path(input_dir)
        self.output_format = output_format
        
        if output_format == "zarr":
            self.output_file = Path(output_file.replace('.hdf5', '.zarr'))
        else:
            self.output_file = Path(output_file)
            
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 配置参数
        self.image_size = (224, 224)
        self.temporal_downsample_ratio = 1
        self.use_absolute_action = True
        
    def load_run_data(self, run_dir):
        """加载Meta Quest采集的单个run数据"""
        run_data = {}
        
        # 加载配置文件
        config_file = run_dir / "config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                run_data['config'] = json.load(f)
        
        # 加载Meta Quest格式的npz文件
        npz_files = {
            'action': 'demo_action.npz',
            'ee_states': 'demo_ee_states.npz', 
            'target_pose_mat': 'demo_target_pose_mat.npz',
            'joint_states': 'demo_joint_states.npz',
            'gripper_states': 'demo_gripper_states.npz',
            'action_hot': 'demo_action_hot.npz'
        }
        
        # 动态查找相机文件
        camera_files = list(run_dir.glob("demo_camera_*.npz"))
        for camera_file in camera_files:
            # 提取相机ID：demo_camera_1.npz -> camera_1
            camera_id = camera_file.stem.replace('demo_', '')
            npz_files[camera_id] = camera_file.name
        
        for key, filename in npz_files.items():
            file_path = run_dir / filename
            if file_path.exists():
                data = np.load(file_path, allow_pickle=True)
                # Meta Quest数据都存储在'data'键下
                if 'data' in data:
                    run_data[key] = data['data']
                else:
                    # 备选方案，如果没有'data'键
                    if len(data.files) == 1:
                        run_data[key] = data[data.files[0]]
                    else:
                        run_data[key] = dict(data)
            else:
                logger.warning(f"{filename} not found in {run_dir}")
        
        return run_data

    def load_images_for_run(self, run_dir, camera_data):
        """加载Meta Quest的图像数据 - 修改版本"""
        # 首先检查run目录内的images
        images_dir = run_dir / "images"
        
        if not images_dir.exists():
            # 如果run目录内没有images，检查父目录的images
            parent_images_dir = run_dir.parent / "images"
            if parent_images_dir.exists():
                logger.info(f"Using parent images directory: {parent_images_dir}")
                images_dir = parent_images_dir
            else:
                logger.warning(f"No images directory found for {run_dir}")
                return None
        
        # 查找图像文件夹 - 移动到这里
        img_folders = list(images_dir.glob("rs_rs_1_*"))
        if not img_folders:
            logger.warning(f"No image folders found in {images_dir}")
            return None
        
        # 选择第一个图像文件夹
        img_folder = img_folders[0]
        logger.info(f"Loading images from {img_folder}")
        
        images = []
        
        # 根据相机数据的长度确定图像数量
        if camera_data is not None and len(camera_data) > 0:
            num_images = len(camera_data)
            logger.info(f"Expected {num_images} images based on camera data")
            
            for i in range(num_images):
                # Meta Quest的图像命名格式：color_000000001.jpg
                img_path = img_folder / f"color_{i+1:09d}.jpg"
                
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, self.image_size)
                        images.append(img)
                    else:
                        logger.warning(f"Failed to read image {img_path}")
                else:
                    logger.warning(f"Image not found: {img_path}")
        
        if not images:
            logger.warning(f"No images loaded for run {run_dir.name}")
            return None
            
        return np.array(images)
    
    def process_meta_quest_observations(self, run_data, images):
        """处理Meta Quest的观测数据"""
        obs = {}
        
        # 处理末端执行器状态 - Meta Quest直接提供6D pose
        if 'ee_states' in run_data:
            ee_data = run_data['ee_states']
            if ee_data.ndim == 2:
                # 假设格式为 [x, y, z, rx, ry, rz] (欧拉角度制)
                obs['robot_eef_pose'] = ee_data
            else:
                logger.warning(f"Unexpected ee_states shape: {ee_data.shape}")
                obs['robot_eef_pose'] = ee_data
        
        # 处理关节状态
        if 'joint_states' in run_data:
            joint_data = run_data['joint_states']
            obs['robot_joint'] = joint_data
            # 创建零速度数据（Meta Quest不采集关节速度）
            obs['robot_joint_vel'] = np.zeros_like(joint_data)
        
        # 处理夹爪状态
        if 'gripper_states' in run_data:
            obs['gripper'] = run_data['gripper_states']
        
        # 处理图像数据
        if images is not None:
            obs['agentview_image'] = images
        
        # 处理动作热编码作为标签
        if 'action_hot' in run_data:
            obs['label'] = run_data['action_hot']
        else:
            # 如果没有action_hot，创建默认标签
            seq_len = self._get_sequence_length(obs)
            obs['label'] = np.zeros(seq_len)
        
        return obs
    
    def process_meta_quest_actions(self, run_data):
        """处理Meta Quest的动作数据"""
        if 'action' in run_data:
            actions = run_data['action']
            
            # Meta Quest的action格式：[x, y, z, rx, ry, rz, grasp]
            if actions.ndim == 2 and actions.shape[1] == 7:
                # 分离位置/姿态和夹爪
                pose_actions = actions[:, :6]  # [x, y, z, rx, ry, rz]
                gripper_actions = actions[:, 6:7]  # [grasp]
                
                # 组合为最终动作
                processed_actions = np.concatenate([pose_actions, gripper_actions], axis=1)
                return processed_actions
            else:
                logger.warning(f"Unexpected action shape: {actions.shape}")
                return actions
        else:
            logger.warning("No action data found")
            return np.array([[]])
    
    def _get_sequence_length(self, obs):
        """获取观测序列的长度"""
        for key, value in obs.items():
            if isinstance(value, np.ndarray) and len(value.shape) > 0:
                return len(value)
        return 0
    
    def create_robomimic_hdf5(self):
        """创建适配Meta Quest数据的robomimic格式HDF5文件"""
        run_dirs = sorted([d for d in self.input_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('run')])
        
        if not run_dirs:
            logger.error("No run directories found!")
            return
        
        logger.info(f"Found {len(run_dirs)} runs to process")
        
        with h5py.File(self.output_file, 'w') as f:
            data_grp = f.create_group("data")
            
            valid_demos = 0
            
            for run_idx, run_dir in enumerate(tqdm(run_dirs, desc="Processing runs")):
                logger.info(f"Processing {run_dir.name}...")
                
                run_data = self.load_run_data(run_dir)
                if not run_data:
                    logger.warning(f"Skipping {run_dir.name} - no data loaded")
                    continue
                
                # 查找相机数据
                camera_data = None
                for key in run_data.keys():
                    if key.startswith('camera_'):
                        camera_data = run_data[key]
                        break
                
                images = self.load_images_for_run(run_dir, camera_data)
                obs = self.process_meta_quest_observations(run_data, images)
                actions = self.process_meta_quest_actions(run_data)
                
                if len(actions) == 0:
                    logger.warning(f"Skipping {run_dir.name} - no valid actions")
                    continue
                
                # 确保观测和动作长度一致
                min_length = min(len(actions), self._get_sequence_length(obs))
                if min_length == 0:
                    logger.warning(f"Skipping {run_dir.name} - zero length sequence")
                    continue
                
                # 截断到一致长度
                actions = actions[:min_length]
                for key in obs:
                    if isinstance(obs[key], np.ndarray) and len(obs[key]) > min_length:
                        obs[key] = obs[key][:min_length]
                
                # 创建demonstration组
                demo_name = f"demo_{valid_demos}"
                demo_grp = data_grp.create_group(demo_name)
                
                # 保存观测数据
                obs_grp = demo_grp.create_group("obs")
                for obs_key, obs_data in obs.items():
                    if obs_data is not None and len(obs_data) > 0:
                        obs_grp.create_dataset(obs_key, data=obs_data)
                
                # 保存动作数据
                demo_grp.create_dataset("actions", data=actions)
                
                # 保存其他必要的数据
                dones = np.zeros(len(actions), dtype=bool)
                dones[-1] = True
                demo_grp.create_dataset("dones", data=dones)
                demo_grp.create_dataset("rewards", data=np.zeros(len(actions)))
                
                logger.info(f"Saved {len(actions)} timesteps for {demo_name}")
                valid_demos += 1
            
            # 添加元数据
            f.attrs["type"] = "low_dim"
            f.attrs["env"] = "xarm_metaquest"
            f.attrs["created_by"] = "MetaQuestDataProcessor"
            
            logger.info(f"HDF5 dataset created successfully: {self.output_file}")
            logger.info(f"Total valid demonstrations: {valid_demos}")
    
    def process_data(self):
        """处理数据的主入口"""
        if self.output_format == "zarr":
            logger.error("Zarr format not implemented for Meta Quest data yet")
            raise NotImplementedError("Use HDF5 format for now")
        else:
            self.create_robomimic_hdf5()

def main():
    parser = argparse.ArgumentParser(description="Convert Meta Quest demo data to training format")
    parser.add_argument("--input_dir", type=str, default="demos_collected", 
                       help="Input directory containing Meta Quest demo data")
    parser.add_argument("--output_file", type=str, default="data/metaquest_dataset.hdf5",
                       help="Output file path")
    parser.add_argument("--output_format", type=str, choices=["hdf5"], 
                       default="hdf5", help="Output format (only HDF5 supported)")
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224],
                       help="Image resize dimensions")
    
    args = parser.parse_args()
    
    processor = MetaQuestDataProcessor(
        input_dir=args.input_dir, 
        output_format=args.output_format,
        output_file=args.output_file
    )
    processor.image_size = tuple(args.image_size)
    
    processor.process_data()

if __name__ == "__main__":
    main()