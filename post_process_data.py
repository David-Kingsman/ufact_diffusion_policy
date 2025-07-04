import os
import json
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import argparse
from tqdm import tqdm
import zarr
from loguru import logger
from scipy.spatial.transform import Rotation

class MetaQuestDataProcessor:
    def __init__(self, input_dir="demos_collected", output_file="data/metaquest_xarm_dataset.zarr"):
        self.input_dir = Path(input_dir)
        
        # zarr output file 
        if not output_file.endswith('.zarr'):
            output_file = output_file.replace('.hdf5', '.zarr')
            if not output_file.endswith('.zarr'):
                output_file += '.zarr'
        
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # configuration parameters
        self.image_size = (224, 224)
        self.temporal_downsample_ratio = 1
        self.use_absolute_action = True
        
    def load_run_data(self, run_dir):
        """Load a single run data collected by Meta Quest"""
        run_data = {}
        
        # configuration 
        config_file = run_dir / "config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                run_data['config'] = json.load(f)
        
        # meta quest data files
        npz_files = {
            'action': 'demo_action.npz',
            'ee_states': 'demo_ee_states.npz', 
            'target_pose_mat': 'demo_target_pose_mat.npz',
            'joint_states': 'demo_joint_states.npz',
            'gripper_states': 'demo_gripper_states.npz',
            'action_hot': 'demo_action_hot.npz'
        }
  
        # camera data 
        camera_files = list(run_dir.glob("demo_camera_*.npz"))
        for camera_file in camera_files:
            # Extract camera ID：demo_camera_1.npz -> camera_1
            camera_id = camera_file.stem.replace('demo_', '')
            npz_files[camera_id] = camera_file.name
        
        for key, filename in npz_files.items():
            file_path = run_dir / filename
            if file_path.exists():
                data = np.load(file_path, allow_pickle=True)
                # ensure the data is loaded correctly
                try:
                    if 'data' in data.files:
                        run_data[key] = data['data']
                    else:
                        # no data key, directly use the loaded data
                        if len(data.files) == 1:
                            run_data[key] = data[data.files[0]]
                        else:
                            run_data[key] = dict(data)
                except Exception as e:
                    logger.warning(f"Error loading {filename}: {e}")
                    # try to handle the case where data is not structured as expected
                    if len(data.files) > 0:
                        first_key = data.files[0]
                        run_data[key] = data[first_key]
                    else:
                        run_data[key] = None

            else:
                logger.warning(f"{filename} not found in {run_dir}")
        
        return run_data

    def load_images_for_run(self, run_dir, camera_data):
        """Load images for a specific run"""
        images_dir = run_dir / "images"
        if not images_dir.exists():
            parent_images_dir = run_dir.parent / "images"
            if parent_images_dir.exists():
                logger.info(f"Using parent images directory: {parent_images_dir}")
                images_dir = parent_images_dir
            else:
                logger.warning(f"No images directory found for {run_dir}")
                return None

        # 找到图片文件夹
        img_folders = list(images_dir.glob("rs_rs_1_*"))
        if not img_folders:
            logger.warning(f"No image folders found in {images_dir}")
            return None

        img_folder = img_folders[0]
        logger.info(f"Loading images from {img_folder}")

        # 直接遍历所有 color_*.jpg 文件，按文件名排序
        img_files = sorted(img_folder.glob("color_*.jpg"))
        images = []
        for img_file in img_files:
            img = cv2.imread(str(img_file))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.image_size)
                images.append(img)
            else:
                logger.warning(f"Failed to read image {img_file}")

        if not images:
            logger.warning(f"No images loaded for run {run_dir.name}")
            return None

        return np.array(images)
    

    def load_multi_camera_images(self, run_dir, run_data):
        """加载多路相机图片，返回 dict"""
        images_dict = {}
        
        for cam_key in ['camera_0', 'camera_1']:
            if cam_key in run_data:
                cam_npz = run_data[cam_key]
                img_names = [item['color_img_name'] for item in cam_npz]
                print(f"[DEBUG] {cam_key} first 3 img_names:", img_names[:3])
                imgs = []

                for img_name in img_names:
                    img_path = Path(img_name)
                    # 如果没有扩展名，自动加上 .jpg
                    if img_path.suffix == "":
                        img_path = img_path.with_suffix(".jpg")
                    img_path = img_path.resolve()
                    if not img_path.exists():
                        print(f"[DEBUG] Image path does not exist: {img_path}")
                    else:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, self.image_size)
                            imgs.append(img)
                        else:
                            print(f"[DEBUG] cv2.imread failed for {img_path}")

                if imgs:
                    images_dict[cam_key] = np.array(imgs)
                else:
                    print(f"[DEBUG] No images loaded for {cam_key}")
        return images_dict

    def process_meta_quest_observations(self, run_data, images_dict):
        """Process Meta Quest observations"""
        obs = {}

        # Process end-effector states
        if 'ee_states' in run_data:
            ee_data = run_data['ee_states']
            if ee_data.ndim == 2:
                obs['robot_eef_pose'] = ee_data
            else:
                logger.warning(f"Unexpected ee_states shape: {ee_data.shape}")
                obs['robot_eef_pose'] = ee_data

        # Process joint states
        if 'joint_states' in run_data:
            joint_data = run_data['joint_states']
            obs['robot_joint'] = joint_data
            # Create zero velocity data
            obs['robot_joint_vel'] = np.zeros_like(joint_data)

        # Process gripper states
        if 'gripper_states' in run_data:
            obs['gripper'] = run_data['gripper_states']

        # 多路相机处理
        camera_key_to_obs_key = {
            'camera_0': 'agentview0_image',
            'camera_1': 'agentview1_image'
        }
        if images_dict is not None:
            for cam_key, imgs in images_dict.items():
                obs_key = camera_key_to_obs_key.get(cam_key, cam_key)
                obs[obs_key] = imgs
            # 默认 agentview_image 指向 agentview0_image
            if 'agentview0_image' in obs:
                obs['agentview_image'] = obs['agentview0_image']

        # Process action hot encoding as labels
        if 'action_hot' in run_data:
            obs['label'] = run_data['action_hot']
        else:
            # If no action_hot, create default labels
            seq_len = self._get_sequence_length(obs)
            obs['label'] = np.zeros(seq_len)
        
        return obs
    
    def process_meta_quest_actions(self, run_data):
        """Process Meta Quest actions"""
        if 'action' in run_data:
            actions = run_data['action']

            # Meta Quest action format: [x, y, z, rx, ry, rz, grasp]
            if actions.ndim == 2 and actions.shape[1] == 7:
                # Separate position/orientation and gripper
                pose_actions = actions[:, :6]  # [x, y, z, rx, ry, rz]
                gripper_actions = actions[:, 6:7]  # [grasp]

                # Combine into final actions
                processed_actions = np.concatenate([pose_actions, gripper_actions], axis=1)
                return processed_actions
            else:
                logger.warning(f"Unexpected action shape: {actions.shape}")
                return actions
        else:
            logger.warning("No action data found")
            return np.array([[]])
    
    def _get_sequence_length(self, obs):
        """Get the length of the observation sequence"""
        for key, value in obs.items():
            if isinstance(value, np.ndarray) and len(value.shape) > 0:
                return len(value)
        return 0
    
    def create_zarr(self):
        """Create Zarr format dataset - fixed version"""
        run_dirs = sorted([d for d in self.input_dir.iterdir() 
                        if d.is_dir() and d.name.startswith('run')])
        
        if not run_dirs:
            logger.error("No run directories found!")
            return
        
        logger.info(f"Found {len(run_dirs)} runs to process")
        
        # delete existing zarr file if it exists
        if self.output_file.exists():
            import shutil
            shutil.rmtree(self.output_file)
            logger.info(f"Removed existing zarr file: {self.output_file}")

        # Create zarr store - use NestedDirectoryStore for compatibility
        try:
            store = zarr.NestedDirectoryStore(str(self.output_file))
        except AttributeError:
            # If NestedDirectoryStore is not available, use DirectoryStore
            store = zarr.DirectoryStore(str(self.output_file))
        
        root = zarr.group(store=store, overwrite=True)
        logger.info(f"Created zarr root group: {root}")

        # Immediately add root attributes
        root.attrs["type"] = "low_dim"
        root.attrs["env"] = "xarm_metaquest"
        root.attrs["created_by"] = "MetaQuestDataProcessor"
        root.attrs["format"] = "zarr"
        root.attrs["compression"] = "gzip"

        # Create data group
        data_grp = root.create_group("data")
        logger.info(f"Created data group: {data_grp}")
        
        valid_demos = 0
        
        for run_idx, run_dir in enumerate(tqdm(run_dirs, desc="Processing runs")):
            logger.info(f"Processing {run_dir.name}...")
            
            run_data = self.load_run_data(run_dir)
            if not run_data:
                logger.warning(f"Skipping {run_dir.name} - no data loaded")
                continue

            # Find camera data
            images_dict = self.load_multi_camera_images(run_dir, run_data)
            obs = self.process_meta_quest_observations(run_data, images_dict)
            actions = self.process_meta_quest_actions(run_data)
            
            if len(actions) == 0:
                logger.warning(f"Skipping {run_dir.name} - no valid actions")
                continue

            # Ensure observation and action lengths match
            min_length = min(len(actions), self._get_sequence_length(obs))
            if min_length == 0:
                logger.warning(f"Skipping {run_dir.name} - zero length sequence")
                continue

            # Truncate to consistent length
            actions = actions[:min_length]
            for key in obs:
                if isinstance(obs[key], np.ndarray) and len(obs[key]) > min_length:
                    obs[key] = obs[key][:min_length]

            # Create demonstration group
            demo_name = f"demo_{valid_demos}"
            demo_grp = data_grp.create_group(demo_name)
            logger.info(f"Created demo group: {demo_name}")

            # Save observation data
            obs_grp = demo_grp.create_group("obs")
            for obs_key, obs_data in obs.items():
                if obs_data is not None and len(obs_data) > 0:
                    logger.info(f"  Saving {obs_key} with shape {obs_data.shape}")

                    # Simplify zarr parameters for compatibility
                    try:
                        if obs_key == 'agentview_image':
                            # Image data: use moderate chunk size and compression
                            chunks = (min(5, obs_data.shape[0]), 56, 56, 3)
                            dataset = obs_grp.create_dataset(
                                obs_key, 
                                data=obs_data,
                                chunks=chunks,
                                compression='gzip',
                                compression_opts=6
                            )
                        else:
                            # Other data: simple settings
                            dataset = obs_grp.create_dataset(
                                obs_key, 
                                data=obs_data,
                                chunks=True,
                                compression='gzip',
                                compression_opts=3
                            )
                        
                        logger.info(f"    Successfully saved {obs_key}")
                        
                    except Exception as e:
                        logger.error(f"    Failed to save {obs_key}: {e}")
                        # Fallback: save without compression
                        try:
                            dataset = obs_grp.create_dataset(obs_key, data=obs_data)
                            logger.info(f"    Saved {obs_key} without compression")
                        except Exception as e2:
                            logger.error(f"    Complete failure for {obs_key}: {e2}")

            # Save action data
            try:
                logger.info(f"  Saving actions with shape {actions.shape}")
                actions_dataset = demo_grp.create_dataset(
                    "actions", 
                    data=actions,
                    chunks=True,
                    compression='gzip',
                    compression_opts=3
                )
                logger.info("    Successfully saved actions")
            except Exception as e:
                logger.error(f"    Failed to save actions: {e}")
                # Fallback: save without compression
                actions_dataset = demo_grp.create_dataset("actions", data=actions)
                logger.info("    Saved actions without compression")

            # Save other necessary data
            try:
                dones = np.zeros(len(actions), dtype=bool)
                dones[-1] = True
                demo_grp.create_dataset("dones", data=dones)
                
                rewards = np.zeros(len(actions), dtype=np.float32)
                demo_grp.create_dataset("rewards", data=rewards)
                
                logger.info("    Successfully saved dones and rewards")
            except Exception as e:
                logger.error(f"    Failed to save dones/rewards: {e}")
            
            logger.info(f"✅ Completed {demo_name} with {len(actions)} timesteps")
            valid_demos += 1

            # Force sync after each demo
            try:
                store.map.sync()
            except:
                pass

        # Update total demonstration count
        root.attrs["total_demonstrations"] = valid_demos

        # Final sync and close
        try:
            logger.info("Syncing zarr store...")
            if hasattr(store, 'sync'):
                store.sync()
            if hasattr(store, 'flush'):
                store.flush()
            if hasattr(store, 'close'):
                store.close()
        except Exception as e:
            logger.warning(f"Sync/close warning: {e}")
        
        logger.info(f"✅ Zarr dataset created successfully: {self.output_file}")
        logger.info(f"✅ Total valid demonstrations: {valid_demos}")

        # Verify save results
        self.verify_zarr_file()
    
    def verify_zarr_file(self):
        """Verify if zarr file is saved correctly"""
        try:
            logger.info("🔍 Verifying zarr file...")

            # Reopen zarr file
            verify_store = zarr.DirectoryStore(str(self.output_file))
            verify_root = zarr.group(store=verify_store)
            
            if 'data' not in verify_root:
                logger.error("❌ Verification failed: no data group found")
                return
            
            verify_data = verify_root['data']
            verify_demos = [k for k in verify_data.keys() if k.startswith('demo_')]
            
            logger.info(f"✅ Verification successful: {len(verify_demos)} demos found")

            # Check contents of the first demo
            if verify_demos:
                demo = verify_data[verify_demos[0]]
                logger.info(f"✅ {verify_demos[0]} contents: {list(demo.keys())}")
                
                if 'actions' in demo:
                    actions_shape = demo['actions'].shape
                    logger.info(f"✅ Actions shape: {actions_shape}")
                
                if 'obs' in demo:
                    obs_keys = list(demo['obs'].keys())
                    logger.info(f"✅ Observation keys: {obs_keys}")

            # Calculate file size
            total_size = 0
            for root, dirs, files in os.walk(self.output_file):
                for file in files:
                    total_size += os.path.getsize(os.path.join(root, file))
            
            logger.info(f"✅ Total file size: {total_size/1024**2:.1f} MB")
            
        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")

    def process_data(self):
        """Process data main entry - only supports Zarr"""
        logger.info("Creating Zarr format dataset...")
        self.create_zarr()

def main():
    parser = argparse.ArgumentParser(description="Convert Meta Quest demo data to Zarr format")
    parser.add_argument("--input_dir", type=str, default="demos_collected", 
                       help="Input directory containing Meta Quest demo data")
    parser.add_argument("--output_file", type=str, default="data/metaquest_dataset.zarr",
                       help="Output Zarr file path")
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224],
                       help="Image resize dimensions")
    
    args = parser.parse_args()
    
    processor = MetaQuestDataProcessor(
        input_dir=args.input_dir, 
        output_file=args.output_file
    )
    processor.image_size = tuple(args.image_size)
    
    processor.process_data()

if __name__ == "__main__":
    main()