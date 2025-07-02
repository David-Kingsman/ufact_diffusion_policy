import torch
from diffusion_policy.dataset.xarm_hdf5_dataset import XarmHDF5Dataset
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
import yaml

# 加载配置
with open('diffusion_policy/config/task/real_lift_image_abs.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 加载数据
dataset = XarmHDF5Dataset('data/metaquest_xarm_dataset.hdf5')
batch = dataset[0]

print("实际数据形状:")
for key, value in batch['obs'].items():
    print(f"  {key}: {value.shape}")

print("\n配置期望形状:")
for key, value in config['shape_meta']['obs'].items():
    print(f"  {key}: {value['shape']}")

print("\n对比:")
for key in batch['obs'].keys():
    actual_shape = list(batch['obs'][key].shape[1:])  # 去掉时间维度
    expected_shape = config['shape_meta']['obs'][key]['shape']
    match = actual_shape == expected_shape
    print(f"  {key}: {actual_shape} vs {expected_shape} {'✅' if match else '❌'}")
