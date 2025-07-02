import yaml

# 读取训练配置文件
with open('diffusion_policy/config/train_diffusion_unet_real_image_workspace.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 修复 hydra 配置，使用固定名称而不是变量引用
config['hydra'] = {
    'job': {
        'override_dirname': 'real_lift_image_abs'
    },
    'run': {
        'dir': 'data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_real_lift_image_abs'
    },
    'sweep': {
        'dir': 'data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_real_lift_image_abs',
        'subdir': '${hydra.job.num}'
    }
}

# 同样修复 multi_run 配置
config['multi_run'] = {
    'run_dir': 'data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_real_lift_image_abs',
    'wandb_name_base': '${now:%Y.%m.%d-%H.%M.%S}_real_lift_image_abs'
}

# 修复 logging 配置
config['logging']['name'] = '${now:%Y.%m.%d-%H.%M.%S}_real_lift_image_abs'

# 写回文件
with open('diffusion_policy/config/train_diffusion_unet_real_image_workspace.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print("Fixed Hydra configuration!")
