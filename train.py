"""
# Usage:
# Training (default: absolute action):
python train.py --config-name=train_diffusion_unet_real_image_workspace

# Relative action training:
python train.py --config-name=train_diffusion_unet_real_image_workspace +action_type=relative

# Absolute action training:
python train.py --config-name=train_diffusion_unet_real_image_workspace +action_type=absolute
"""

import sys
import os
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)   
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)

# action type can be 'absolute' or 'relative'
def main(cfg: OmegaConf):
    action_type = cfg.get('action_type', 'absolute')
    use_relative_action = (action_type == 'relative')
    
    # Set the action type in the configuration   
    OmegaConf.set_struct(cfg, False)
    if 'dataset' in cfg:
        cfg.dataset.abs_action = not use_relative_action
        cfg.dataset.use_relative_action = use_relative_action
    if 'policy' in cfg:
        cfg.policy.use_relative_action = use_relative_action
    if 'task' in cfg:
        cfg.task.use_relative_action = use_relative_action
    OmegaConf.set_struct(cfg, True)

    
    # Print training configuration info
    print("="*60)
    print("Training Configuration Info")
    print("="*60)
    print(f"Config file: {cfg.get('name', 'N/A')}")
    print(f"Action type: {action_type.upper()}")
    print(f"Experiment name: {cfg.get('exp_name', cfg.get('name', 'N/A'))}")
    print(f"Data path: {cfg.get('dataset_path', 'N/A')}")
    print("="*60)
    
    # Get workspace class and instantiate
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
