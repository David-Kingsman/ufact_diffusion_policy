"""
Usage:
Training:
python train.py --config-name=train_diffusion_unet_real_image_workspace
# Relative action training (use + prefix)
python train.py --config-name=train_diffusion_unet_real_image_workspace +action_type=relative +relative_type=6D
# Absolute action training (use action_type=absolute)
python train.py --config-name=train_diffusion_unet_real_image_workspace action_type=absolute
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
def main(cfg: OmegaConf):
    # Check action type - unified naming
    action_type = cfg.get('action_type', 'absolute')  # Default to absolute action
    relative_type = cfg.get('relative_type', '6D')
    
    # Unified action type handling
    use_relative_action = (action_type == 'relative')
    use_absolute_action = (action_type == 'absolute')
    
    if use_relative_action:
        print(f"Enable relative action training mode")
        print(f"Relative action type: {relative_type}")
        
        # Use OmegaConf.set_struct to temporarily allow new keys
        OmegaConf.set_struct(cfg, False)
        
        # Modify experiment name
        original_name = cfg.get('name', 'train_diffusion_unet_image')
        cfg.name = f"{original_name}_relative_{relative_type}"
        cfg.exp_name = f"relative_{relative_type}_training"
        
        # Modify dataset configuration
        if 'dataset' in cfg:
            cfg.dataset.abs_action = False          # Disable absolute action
            cfg.dataset.use_relative_action = True  # Enable relative action
            cfg.dataset.relative_type = relative_type
        
        # Modify policy configuration
        if 'policy' in cfg:
            cfg.policy.use_relative_action = True
            cfg.policy.relative_type = relative_type
        
        # Modify task configuration
        if 'task' in cfg:
            cfg.task.use_relative_action = True
            cfg.task.relative_type = relative_type
        
        OmegaConf.set_struct(cfg, True)
        
    else:  # Absolute action mode
        print(f"Using absolute action training mode")
        OmegaConf.set_struct(cfg, False)
        
        # Modify dataset configuration
        if 'dataset' in cfg:
            cfg.dataset.abs_action = True           # Enable absolute action
            cfg.dataset.use_relative_action = False # Disable relative action
        
        # Modify policy configuration
        if 'policy' in cfg:
            cfg.policy.use_relative_action = False
        
        # Modify task configuration
        if 'task' in cfg:
            cfg.task.use_relative_action = False
        
        OmegaConf.set_struct(cfg, True)
    
    # Resolve configuration
    OmegaConf.resolve(cfg)
    
    # Print training configuration info
    print("="*60)
    print("Training Configuration Info")
    print("="*60)
    print(f"Config file: {cfg.get('name', 'N/A')}")
    print(f"Action type: {action_type.upper()}")
    if use_relative_action:
        print(f"Relative type: {relative_type}")
    print(f"Experiment name: {cfg.get('exp_name', cfg.get('name', 'N/A'))}")
    print(f"Data path: {cfg.get('dataset_path', 'N/A')}")
    print("="*60)
    
    # Get workspace class and instantiate
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
