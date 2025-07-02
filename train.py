"""
Usage:
Training:
python train.py --config-name=train_diffusion_unet_real_image_workspace
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)   
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
import json
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)  
print(OmegaConf)
print(str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config')))

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))

    
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)
    

    cls = hydra.utils.get_class(cfg._target_)   # get workspace class
    # print(cfg._target_)
    # print(cls)
    # print(cfg)
    workspace: BaseWorkspace = cls(cfg) # instantiate workspace
    workspace.run()                         # start training

if __name__ == "__main__":
    main()