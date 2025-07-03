"""
# Usage:
python eval_real_ufact_robots.py --config-name=eval_diffusion_unet_real_image_workspace ckpt_path=<ckpt_path>
"""

import os
import sys
import pathlib
import torch
import dill
import hydra
import psutil
import cv2
from omegaconf import OmegaConf
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

# limit the number of threads used by various libraries to avoid performance issues
os.environ["OPENBLAS_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"
os.environ["OMP_NUM_THREADS"] = "12"
cv2.setNumThreads(12)
total_cores = psutil.cpu_count()
num_cores_to_bind = 10
cores_to_bind = set(range(min(num_cores_to_bind, total_cores)))
os.sched_setaffinity(0, cores_to_bind)

OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy', 'config'))
)
def main(cfg):
    # 1. Load checkpoint
    ckpt_path = getattr(cfg, "ckpt_path", None)
    if ckpt_path is None:
        raise ValueError("Please provide ckpt_path=<path_to_checkpoint> in command line.")
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)

    # 2. Instantiate workspace and load weights
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # 3. Get policy object
    if hasattr(cfg, "training") and getattr(cfg.training, "use_ema", False):
        policy = getattr(workspace, "ema_model", workspace.model)
    else:
        policy = workspace.model
    policy: BaseImagePolicy

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy.eval().to(device)

    # 4. set policy attributes based on configuration
    if hasattr(policy, "num_inference_steps"):
        policy.num_inference_steps = getattr(cfg, "num_inference_steps", 16)
    if hasattr(policy, "n_action_steps") and hasattr(policy, "horizon") and hasattr(policy, "n_obs_steps"):
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

    # 5. print evaluation configuration info
    print("="*60)
    print("Evaluation Configuration Info")
    print("="*60)
    print(f"Config file: {cfg.get('name', 'N/A')}")
    print(f"Action type: {'RELATIVE' if getattr(cfg, 'use_relative_action', False) else 'ABSOLUTE'}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Device: {device}")
    print("="*60)

    # 6. Instantiate env_runner and run evaluation
    env_runner = hydra.utils.instantiate(cfg.task.env_runner)
    env_runner.run(policy)

if __name__ == '__main__':
    main()