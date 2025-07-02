import os
import numpy as np
from typing import Dict, Any
from diffusion_policy.env_runner.base_runner import BaseRunner

class BaseImageRunner(BaseRunner):
    def __init__(self, output_dir="./outputs"):
        super().__init__(output_dir)
    
    def run(self, policy) -> Dict[str, Any]:
        return {
            'test_mean_score': 0.0,
            'test_std_score': 0.0,
            'train_mean_score': 0.0,
            'train_std_score': 0.0
        }
