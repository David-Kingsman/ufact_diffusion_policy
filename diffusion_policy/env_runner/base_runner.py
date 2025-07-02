import os
from typing import Dict, Any

class BaseRunner:
    def __init__(self, output_dir="./outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def run(self, policy) -> Dict[str, Any]:
        raise NotImplementedError()
