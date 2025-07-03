import sys
sys.path.append('.')

from diffusion_policy.workspace.train_diffusion_unet_image_workspace import TrainDiffusionUnetImageWorkspace
import inspect

# 查看workspace类的方法
print("🔍 TrainDiffusionUnetImageWorkspace 类方法:")
methods = [method for method in dir(TrainDiffusionUnetImageWorkspace) if not method.startswith('_')]
for method in methods:
    print(f"  - {method}")

# 查看run方法的源码
print("\n🚀 run方法签名:")
print(inspect.signature(TrainDiffusionUnetImageWorkspace.run))

# 查看__init__方法
print("\n🔧 __init__方法签名:")
print(inspect.signature(TrainDiffusionUnetImageWorkspace.__init__))
