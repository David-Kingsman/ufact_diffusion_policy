import sys
sys.path.append('.')

from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
import inspect

# 查看policy类的方法
print("🤖 DiffusionUnetImagePolicy 类方法:")
methods = [method for method in dir(DiffusionUnetImagePolicy) if not method.startswith('_')]
for method in methods:
    print(f"  - {method}")

# 查看关键方法
print("\n🎯 predict_action方法签名:")
try:
    print(inspect.signature(DiffusionUnetImagePolicy.predict_action))
except:
    print("未找到predict_action方法")

print("\n🔧 __init__方法签名:")
print(inspect.signature(DiffusionUnetImagePolicy.__init__))
