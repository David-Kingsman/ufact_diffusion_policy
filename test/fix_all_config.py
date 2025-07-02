import yaml
import re

# 修复训练配置文件
with open('diffusion_policy/config/train_diffusion_unet_real_image_workspace.yaml', 'r') as f:
    content = f.read()

# 替换所有变量引用为固定值
content = content.replace('${task.name}', 'real_lift_image_abs')
content = content.replace('${task_name}', 'real_lift_image_abs')
content = content.replace('${name}', 'train_diffusion_unet_image')

with open('diffusion_policy/config/train_diffusion_unet_real_image_workspace.yaml', 'w') as f:
    f.write(content)

print("Fixed all variable references in training config!")
