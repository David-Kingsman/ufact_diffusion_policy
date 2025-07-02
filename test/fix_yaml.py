# 修复 YAML 文件中的语法错误
with open('diffusion_policy/config/task/real_lift_image_abs.yaml', 'r') as f:
    content = f.read()

# 修复 camera_names 格式
content = content.replace(
    'camera_names: \n  - agentview_image  # 修复：改为YAML列表格式',
    'camera_names:\n  - agentview_image'
)

with open('diffusion_policy/config/task/real_lift_image_abs.yaml', 'w') as f:
    f.write(content)

print("Fixed YAML syntax error!")
