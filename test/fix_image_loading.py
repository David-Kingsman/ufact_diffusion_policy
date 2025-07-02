import re

# 读取文件
with open('post_process_data.py', 'r') as f:
    content = f.read()

# 修复load_images_for_run方法
old_method = '''    def load_images_for_run(self, run_dir, camera_data):
        """加载Meta Quest的图像数据 - 修改版本"""
        # 首先检查run目录内的images
        images_dir = run_dir / "images"
        
        if not images_dir.exists():
            # 如果run目录内没有images，检查父目录的images
            parent_images_dir = run_dir.parent / "images"
            if parent_images_dir.exists():
                logger.info(f"Using parent images directory: {parent_images_dir}")
                images_dir = parent_images_dir
            else:
                logger.warning(f"No images directory found for {run_dir}")
                return None
        
        # 使用第一个找到的图像文件夹
        img_folder = img_folders[0]
        logger.info(f"Loading images from {img_folder}")
        
        images = []'''

new_method = '''    def load_images_for_run(self, run_dir, camera_data):
        """加载Meta Quest的图像数据 - 修改版本"""
        # 首先检查run目录内的images
        images_dir = run_dir / "images"
        
        if not images_dir.exists():
            # 如果run目录内没有images，检查父目录的images
            parent_images_dir = run_dir.parent / "images"
            if parent_images_dir.exists():
                logger.info(f"Using parent images directory: {parent_images_dir}")
                images_dir = parent_images_dir
            else:
                logger.warning(f"No images directory found for {run_dir}")
                return None
        
        images = []'''

# 替换内容
content = content.replace(old_method, new_method)

# 写回文件
with open('post_process_data.py', 'w') as f:
    f.write(content)

print("Fixed the img_folders bug!")
