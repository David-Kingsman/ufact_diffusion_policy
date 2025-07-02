import numpy as np
from pathlib import Path

def safe_load_npz(file_path):
    """安全加载npz文件"""
    try:
        data = np.load(file_path, allow_pickle=True)
        # 尝试不同的键
        if 'data' in data.files:
            return data['data']
        elif len(data.files) == 1:
            return data[data.files[0]]
        else:
            # 返回所有数据的字典
            return {key: data[key] for key in data.files}
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# 测试加载demo数据
demo_dir = Path("demos_collected/run001")
if demo_dir.exists():
    action_file = demo_dir / "demo_action.npz"
    if action_file.exists():
        action_data = safe_load_npz(action_file)
        print(f"Action data type: {type(action_data)}")
        if hasattr(action_data, 'shape'):
            print(f"Action shape: {action_data.shape}")
        else:
            print(f"Action keys: {action_data.keys() if isinstance(action_data, dict) else 'Not a dict'}")
