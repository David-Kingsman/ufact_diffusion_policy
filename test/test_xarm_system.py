import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pathlib

@hydra.main(config_path="diffusion_policy/config", config_name="train_diffusion_unet_real_image_workspace", version_base=None)
def test_system(cfg: DictConfig):
    from diffusion_policy.env_runner.xarm_real_runner import XarmRealRunner
    
    print("🚀 测试 XarmRealRunner 系统初始化...")
    
    try:
        # 初始化runner
        runner = XarmRealRunner(**cfg.env_runner)
        print("✅ XarmRealRunner 初始化成功!")
        
        # 检查硬件状态
        print(f"🤖 XARM连接状态: {'✅ 已连接' if hasattr(runner, 'xarm') and runner.xarm else '❌ 模拟模式'}")
        print(f"📷 相机数量: {len(getattr(runner, 'cameras', []))}")
        
        # 测试观测获取
        obs = runner._get_observation()
        print(f"📊 观测数据结构:")
        for key, value in obs.items():
            if torch.is_tensor(value):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")
                
        print("🎉 系统测试完成!")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    test_system()
