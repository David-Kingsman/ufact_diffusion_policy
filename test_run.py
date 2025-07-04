import zarr
import numpy as np

# 打开 zarr 数据集
z = zarr.open('data/metaquest_xarm_dataset.zarr', mode='r')

# 输出根属性
print("Root attrs:", dict(z.attrs))

# 检查 'data' group 是否存在
if 'data' not in z:
    raise ValueError("Dataset is missing 'data' group")

# 检查每个 demo 的配置和内容
for demo in z['data']:
    demo_group = z['data'][demo]
    print(f"Demo: {demo}")
    print("  attrs:", dict(demo_group.attrs))
    # 检查必要字段
    if 'obs' not in demo_group or 'actions' not in demo_group:
        raise ValueError(f"{demo} is missing 'obs' or 'actions'")
    obs = demo_group['obs']
    actions = demo_group['actions']
    print(f"  actions shape: {actions.shape}")

    # 判断并输出相机图片 shape
    if 'agentview0_image' in obs:
        print(f"  agentview0_image shape: {obs['agentview0_image'].shape}")
    if 'agentview1_image' in obs:
        print(f"  agentview1_image shape: {obs['agentview1_image'].shape}")

    # 兼容单相机和多相机情况，优先用 agentview_image，否则用 agentview1_image
    if 'agentview_image' in obs:
        img = obs['agentview_image']
        print(f"  agentview_image shape: {img.shape}")
        # 检查长度一致
        assert actions.shape[0] == img.shape[0]
        # 如果有 agentview0_image，检查内容一致
        if 'agentview0_image' in obs:
            assert np.all(img[:] == obs['agentview0_image'][:])
    elif 'agentview1_image' in obs:
        img = obs['agentview1_image']
        print(f"  [No agentview_image] Use agentview1_image shape: {img.shape}")
        assert actions.shape[0] == img.shape[0]
    else:
        raise ValueError(f"{demo} is missing both 'agentview_image' and 'agentview1_image'")

    print("-" * 40)

    # 检查第一个 demo 的 obs 下有哪些 key
    demo_keys = list(z['data'].keys())
    print("Demo keys:", demo_keys)

    if demo_keys:
        obs_keys = list(z['data'][demo_keys[0]]['obs'].keys())
        print("obs keys in first demo:", obs_keys)
    else:
        print("No demo found in data group.")