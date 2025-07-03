#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    dataset_path=data/metaquest_xarm_dataset.zarr \
    name=xarm_metaquest_diffusion \
    logging.mode=online \
    logging.project=xarm_metaquest_diffusion \
    +action_type=relative  #  +action_type=absolute, change to absolute if needed
