#!/usr/bin/env bash

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=str, default="0")
    parser.add_argument('--node', type=str, default="0015")
    opt = parser.parse_args()

    return opt

args = parse_args() 
#500000 on the max_iter
# data path can be /workspaces/container-workspace/standardized_view for normal daghar data

# if you want to use the normalized data, you can use the following paths:
# /workspaces/container-workspace/data/normalized_label for normalized label data
# or /workspaces/container-workspace/data/normalized_all for normalized all data

#if not, take out the data_normalized argument
# and set the data_path to the path of the data you want to use, e.g., /workspaces/container-workspace/data/standardized_view


message = f"CUDA_VISIBLE_DEVICES=0 python trainCGAN_daghar.py \
-gen_bs 64 \
-dis_bs 64 \
--data_path /workspaces/container-workspace/data/normalized_all \
--data_normalized \
--dist-url 'tcp://localhost:4321' \
--dist-backend 'nccl' \
--world-size 1 \
--rank {args.rank} \
--dataset daghar \
--bottom_width 8 \
--max_iter 200000 \
--max_epoch 100 \
--img_size 32 \
--gen_model my_gen \
--dis_model my_dis \
--df_dim 384 \
--d_heads 4 \
--d_depth 3 \
--g_depth 5,4,2 \
--dropout 0 \
--latent_dim 100 \
--gf_dim 1024 \
--num_workers 16 \
--g_lr 0.00001 \
--d_lr 0.00003 \
--optimizer adam \
--loss lsgan \
--wd 1e-3 \
--beta1 0.9 \
--beta2 0.999 \
--phi 1 \
--batch_size 64 \
--num_eval_imgs 50000 \
--init_type xavier_uniform \
--n_critic 1 \
--val_freq 20 \
--print_freq 150 \
--grow_steps 0 0 \
--fade_in 0 \
--patch_size 2 \
--ema_kimg 500 \
--ema_warmup 0.1 \
--ema 0.9999 \
--diff_aug translation,cutout,color \
--exp_name normalized_all_dagharCGAN"

print(message)
os.system(message)