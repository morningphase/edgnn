#!/bin/bash

cd src
  
# 算法数组
algorithms=("GIN")

# 数据集数组
datasets=("Apache-1")

# 循环遍历数据集
for dataset in "${datasets[@]}"; do
    for backbone in "${algorithms[@]}"; do
        python run_dgib.py --dataset=$dataset --backbone $backbone --cuda 0 --disc_coef 0
        python run_dgib.py --dataset=$dataset --backbone $backbone --cuda 0 --pred_coef 0
        python run_dgib.py --dataset=$dataset --backbone $backbone --cuda 0 --recon_coef 0
    done
  done