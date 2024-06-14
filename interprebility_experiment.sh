#!/bin/bash

cd src
  
# 算法数组
algorithms=("PNA" "GIN")

# 循环遍历数据集
for backbone in "${algorithms[@]}"; do
    python run_dgib.py --dataset=Apache-1 --backbone $backbone --cuda 0 --batch_size 80
    python run_gsat.py --dataset=Apache-1 --backbone $backbone --cuda 0
done