#!/bin/bash

# 算法数组
algorithms=("GIN")

# 数据集数组
datasets=("Apache-1")

# lr
lrs=("1.0e-3")

# batch size数组
bs_params=(192)

# hidden size数组
hidden_params=(192)

# dropout_p数组
dropout_ps=("0.3")

# extractor_dropout_p数组
extractor_dropout_ps=("0.3")

# disc_coef数组
disc_coefs=("0.1" "0.3" "1" "2" "3" "5" "10" "30")

cd src

# 循环遍历数据集
for dataset in "${datasets[@]}"; do
    for backbone in "${algorithms[@]}"; do
      for lr in "${lrs[@]}"; do
        for batch_size in "${bs_params[@]}"; do
          for hidden_size in "${hidden_params[@]}"; do
            for dropout_p in "${dropout_ps[@]}"; do
              for extractor_dropout_p in "${extractor_dropout_ps[@]}"; do
                for disc_coef in "${disc_coefs[@]}"; do
                  python run_dgib.py --dataset=$dataset --backbone $backbone --lr $lr --batch_size $batch_size --hidden_size $hidden_size --dropout_p $dropout_p --extractor_dropout_p $extractor_dropout_p --disc_coef $disc_coef
                done
              done
            done
          done
        done
      done
    done
  done