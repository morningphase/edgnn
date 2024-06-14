#!/bin/bash

# 算法数组
algorithms=("PGExplainer")

# 数据集数组
datasets=("ba_2motifs")

# 循环遍历数据集
for dataset in "${datasets[@]}"; do
    for algorithm in "${algorithms[@]}"; do
        python gin.py --explainer $algorithm --dataset $dataset
    done
  done