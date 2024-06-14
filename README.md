# EDHGNN

## Breif Overview

> To identify the root causes of attacks, behavior abstraction (BA) converts audit logs into multiple behavior graphs and finds similar ones, which has proven effective in bridging the semantic gap and reducing manual workload. Existing works fail to achieve both interpretability and generalization, while also exhibiting limited robustness when facing adversarial attacks. In this paper, we give the first attempt at interpretable and robust behavior abstraction and propose a novel method called 
\textit{\textbf{E}nvironment-\textbf{D}isentangled \textbf{H}eterogeneous \textbf{G}raph \textbf{N}eural \textbf{N}etwork (\textbf{EDHGNN})}. Motivated by Information Bottleneck (IB) principle, we introduce a Heterogeneous Subgraph Disentanglement (HSD) module to disentangle the label-relevant and environmental heterogeneous subgraph from the input behavior graph, which eliminates the need for multiple optimizations and consistently performs maximum compression with a single optimization. Based on the label-relevant subgraph, we propose a Adapted Graph-Level Attention (AGLA) module to extract the minimal and sufficient graph-level representation. We also propose a Label-Guided Graph Reconstructor (LGGR) method that uses environmental subgraph and label information to reconstruct the original graph, so that the coverage of environmental information can be maximized.  A Relevance Discriminator (RD) is proposed to encourage better disentanglement. Additionally, we construct a new dataset contains ground-truth explanations and 4,160 behavior graphs. Extensive experiments demonstrate that EDHGNN outperforms the state-of-the-art methods in terms of interpretability and robustness against
adversarial attacks.

## Python environment setup with Conda

Our code is written in Python3.10.8 with cuda 12.1 and pytorch 2.1.0 on Ubuntu 22.04.

install anaconda：https://repo.anaconda.com/archive/index.html.

install torch-scatter 2.1.2+pt21cu121：https://pytorch-geometric.com/whl/torch-2.1.2%2Bcu121.html.

```
conda create --name EDHGNN
conda activate EDHGNN
pip install -r requirments.txt
```


## Directory

We present a brief introduction about the directories.

- Attack/    # Code for Adversarial Attacks
- data/    # The directory for storing data and intermediate results of the algorithm. Owing to the space constraints, we provide the Apache-1 dataset.
- baseline/    # Baseline code for graph pooling method.
- example/    # Baseline GSAT code.
- example_edhgnn/    # Code for EDHGNN.
- README.md # Guide to the project
- ablation_experiment.sh  # Script for Ablation Study
- grid_search.sh  # Script for grid search
- interprebility_experiment.sh * # Script for running experiments on interpretability
- requirements.txt # Dependencies installed py pip install


## Reproducibility

Use `ablation_experiment.sh` to reproduce the results of ablation study.

```
bash ablation_experiment.sh
```

Use `grid_search.sh` to  reproduce the results of grid search.

```
bash grid_search.sh
```

Use `interprebility_experiment.sh` to reproduce the results of experiments on interpretability.

```
bash interprebility_experiment.sh
```