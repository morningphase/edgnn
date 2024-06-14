from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch.nn import Parameter
import torch
import sys
sys.path.append('../src')
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv, TopKPooling, JumpingKnowledge,  DenseSAGEConv, dense_diff_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric import utils
import argparse
import os
from torch.utils.data import random_split
import torch.nn as nn
from copy import deepcopy
from pathlib import Path
from utils import get_data_loaders, get_model, set_seed, CriterionClf, init_metric_dict, load_checkpoint, generate_all_cases, get_local_config_name
from datetime import datetime
import yaml
from math import ceil
from torch_geometric.utils import to_dense_adj  # Import this to convert edge index to dense adjacency matrix
from torch_geometric.loader import DenseDataLoader
from datasets import  LogFusionDataset
import numpy as np

from torch.nn import Linear

max_nodes = 250

class Net(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.conv1 = GraphConv(args.num_features, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)

        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, args.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x

def test(model,loader):
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        data.y = torch.tensor(data.y,dtype=torch.int64)
        loss += F.nll_loss(out,data.y,reduction='sum').item()
    return correct / len(loader.dataset),loss / len(loader.dataset)


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777,
                    help='seed')
#parser.add_argument('--batch_size', type=int, default=128,
#                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.5,
                    help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.5,
                    help='dropout ratio')
parser.add_argument('--epochs', type=int, default=1000,
                    help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50,
                    help='patience for earlystopping')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
parser.add_argument('--dataset', type=str, default="Apache-1", help='dataset used')
parser.add_argument('--backbone', type=str, default="GIN", help='backbone model used')
parser.add_argument('--cuda', type=int, default=0, help='cuda device id, -1 for cpu')
parser.add_argument('--batch_size', type=str, default="128", help='batch size')
parser.add_argument('--dropout_p', type=str, default="0.3", help='dropout portion')
parser.add_argument('--args.num_layer', type=int, default=3, help='number of layers')
parser.add_argument('--args.num_features', type=int, default=64, help='number of features')
parser.add_argument('--args.num_classes', type=int, default=4, help='number of features')


args = parser.parse_args()
args.device = 'cpu'
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:0'

dataset_name = args.dataset
model_name = 'PNA'

method_name = 'GSAT'
cuda_id = 0

data_dir = Path('../data')
device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')

if model_name == 'GIN':
    model_config = {'model_name': 'GIN', 'hidden_size': 64, 'n_layers': 2, 'dropout_p': 0.3, 'use_edge_attr': True}
elif model_name == 'PNA':
    model_config = {'model_name': 'PNA', 'hidden_size': 80, 'n_layers': 4, 'dropout_p': 0.3, 'use_edge_attr': False, 
                    'atom_encoder': False, 'aggregators': ['mean', 'min', 'max', 'std'], 'scalers': False}
else:
    assert model_name == 'RGCN'
    model_config = {'model_name': 'RGCN', 'hidden_size': 64, 'n_layers': 2, 'dropout_p': 0.3, 'use_edge_attr': False}

torch.set_num_threads(5)
config_dir = Path('../src/configs')

print('====================================')
print('====================================')
print(f'[INFO] Running {method_name} on {dataset_name} with {model_name}')
print('====================================')

global_config = yaml.safe_load((config_dir / 'global_config.yml').open('r'))
local_config_name = get_local_config_name(model_name, dataset_name)
local_config = yaml.safe_load((config_dir / local_config_name).open('r'))


all_cases = generate_all_cases(local_config['mode_config'], dataset_name)
for case in all_cases:
    mode = case[0]
    params = case[1]
    results = []
    for seed in range(20):
        set_seed(seed)


        metric_dict = deepcopy(init_metric_dict)
        model_dir = data_dir / dataset_name / 'logs' / (datetime.now().strftime("%m_%d_%Y-%H_%M_%S") + '-' + dataset_name + '-' + model_name + '-seed' + str(seed) + '-' + method_name)
        dataset = LogFusionDataset(data_dir, dataset_name, False, mode, params)
        loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info = get_data_loaders(data_dir, dataset_name, batch_size=128, random_state=seed, mode=mode, parameters=params,
                                                                                        splits={'train': 0.8, 'valid': 0.1, 'test': 0.1},
                                                                                        mutag_x=True if dataset_name == 'mutag' else False)

        train_loader = loaders['train']
        val_loader = loaders['valid']
        test_loader = loaders['test']
        args.num_classes = num_class
        args.num_features = x_dim

        model = Net(args).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        min_loss = 1e10
        patience = 0

        for epoch in range(args.epochs):
            model.train()
            for i, data in enumerate(train_loader):
                data = data.to(args.device)
                out = model(data)
                # data.y = torch.tensor(data.y,dtype=torch.int64)
                data.y = data.y.view(-1).long()  # Ensure data.y is a 1D LongTensor
                loss = F.nll_loss(out, data.y)
                print("Training loss:{}".format(loss.item()))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            val_acc,val_loss = test(model,val_loader)
            print("Validation loss:{}\taccuracy:{}".format(val_loss,val_acc))
            if val_loss < min_loss:
                torch.save(model.state_dict(),'latest.pth')
                print("Model saved at epoch{}".format(epoch))
                min_loss = val_loss
                patience = 0
            else:
                patience += 1
            if patience > args.patience:
                break

        model = Net(args).to(args.device)
        model.load_state_dict(torch.load('latest.pth'))
        val_acc,val_loss = test(model,val_loader)
        test_acc,test_loss = test(model,test_loader)
        print("Test accuarcy:{}".format(test_acc),val_acc)

        results.append(test_acc)

    results = np.array(results)

    # 计算平均值
    mean = np.mean(results)

    # 计算标准差
    std_dev = np.std(results)

    # 打印结果
    print(f"平均值: {mean}")
    print(f"标准差: {std_dev}")

    # 格式化字符串，保留两位小数
    formatted_output = f"{mean:.8f} ± {std_dev:.9f}"

    params_str = ' '.join([f"{key}: {value}" for key, value in params.items()])

    # 写入到文件
    with open(f'results_topkpool_{mode}_{params_str}_{dataset_name}.txt', 'w') as f:
        f.write(formatted_output)

        # 打印到控制台（如果需要）
    print(formatted_output)