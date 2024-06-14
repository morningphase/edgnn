import os
import argparse

import os.path as osp
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BatchNorm
from torch.nn import Linear, ReLU, Sequential
import yaml
from torch_geometric.nn import GINConv, global_add_pool
import sys
sys.path.append('../src')
import numpy as np
import networkx as nx
from torch_geometric.utils import subgraph, to_networkx
from sklearn.metrics import roc_auc_score
import csv
from torch_geometric.data import Data
from pyecharts.charts import Graph
from pyecharts import options as opts
import random
import pickle
import torch
import torch.nn as nn
from pathlib import Path
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig, PGExplainer, GraphMaskExplainer
from utils import get_data_loaders, set_seed, generate_all_cases, get_local_config_name

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyGINConv(GINConv):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        mlp = Sequential(
            Linear(in_channels, 2 * out_channels),
            BatchNorm(2 * out_channels),
            ReLU(),
            Linear(2 * out_channels, out_channels),
        )
        super().__init__(mlp, train_eps=True)

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for i in range(num_layers):
            conv = MyGINConv(in_channels, hidden_channels)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_channels))
            in_channels = hidden_channels

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.batch_norm1 = BatchNorm(hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))
        x = global_add_pool(x, batch)
        x = F.relu(self.batch_norm1(self.lin1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

def train(model, data, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.batch)
    data.y = data.y.clone().detach().long()
    loss = F.nll_loss(out, data.y.squeeze())
    loss.backward()
    optimizer.step()
    
    return loss.item()

@torch.no_grad()
def test(model, loader):
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.max(dim=1)[1]
        total_correct += pred.eq(data.y.squeeze()).sum().item()
    return total_correct / len(loader.dataset)

def calculate_explanation_accuracy(model, data, explainer, explainer_name, dataset_name, stage, num_edges, random_state, render):
    # 初始化 ROU_AUC 列表，用于存储每个图的 ROU_AUC 值
    rou_auc_list = []
    precision_at_k_list = []
    
    # 对于数据loader中的每个批次
    for graph_data in data:
        # 解释模型输出并归一化解释结果
        graph_data = graph_data.to(device)
        
        if explainer_name == "GNNExplainer":
            # 训练 GNNExplainer
            explanation = explainer(graph_data.x, graph_data.edge_index, batch=graph_data.batch)
            
        elif explainer_name == "PGExplainer":
            # 训练 PGExplainer
            for epoch in range(200):
                loss = explainer.algorithm.train(epoch, model, graph_data.x, graph_data.edge_index,
                                                target=graph_data.y.squeeze().long(), index=0, batch=graph_data.batch)  # 在第 0 个节点上训练
            # 生成解释
            explanation = explainer(graph_data.x, graph_data.edge_index, target=graph_data.y.long(), index=0, batch=graph_data.batch)
            
        elif explainer_name == "GraphMaskExplainer":
            # 训练 GraphMaskExplainer
            explanation = explainer(graph_data.x, graph_data.edge_index, batch=graph_data.batch)

        # 计算 ROU_AUC 值
        edge_label = graph_data.edge_label.data.cpu()
        explanation = explanation.edge_mask.cpu()
        edge_index = graph_data.edge_index.data.cpu()
        batch = graph_data.batch.data.cpu()

        precision_at_k = []
        for i in range(batch.max()+1):
            nodes_for_graph_i = batch == i
            edges_for_graph_i = nodes_for_graph_i[edge_index[0]] & nodes_for_graph_i[edge_index[1]]
            labels_for_graph_i = edge_label[edges_for_graph_i]
            mask_log_logits_for_graph_i = explanation[edges_for_graph_i]
            precision_at_k.append(labels_for_graph_i[np.argsort(-mask_log_logits_for_graph_i)[:2]].sum().item() / 2)
        
        att_auroc = roc_auc_score(edge_label, explanation) if np.unique(edge_label).shape[0] > 1 else 0
        
        # 将 ROU_AUC 值添加到列表中
        rou_auc_list.append(att_auroc)
        precision_at_k_list.extend(precision_at_k)

    if stage == "test" and render:
        for i in range(batch.max()+1):
            graph_to_show = graph_data[i]
            nodes_for_graph_i = batch == i
            edges_for_graph_i = nodes_for_graph_i[edge_index[0]] & nodes_for_graph_i[edge_index[1]]
            mask_log_logits_for_graph_i = explanation[edges_for_graph_i]

            visualize_result(graph_to_show.edge_index, mask_log_logits_for_graph_i, graph_to_show.node_label, graph_to_show.node_type, graph_to_show.edge_label, graph_to_show.edge_type, dataset_name, int(graph_to_show.y.item()), i, explainer_name, num_edges, random_state)
    # 计算 ROU_AUC 平均值
    average_rou_auc = np.mean(rou_auc_list)
    average_precision_at_k = np.mean(precision_at_k_list)
    return average_rou_auc, average_precision_at_k

def get_model_config():
    model_config = ModelConfig(
        mode='multiclass_classification',
        task_level='graph',
        return_type='log_probs',
    )
    return model_config

def get_explainer(model, explainer_name, model_config):
    if explainer_name == "GNNExplainer":
        explainer = Explainer(
            model=model,
            explanation_type='model',
            algorithm=GNNExplainer(epochs=200),
            edge_mask_type='object',
            model_config=model_config,
        )

    elif explainer_name == "PGExplainer":
        explainer = Explainer(
            model=model,
            explanation_type='phenomenon',
            algorithm=PGExplainer(epochs=200, lr=0.003),
            edge_mask_type='object',
            model_config=model_config,
        )

    elif explainer_name == "GraphMaskExplainer":
        explainer = Explainer(
            model=model,
            algorithm=GraphMaskExplainer(3, epochs=200),
            explanation_type='model',
            edge_mask_type='object',
            model_config=model_config
        )
    return explainer

# 注意力分数的值范围为 [0, 1]
# 这里通过一个简单的函数将其转换为颜色，注意力分数接近1时偏橙色，接近0时偏蓝色
def score_to_color(score):
    # 橙色 [255, 165, 0]
    orange = [255, 165, 0]
    # 蓝色 [0, 0, 255]
    blue = [0, 0, 255]
    
    # 根据注意力分数的值计算颜色
    color = [int(orange[i] * score + blue[i] * (1 - score)) for i in range(3)]
    return 'rgb({},{},{})'.format(color[0], color[1], color[2])

def get_labels_for_log_fusion(data, G, dataset_name):
    node_map_file = os.path.join(f'../data/{dataset_name}/common/raw', 'node_mapping.pickle')
    edge_map_file = os.path.join(f'../data/{dataset_name}/common/raw', 'edge_mapping.pickle')
    with open(node_map_file, 'rb') as f:
        node_map = pickle.load(f)
    with open(edge_map_file, 'rb') as f:
        edge_map = pickle.load(f)
    for i, (u, v) in enumerate(data.edge_index.t().tolist()):
        edge_type = data.edge_type[i].item()
        G[u][v]['edge_type'] = edge_type
    node_types = nx.get_node_attributes(G, 'node_type')
    node_map_reverse = {value: key for key, value in node_map.items()}
    node_names = {key_node: node_map_reverse[value_node] for key_node, value_node in node_types.items()}
    edge_types = nx.get_edge_attributes(G, 'edge_type')
    edge_map_reverse = {value: key for key, value in edge_map.items()}
    edge_names = {key_edge: edge_map_reverse[value_edge] for key_edge, value_edge in edge_types.items()}
    return node_names, edge_names

def visualize_result(edge_index, edge_att, node_label, node_type, edge_label, edge_type, dataset_name, class_index, i, explainer_name, num_random_edges=100, random_seed=0):
    # 设置随机种子
    random.seed(random_seed)
    
    edge_att = edge_att**10
    edge_att = (edge_att - edge_att.min()) / (edge_att.max() - edge_att.min() + 1e-6)
    data = Data(edge_index=edge_index, att=edge_att, node_type=node_type, edge_type=edge_type, node_label=node_label, edge_label=edge_label, num_nodes=node_label.size(0)).to('cpu')
    G = to_networkx(data, node_attrs=['node_label', 'node_type'], edge_attrs=['att', 'edge_label', 'edge_type'])
    node_names, edge_names = get_labels_for_log_fusion(data, G, dataset_name)
    
    # 选择 edge_label 为 1 的边
    positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d['edge_label'] == 1]
    
    # 获取所有正边的节点集合
    positive_nodes = set([u for u, v in positive_edges] + [v for u, v in positive_edges])
    
    # 从 edge_label 为 0 的边中选择能够与正边相连的边
    candidate_negative_edges = [(u, v) for u, v, d in G.edges(data=True) if d['edge_label'] == 0 and (u in positive_nodes or v in positive_nodes)]
    
    # 随机选择一定数量的负边
    if len(candidate_negative_edges) > num_random_edges:
        negative_edges = random.sample(candidate_negative_edges, num_random_edges)
    else:
        negative_edges = candidate_negative_edges
    
    # 获取所有负边（edge_label 为 0 的边）
    all_negative_edges = [(u, v, d['att']) for u, v, d in G.edges(data=True) if d['edge_label'] == 0]
    
    # 根据 att 值排序并选择前 10 条边
    top_negative_edges = sorted(all_negative_edges, key=lambda x: x[2], reverse=True)[:10]
    top_negative_edges = [(u, v) for u, v, _ in top_negative_edges]
    
    # 合并边，确保没有重复
    selected_edges = list(set(positive_edges + negative_edges + top_negative_edges))
    
    # 创建新的子图，只包含选中的边
    subgraph = G.edge_subgraph(selected_edges)
    
    tmp_edge_names = [edge_names[(source, target)] for source, target in subgraph.edges()]
    tmp_node_labels = [node_label[i].item() for i in subgraph.nodes()]
    
    subgraph = nx.convert_node_labels_to_integers(subgraph)
    
    # 将NetworkX图转换为Pyecharts所需的格式，同时添加节点和边标签
    nodes = [{'name': node_names[i], 'symbolSize': 30 if tmp_node_labels[i] == 1 else 10, 'itemStyle': {'color': score_to_color(tmp_node_labels[i])}} for i in subgraph.nodes()]
    
    links = [{'source': source, 'target': target, "label": {"show": True, "formatter": tmp_edge_names[i]}} for i, (source, target) in enumerate(subgraph.edges())]
    # 根据边数据中的属性值设置边的透明度
    for link in links:
        source = int(link['source'])
        target = int(link['target'])
        link['lineStyle'] = {'color': score_to_color(subgraph[source][target]['att']), 'curveness': random.random() * 0.25}
    
    # 使用Pyecharts绘制图形
    graph = (
        Graph(init_opts=opts.InitOpts(width="100vw", height="100vh"))
        .add("", nodes, links, repulsion=8000)
        .set_global_opts(title_opts=opts.TitleOpts(title="NetworkX Graph with Pyecharts"))
    )
    
    picname = f"../data/{dataset_name}/common/pics/{explainer_name}/networkx_pyecharts_index{i}_type{class_index}.html"
    
    dir_path = os.path.dirname(picname)
    
    # 创建目录（如果不存在）
    os.makedirs(dir_path, exist_ok=True)
    
    # 保存图形
    graph.render(picname)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Baseline with GIN')
    parser.add_argument('--explainer', type=str, default="GraphMaskExplainer", help='Explainer of the Model') # ["GNNExplainer", "PGExplainer", "GraphMaskExplainer"]
    parser.add_argument('--dataset', type=str, default="Apache-1", help='Dataset') # ["Apache-1", "Apache-2", "ImageMagick-1", "ImageMagick-2"]
    args = parser.parse_args()
    num_seeds = 1
    num_edges = 25

    dataset_name = args.dataset

    data_dir = Path('../data')

    for random_state in range(num_seeds):
        set_seed(random_state)
        print(f"explainer: {args.explainer}, dataset: {args.dataset}, seed: {random_state}")

        data_dir = Path('../data')
        loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info = get_data_loaders(data_dir, dataset_name,
                                                                                        batch_size=192, random_state=random_state, mode='common', parameters=None,
                                                                                        splits={'train': 0.8, 'valid': 0.1,
                                                                                                'test': 0.1},
                                                                                        mutag_x=True if dataset_name == 'mutag' else False)

        model = GIN(x_dim, 64, num_class, num_layers=3)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        minloss = 0
        bEpoch = 0
        btrainacc = 0
        btestacc = 0
        bvalidacc = 0

        model_config = get_model_config()
        explainer = get_explainer(model, args.explainer, model_config)
        explainer.algorithm.to(device)

        for epoch in range(1, 401):
            epoch_loss = 0
            for data in loaders['train']:
                data = data.to(device)
                loss = train(model, data, optimizer)
                epoch_loss += loss

            if minloss == 0:
                minloss = epoch_loss
            train_acc = test(model, loaders['train'])
            test_acc = test(model, loaders['test'])
            valid_acc = test(model, loaders['valid'])  # 计算验证集准确率

            # 计算可解释性和 ROU_AUC 值
            if epoch_loss < minloss:
                minloss = epoch_loss
                bEpoch = epoch
                btrainacc = train_acc
                btestacc = test_acc
                bvalidacc = valid_acc  # 更新最佳验证集准确率
            print(f'Epoch: {epoch:03d}, Loss: {epoch_loss:.4f}, Train: {train_acc:.4f}, Valid: {valid_acc:.4f}, Test: {test_acc:.4f}')

        train_explanation, train_precision_at_k = calculate_explanation_accuracy(model, loaders['train'], explainer, args.explainer, args.dataset, "train", num_edges, random_state, False)
        valid_explanation, valid_precision_at_k = calculate_explanation_accuracy(model, loaders['valid'], explainer, args.explainer, args.dataset, "valid", num_edges, random_state, False)
        test_explanation, test_precision_at_k = calculate_explanation_accuracy(model, loaders['test'], explainer, args.explainer, args.dataset, "test", num_edges, random_state, True)

        print(f'Best Epoch: {bEpoch:03d}, Loss: {minloss:.4f}, Train: {btrainacc:.4f}, Test: {btestacc:.4f}, Valid: {bvalidacc:.4f}')
        print(f'Train Explanation: {train_explanation:.4f}, Valid Explanation: {valid_explanation:.4f}, Test Explanation: {test_explanation:.4f}')
        print(f'Train Precision at K: {train_precision_at_k:.4f}, Valid Precision at K: {valid_precision_at_k:.4f}, Test Precision at K: {test_precision_at_k:.4f}')

        filename = f"../data/logs/{dataset_name}/{dataset_name}_{args.explainer}_{random_state}.csv"
        
        dir_path = os.path.dirname(filename)

        # 创建目录（如果不存在）
        os.makedirs(dir_path, exist_ok=True)

        # Write the results to a CSV file
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([
                'explainer', 'dataset', 'seed', 'Best Epoch', 'Loss', 'Train', 'Test', 'Valid',
                'Train Explanation', 'Valid Explanation', 'Test Explanation',
                'Train Precision at K', 'Valid Precision at K', 'Test Precision at K'
            ])
            csvwriter.writerow([
                args.explainer, args.dataset, random_state, bEpoch, minloss, btrainacc, btestacc, bvalidacc,
                train_explanation, valid_explanation, test_explanation,
                train_precision_at_k, valid_precision_at_k, test_precision_at_k
            ])
            
if __name__ == '__main__':
    main()
