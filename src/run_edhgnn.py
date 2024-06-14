import yaml
import scipy
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_sparse import transpose
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph, is_undirected, to_networkx
import networkx as nx
from ogb.graphproppred import Evaluator
from sklearn.metrics import roc_auc_score
from rdkit import Chem
import random
import os
from torch_geometric.data import Data
from pyecharts.charts import Graph
from pyecharts import options as opts

from pretrain_clf import train_clf_one_seed
from utils import Writer, CriterionClf, CriterionRecon, MLP, SimpleMLP, visualize_a_graph, save_checkpoint, load_checkpoint, get_preds, get_lr, set_seed, process_data
from utils import get_local_config_name, get_model, get_data_loaders, write_stat_from_metric_dicts, reorder_like, init_metric_dict, generate_all_cases, get_labels_for_log_fusion, score_to_color


class DGIB(nn.Module):

    def __init__(self, gnn_t, gnn_s, extractor, discriminator, optimizer, scheduler, writer, device, model_dir, dataset_name, num_class, multi_label, random_state,
                 method_config, shared_config):
        super().__init__()
        self.gnn_t = gnn_t
        self.gnn_s = gnn_s
        self.extractor = extractor
        self.discriminator = discriminator
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.writer = writer
        self.device = device
        self.model_dir = model_dir
        self.dataset_name = dataset_name
        self.random_state = random_state
        self.method_name = method_config['method_name']

        self.learn_edge_att = shared_config['learn_edge_att']
        self.k = shared_config['precision_k']
        self.num_viz_samples = shared_config['num_viz_samples']
        self.viz_interval = shared_config['viz_interval']
        self.viz_norm_att = shared_config['viz_norm_att']

        self.epochs = method_config['epochs']
        self.pred_loss_coef = method_config['pred_loss_coef']
        self.recon_loss_coef = method_config['recon_loss_coef']
        self.joint_disc_loss_coef = method_config['joint_disc_loss_coef']
        
        self.joint_disc_loss2_coef = method_config['joint_disc_loss2_coef']
        self.marginal_disc_loss2_coef = method_config['marginal_disc_loss2_coef']

        self.fix_r = method_config.get('fix_r', None)
        self.decay_interval = method_config.get('decay_interval', None)
        self.decay_r = method_config.get('decay_r', None)
        self.final_r = method_config.get('final_r', 0.1)
        self.init_r = method_config.get('init_r', 0.9)

        self.multi_label = multi_label
        self.criterion_clf = CriterionClf(num_class, multi_label)
        self.criterion_recon = CriterionRecon()

    def __loss__(self, att, clf_logits, clf_labels, data_batch, emb_recon, h_t, h_s, epoch):
        pred_loss = self.criterion_clf(clf_logits, clf_labels)

        # 获取batch中所有图的节点数平方总和
        total_num_nodes = 0

        # 初始化重构损失
        recon_loss = 0.0

        # 遍历每张图
        start_idx = 0  # emb_recon中当前图的起始索引
        for data in data_batch.to_data_list():
            # 获取当前图的节点数
            num_nodes = data.num_nodes

            total_num_nodes += num_nodes*num_nodes

            # 提取当前图的嵌入
            z = emb_recon[start_idx:start_idx+num_nodes]

            # 计算重构邻接矩阵
            adj_reconstructed = torch.sigmoid(torch.mm(z, z.t()))

            # 提取当前图的边索引
            edge_index = data.edge_index

            # 构建原始邻接矩阵，讨论点2：重构需不需要考虑多重边？
            adj_original = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]).to(edge_index.device), size=(num_nodes, num_nodes))
            adj_original = torch.clamp(adj_original.to_dense(), max=1)  # 将大于1的元素截断为1，确保为0-1矩阵

            # recon_loss = self.criterion_recon(adj_reconstructed, adj_original)

            # 计算逐元素平方差
            recon_loss += torch.sum((adj_reconstructed - adj_original) ** 2)

            # 更新emb_recon中当前图的起始索引
            start_idx += num_nodes

        # 计算平均重构损失
        recon_loss /= total_num_nodes

        # recon_loss *= 5

        joint_disc = self.discriminator(h_t, h_s)

        joint_disc_loss = -torch.mean(torch.log(joint_disc))

        r = self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r)
        info_loss = (att * torch.log(att/r + 1e-6) + (1-att) * torch.log((1-att)/(1-r+1e-6) + 1e-6)).mean()

        pred_loss = pred_loss * self.pred_loss_coef
        recon_loss = recon_loss * self.recon_loss_coef
        joint_disc_loss = joint_disc_loss * self.joint_disc_loss_coef
    
        loss1 = pred_loss + recon_loss + joint_disc_loss

        loss_dict = {'loss1': loss1.item(), 'pred': pred_loss.item(), 'recon': recon_loss.item(), 'joint_disc_loss': joint_disc_loss.item()}
        return loss1, loss_dict
    
    def __loss2__(self, h_t, h_s):
        joint_disc = self.discriminator(h_t, h_s)
        perm_indices = torch.randperm(h_s.shape[0])
        marginal_disc = self.discriminator(h_t, h_s[perm_indices])
        joint_disc_loss2 = torch.mean(-torch.log(1.0 - joint_disc))
        marginal_disc_loss2 = torch.mean(-torch.log(marginal_disc))
        
        joint_disc_loss2 = joint_disc_loss2 * self.joint_disc_loss2_coef
        marginal_disc_loss2 = marginal_disc_loss2 * self.marginal_disc_loss2_coef

        loss2 = joint_disc_loss2 + marginal_disc_loss2
        loss_dict = {'loss2': loss2.item(), 'joint2': joint_disc_loss2.item(), 'marginal2': marginal_disc_loss2.item()}
        return loss2, loss_dict

    def forward_pass(self, data, epoch, training):
        if not hasattr(data, 'edge_type'):
            emb_t = self.gnn_t.get_attention_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
            emb_s = self.gnn_s.get_attention_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
        else:
            emb_t = self.gnn_t.get_attention_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr, edge_type=data.edge_type)
            emb_s = self.gnn_s.get_attention_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr, edge_type=data.edge_type)

        att_log_logits_t = self.extractor(emb_t, data.edge_index, data.batch)
        att_log_logits_s = self.extractor(emb_s, data.edge_index, data.batch)
        att_t = self.sampling(att_log_logits_t, training)
        att_s = self.sampling(att_log_logits_s, training)

        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                trans_idx, trans_val = transpose(data.edge_index, att_t, None, None, coalesced=False)
                trans_val_perm = reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att_t = (att_t + trans_val_perm) / 2

                trans_idx, trans_val = transpose(data.edge_index, att_s, None, None, coalesced=False)
                trans_val_perm = reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att_s = (att_s + trans_val_perm) / 2
            else:
                edge_att_t = att_t
                edge_att_s = att_s
        else:
            edge_att_t = self.lift_node_att_to_edge_att(att_t, data.edge_index)
            edge_att_s = self.lift_node_att_to_edge_att(att_s, data.edge_index)
        if not hasattr(data, 'edge_type_recon'):
            edge_type_recon = data.y[data.batch[data.edge_index[0]]].squeeze().int()
        else:
            edge_type_recon = data.edge_type_recon
        if not hasattr(data, 'edge_type'):
            emb_disc_t = self.gnn_t(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att_t)
            emb_disc_s = self.gnn_s(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att_s)
            emb_recon = self.gnn_s.get_reconstruction_emb(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att_s, edge_type=edge_type_recon)
        else:
            emb_disc_t = self.gnn_t(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att_t, edge_type=data.edge_type)
            emb_disc_s = self.gnn_s(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att_s, edge_type=data.edge_type)
            emb_recon = self.gnn_s.get_reconstruction_emb(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att_s, edge_type=edge_type_recon)

        # TODO: 模型提点的方式2：clf_logits所用的emb是否需要再另起一组卷积参数？
        clf_logits = self.gnn_t.get_pred_from_emb(emb_disc_t, data.batch, data.edge_index)
        h_t = self.gnn_t.get_graph_emb_from_node_emb(emb_disc_t, data.batch, data.edge_index)
        h_s = self.gnn_s.get_graph_emb_from_node_emb(emb_disc_s, data.batch, data.edge_index)
        loss1, loss_dict = self.__loss__(att_t, clf_logits, data.y, data, emb_recon, h_t, h_s, epoch)
        return edge_att_t, loss1, loss_dict, clf_logits
    
    def forward_pass2(self, data, epoch, training):
        if not hasattr(data, 'edge_type'):
            emb_t = self.gnn_t.get_attention_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
            emb_s = self.gnn_s.get_attention_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
        else:
            emb_t = self.gnn_t.get_attention_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr, edge_type=data.edge_type)
            emb_s = self.gnn_s.get_attention_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr, edge_type=data.edge_type)

        h_t = self.gnn_t.get_graph_emb_from_node_emb(emb_t, data.batch, data.edge_index)
        h_s = self.gnn_s.get_graph_emb_from_node_emb(emb_s, data.batch, data.edge_index)
        loss2, loss_dict = self.__loss2__(h_t, h_s)
        return loss2, loss_dict


    @torch.no_grad()
    def eval_one_batch(self, data, epoch):
        self.extractor.eval()
        self.gnn_t.eval()
        self.gnn_s.eval()
        self.discriminator.eval()

        att, loss1, loss_dict, clf_logits = self.forward_pass(data, epoch, training=False)
        loss2, loss_dict_2 = self.forward_pass2(data, epoch, training=False)

        return att.data.cpu().reshape(-1), {**loss_dict, **loss_dict_2}, clf_logits.data.cpu()

    def train_one_batch(self, data, epoch):
        self.extractor.train()
        self.gnn_t.train()
        self.gnn_s.train()
        self.discriminator.train()
        att, loss1, loss_dict, clf_logits = self.forward_pass(data, epoch, training=True)
        
        self.optimizer.zero_grad()
        loss1.backward(retain_graph=True)
        self.optimizer.step()

        loss2, loss_dict_2 = self.forward_pass2(data, epoch, training=True)

        self.optimizer.zero_grad()
        loss2.backward()
        self.optimizer.step()
        return att.data.cpu().reshape(-1), {**loss_dict, **loss_dict_2}, clf_logits.data.cpu()

    def run_one_epoch(self, data_loader, epoch, phase, use_edge_attr):
        loader_len = len(data_loader)
        run_one_batch = self.train_one_batch if phase == 'train' else self.eval_one_batch
        phase = 'test ' if phase == 'test' else phase  # align tqdm desc bar

        all_loss_dict = {}
        all_exp_labels, all_att, all_clf_labels, all_clf_logits, all_precision_at_k = ([] for i in range(5))
        pbar = tqdm(data_loader)
        for idx, data in enumerate(pbar):
            data = process_data(data, use_edge_attr)
            att, loss_dict, clf_logits = run_one_batch(data.to(self.device), epoch)

            exp_labels = data.edge_label.data.cpu()
            precision_at_k = self.get_precision_at_k(att, exp_labels, self.k, data.batch, data.edge_index)
            desc, _, _, _, _, _ = self.log_epoch(epoch, phase, loss_dict, exp_labels, att, precision_at_k,
                                                 data.y.data.cpu(), clf_logits, batch=True)
            for k, v in loss_dict.items():
                all_loss_dict[k] = all_loss_dict.get(k, 0) + v

            all_exp_labels.append(exp_labels), all_att.append(att), all_precision_at_k.extend(precision_at_k)
            all_clf_labels.append(data.y.data.cpu()), all_clf_logits.append(clf_logits)

            if idx == loader_len - 1:
                all_exp_labels, all_att = torch.cat(all_exp_labels), torch.cat(all_att),
                all_clf_labels, all_clf_logits = torch.cat(all_clf_labels), torch.cat(all_clf_logits)

                for k, v in all_loss_dict.items():
                    all_loss_dict[k] = v / loader_len
                desc, att_auroc, precision, clf_acc, clf_roc, avg_loss = self.log_epoch(epoch, phase, all_loss_dict, all_exp_labels, all_att,
                                                                                        all_precision_at_k, all_clf_labels, all_clf_logits, batch=False)
            pbar.set_description(desc)
        return att_auroc, precision, clf_acc, clf_roc, avg_loss

    def train(self, loaders, test_set, metric_dict, use_edge_attr):
        viz_set = self.get_viz_idx(test_set, self.dataset_name)
        for epoch in range(self.epochs):
            train_res = self.run_one_epoch(loaders['train'], epoch, 'train', use_edge_attr)
            valid_res = self.run_one_epoch(loaders['valid'], epoch, 'valid', use_edge_attr)
            test_res = self.run_one_epoch(loaders['test'], epoch, 'test', use_edge_attr)
            self.writer.add_scalar('dgib_train/lr', get_lr(self.optimizer), epoch)

            assert len(train_res) == 5
            main_metric_idx = 3 if 'ogb' in self.dataset_name else 2  # clf_roc or clf_acc
            if self.scheduler is not None:
                self.scheduler.step(valid_res[main_metric_idx])

            # r = self.fix_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r)
            if epoch > 10 and ((valid_res[main_metric_idx] > metric_dict['metric/best_clf_valid'])
                                                                     or (valid_res[main_metric_idx] == metric_dict['metric/best_clf_valid']
                                                                         and valid_res[4] < metric_dict['metric/best_clf_valid_loss'])):

                metric_dict = {'metric/best_clf_epoch': epoch, 'metric/best_clf_valid_loss': valid_res[4],
                               'metric/best_clf_train': train_res[main_metric_idx], 'metric/best_clf_valid': valid_res[main_metric_idx], 'metric/best_clf_test': test_res[main_metric_idx],
                               'metric/best_x_roc_train': train_res[0], 'metric/best_x_roc_valid': valid_res[0], 'metric/best_x_roc_test': test_res[0],
                               'metric/best_x_precision_train': train_res[1], 'metric/best_x_precision_valid': valid_res[1], 'metric/best_x_precision_test': test_res[1]}
                """ save_checkpoint(self.gnn_t, self.model_dir, model_name='dgib_gnn_t_epoch_' + str(epoch))
                save_checkpoint(self.gnn_s, self.model_dir, model_name='dgib_gnn_s_epoch_' + str(epoch))
                save_checkpoint(self.extractor, self.model_dir, model_name='dgib_att_epoch_' + str(epoch))
                save_checkpoint(self.discriminator, self.model_dir, model_name='dgib_discriminator_epoch_' + str(epoch)) """
            for metric, value in metric_dict.items():
                metric = metric.split('/')[-1]
                self.writer.add_scalar(f'dgib_best/{metric}', value, epoch)

            if self.num_viz_samples != 0 and epoch == self.epochs - 1:
                if self.multi_label:
                    raise NotImplementedError
                for idx, tag in viz_set:
                    self.visualize_results(test_set, idx, epoch, tag, use_edge_attr)

            if epoch == self.epochs - 1:
                """ save_checkpoint(self.gnn_t, self.model_dir, model_name='dgib_gnn_t_epoch_' + str(epoch))
                save_checkpoint(self.gnn_s, self.model_dir, model_name='dgib_gnn_s_epoch_' + str(epoch))
                save_checkpoint(self.extractor, self.model_dir, model_name='dgib_att_epoch_' + str(epoch))
                save_checkpoint(self.discriminator, self.model_dir, model_name='dgib_discriminator_epoch_' + str(epoch)) """

            print(f'[Seed {self.random_state}, Epoch: {epoch}]: Best Epoch: {metric_dict["metric/best_clf_epoch"]}, '
                  f'Best Val Pred ACC/ROC: {metric_dict["metric/best_clf_valid"]:.3f}, Best Test Pred ACC/ROC: {metric_dict["metric/best_clf_test"]:.3f}, '
                  f'Best Test X AUROC: {metric_dict["metric/best_x_roc_test"]:.3f}')
            print('====================================')
            print('====================================')
        return metric_dict

    def log_epoch(self, epoch, phase, loss_dict, exp_labels, att, precision_at_k, clf_labels, clf_logits, batch):
        desc = f'[Seed {self.random_state}, Epoch: {epoch}]: dgib_{phase}........., ' if batch else f'[Seed {self.random_state}, Epoch: {epoch}]: dgib_{phase} finished, '
        for k, v in loss_dict.items():
            if not batch:
                self.writer.add_scalar(f'dgib_{phase}/{k}', v, epoch)
            desc += f'{k}: {v:.3f}, '

        eval_desc, att_auroc, precision, clf_acc, clf_roc = self.get_eval_score(epoch, phase, exp_labels, att, precision_at_k, clf_labels, clf_logits, batch)
        desc += eval_desc
        return desc, att_auroc, precision, clf_acc, clf_roc, loss_dict['pred']

    def get_eval_score(self, epoch, phase, exp_labels, att, precision_at_k, clf_labels, clf_logits, batch):
        clf_preds = get_preds(clf_logits, self.multi_label)
        clf_acc = 0 if self.multi_label else (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]

        if batch:
            return f'clf_acc: {clf_acc:.3f}', None, None, None, None

        precision_at_k = np.mean(precision_at_k)
        clf_roc = 0
        if 'ogb' in self.dataset_name:
            evaluator = Evaluator(name='-'.join(self.dataset_name.split('_')))
            clf_roc = evaluator.eval({'y_pred': clf_logits, 'y_true': clf_labels})['rocauc']

        att_auroc, bkg_att_weights, signal_att_weights = 0, att, att
        if np.unique(exp_labels).shape[0] > 1:
            att_auroc = roc_auc_score(exp_labels, att)
            bkg_att_weights = att[exp_labels == 0]
            signal_att_weights = att[exp_labels == 1]
        #self.writer.add_histogram(f'dgib_{phase}/bkg_att_weights', bkg_att_weights, epoch)
        #self.writer.add_histogram(f'dgib_{phase}/signal_att_weights', signal_att_weights, epoch)
        self.writer.add_scalar(f'dgib_{phase}/clf_acc/', clf_acc, epoch)
        self.writer.add_scalar(f'dgib_{phase}/clf_roc/', clf_roc, epoch)
        self.writer.add_scalar(f'dgib_{phase}/att_auroc/', att_auroc, epoch)
        self.writer.add_scalar(f'dgib_{phase}/precision@{self.k}/', precision_at_k, epoch)
        self.writer.add_scalar(f'dgib_{phase}/avg_bkg_att_weights/', bkg_att_weights.mean(), epoch)
        self.writer.add_scalar(f'dgib_{phase}/avg_signal_att_weights/', signal_att_weights.mean(), epoch)
        self.writer.add_pr_curve(f'PR_Curve/dgib_{phase}/', exp_labels, att, epoch)

        desc = f'clf_acc: {clf_acc:.3f}, clf_roc: {clf_roc:.3f}, ' + \
               f'att_roc: {att_auroc:.3f}, att_prec@{self.k}: {precision_at_k:.3f}'
        return desc, att_auroc, precision_at_k, clf_acc, clf_roc

    def get_precision_at_k(self, att, exp_labels, k, batch, edge_index):
        precision_at_k = []
        for i in range(batch.max()+1):
            nodes_for_graph_i = batch == i
            edges_for_graph_i = nodes_for_graph_i[edge_index[0]] & nodes_for_graph_i[edge_index[1]]
            labels_for_graph_i = exp_labels[edges_for_graph_i]
            mask_log_logits_for_graph_i = att[edges_for_graph_i]
            precision_at_k.append(labels_for_graph_i[np.argsort(-mask_log_logits_for_graph_i)[:k]].sum().item() / k)
        return precision_at_k

    def get_viz_idx(self, test_set, dataset_name):
        y_dist = test_set.data.y.numpy().reshape(-1)
        num_nodes = np.array([each.x.shape[0] for each in test_set])
        classes = np.unique(y_dist)
        res = []
        for each_class in classes:
            tag = 'class_' + str(each_class)
            if dataset_name == 'Graph-SST2':
                condi = (y_dist == each_class) * (num_nodes > 5) * (num_nodes < 10)  # in case too short or too long
                candidate_set = np.nonzero(condi)[0]
            else:
                candidate_set = np.nonzero(y_dist == each_class)[0]
            idx = np.random.choice(candidate_set, self.num_viz_samples, replace=False) if len(candidate_set) >= self.num_viz_samples else candidate_set
            res.append((idx, tag))
        return res

    def visualize_result(edge_index, edge_att, node_label, node_type, edge_label, edge_type, dataset_name, class_index, i, num_random_edges=100, random_seed=0):
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
        
        picname = f"../data/{dataset_name}/common/pics/DGIB/index{i}_type{class_index}.html"
        
        dir_path = os.path.dirname(picname)
        
        # 创建目录（如果不存在）
        os.makedirs(dir_path, exist_ok=True)
        
        # 保存图形
        graph.render(picname)
    
    def visualize_results(self, test_set, idx, epoch, tag, use_edge_attr):
        viz_set = test_set[idx]
        data = next(iter(DataLoader(viz_set, batch_size=len(idx), shuffle=False)))
        data = process_data(data, use_edge_attr)
        batch_att, _, clf_logits = self.eval_one_batch(data.to(self.device), epoch)
        imgs = []
        for i in tqdm(range(len(viz_set))):
            mol_type, coor = None, None
            if self.dataset_name == 'mutag':
                node_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S', 8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
                mol_type = {k: node_dict[v.item()] for k, v in enumerate(viz_set[i].node_type)}
            elif self.dataset_name == 'Graph-SST2':
                mol_type = {k: v for k, v in enumerate(viz_set[i].sentence_tokens)}
                num_nodes = data.x.shape[0]
                x = np.linspace(0, 1, num_nodes)
                y = np.ones_like(x)
                coor = np.stack([x, y], axis=1)
            elif self.dataset_name == 'ogbg_molhiv':
                element_idxs = {k: int(v+1) for k, v in enumerate(viz_set[i].x[:, 0])}
                mol_type = {k: Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), int(v)) for k, v in element_idxs.items()}
            elif self.dataset_name == 'mnist':
                raise NotImplementedError

            node_subset = data.batch == i
            _, edge_att = subgraph(node_subset, data.edge_index, edge_attr=batch_att)

            node_label = viz_set[i].node_label.reshape(-1) if viz_set[i].get('node_label', None) is not None else torch.zeros(viz_set[i].x.shape[0])
            fig, img = self.visualize_result(viz_set[i].edge_index, edge_att, node_label, viz_set[i].node_type, viz_set[i].edge_label, viz_set[i].edge_type, self.dataset_name, int(viz_set[i].y.item()), i)
        """     imgs.append(img)
        imgs = np.stack(imgs)
        #self.writer.add_images(tag, imgs, epoch, dataformats='NHWC') """

    def get_r(self, decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r

    @staticmethod
    def sampling(att_log_logit, training):
        temp = 1
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att


class ExtractorMLP(nn.Module):

    def __init__(self, hidden_size, shared_config):
        super().__init__()
        self.learn_edge_att = shared_config['learn_edge_att']
        dropout_p = shared_config['extractor_dropout_p']

        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=dropout_p)
        else:
            self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=dropout_p)

    def forward(self, emb, edge_index, batch):
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12, batch[col])
        else:
            att_log_logits = self.feature_extractor(emb, batch)
        return att_log_logits

class Discriminator(nn.Module):

    def __init__(self, h_dim):
        super().__init__()
        self.discriminator = SimpleMLP([h_dim * 2, h_dim * 4, h_dim, 1], dropout=0.5)

    def forward(self, h_t, h_s):
        h = torch.cat((h_t, h_s), dim = 1)
        discriminated_logits = self.discriminator(h)
        return discriminated_logits

def train_dgib_one_seed(local_config, data_dir, log_dir, mode, params, model_name, dataset_name, method_name, device, random_state):
    print('====================================')
    print('====================================')
    print(f'[INFO] Using device: {device}')
    print(f'[INFO] Using random_state: {random_state}')
    print(f'[INFO] Using dataset: {dataset_name}')
    print(f'[INFO] Using model: {model_name}')

    set_seed(random_state)

    model_config = local_config['model_config']
    data_config = local_config['data_config']
    method_config = local_config[f'{method_name}_config']
    shared_config = local_config['shared_config']
    assert model_config['model_name'] == model_name
    assert method_config['method_name'] == method_name

    batch_size, splits = data_config['batch_size'], data_config.get('splits', None)
    loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info = get_data_loaders(data_dir, dataset_name, batch_size, splits, random_state, mode, params, data_config.get('mutag_x', False))
    first_batch = next(iter(loaders['train']))
    first_graph = first_batch.to_data_list()[0]
    print(first_batch)
    print(first_graph)

    model_config['deg'] = aux_info['deg']
    gnn_t = get_model(x_dim, edge_attr_dim, num_class, aux_info['multi_label'], model_config, device, aux_info['num_relations'])
    gnn_s = get_model(x_dim, edge_attr_dim, num_class, aux_info['multi_label'], model_config, device, aux_info['num_relations'])
    print('====================================')
    print('====================================')

    log_dir.mkdir(parents=True, exist_ok=True)
    print('[INFO] Training both the model and the attention from scratch...')

    extractor = ExtractorMLP(model_config['hidden_size'], shared_config).to(device)
    discriminator = Discriminator(model_config['hidden_size']).to(device)

    lr, wd = method_config['lr'], method_config.get('weight_decay', 0)
    optimizer = torch.optim.Adam(list(extractor.parameters()) + list(gnn_t.parameters()) + list(gnn_s.parameters()) + list(discriminator.parameters()), lr=lr, weight_decay=wd)

    scheduler_config = method_config.get('scheduler', {})
    scheduler = None if scheduler_config == {} else ReduceLROnPlateau(optimizer, mode='max', **scheduler_config)

    writer = Writer(log_dir=log_dir)
    hparam_dict = {**model_config, **data_config}
    hparam_dict = {k: str(v) if isinstance(v, (dict, list)) else v for k, v in hparam_dict.items()}
    metric_dict = deepcopy(init_metric_dict)
    writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)
    print(hparam_dict)

    print('====================================')
    print('[INFO] Training DGIB...')
    dgib = DGIB(gnn_t, gnn_s, extractor, discriminator, optimizer, scheduler, writer, device, log_dir, dataset_name, num_class, aux_info['multi_label'], random_state, method_config, shared_config)
    metric_dict = dgib.train(loaders, test_set, metric_dict, model_config.get('use_edge_attr', True))
    writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)
    return hparam_dict, metric_dict


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train DGIB')
    parser.add_argument('--dataset', type=str, help='dataset used')
    parser.add_argument('--backbone', type=str, default="GIN", help='backbone model used')
    parser.add_argument('--cuda', type=int, default=0, help='cuda device id, -1 for cpu')
    parser.add_argument('--batch_size', type=str, default="192", help='batch size')
    parser.add_argument('--hidden_size', type=str, default="192", help='hidden size')
    parser.add_argument('--dropout_p', type=str, default="0.3", help='dropout portion')
    parser.add_argument('--extractor_dropout_p', type=str, default="0.5",  help='extractor dropout portion')
    parser.add_argument('--lr', type=str, default="1.0e-3", help='learning rate')
    parser.add_argument('--disc_coef', type=str, default="3", help='coefficient of discriminator')
    parser.add_argument('--pred_coef', type=str, default="1", help='coefficient of prediction')
    parser.add_argument('--recon_coef', type=str, default="1", help='coefficient of reconstruction')
    


    args = parser.parse_args()
    dataset_name = args.dataset
    model_name = args.backbone
    cuda_id = args.cuda

    torch.set_num_threads(5)
    config_dir = Path('./configs')
    method_name = 'DGIB'

    print('====================================')
    print('====================================')
    print(f'[INFO] Running {method_name} on {dataset_name} with {model_name}')
    print('====================================')

    global_config = yaml.safe_load((config_dir / 'global_config.yml').open('r'))
    local_config_name = get_local_config_name(model_name, dataset_name)
    local_config = yaml.safe_load((config_dir / local_config_name).open('r'))

    local_config['DGIB_config']['lr'] = float(args.lr)
    local_config['DGIB_config']['pred_loss_coef'] = float(args.pred_coef)
    local_config['DGIB_config']['recon_loss_coef'] = float(args.recon_coef)
    local_config['DGIB_config']['joint_disc_loss_coef'] = float(args.disc_coef)
    local_config['DGIB_config']['joint_disc_loss2_coef'] = float(args.disc_coef)
    local_config['DGIB_config']['marginal_disc_loss2_coef'] = float(args.disc_coef)
    local_config['data_config']['batch_size'] = int(args.batch_size)
    local_config['model_config']['hidden_size'] = int(args.hidden_size)
    local_config['model_config']['dropout_p'] = float(args.dropout_p)
    local_config['shared_config']['extractor_dropout_p'] = float(args.extractor_dropout_p)

    data_dir = Path(global_config['data_dir'])
    num_seeds = global_config['num_seeds']

    time = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')

    all_cases = generate_all_cases(local_config['mode_config'], dataset_name)
    for case in all_cases:
        mode = case[0]
        params = case[1]
        dir = case[2]
        metric_dicts = []
        for random_state in range(num_seeds):
            log_dir = data_dir / dataset_name / dir /'logs' / (time + '-' + dataset_name + '-' + model_name + '-seed' + str(random_state) + '-' + args.lr + "-" + args.batch_size + "-" + args.hidden_size + "-" + args.dropout_p + "-" + args.extractor_dropout_p + '-' + args.pred_coef + '-' + args.recon_coef + '-' + args.disc_coef + '-' + method_name)
            hparam_dict, metric_dict = train_dgib_one_seed(local_config, data_dir, log_dir, mode, params, model_name, dataset_name, method_name, device, random_state)
            metric_dicts.append(metric_dict)

        log_dir = data_dir / dataset_name / dir /'logs' / (time + '-' + dataset_name + '-' + model_name + '-seed99-'+ args.lr + "-" + args.batch_size + "-" + args.hidden_size + "-" + args.dropout_p + "-" + args.extractor_dropout_p + '-' + args.pred_coef + '-' + args.recon_coef + '-' + args.disc_coef + '-' + method_name + '-stat')
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = Writer(log_dir=log_dir)
        write_stat_from_metric_dicts(hparam_dict, metric_dicts, writer)


if __name__ == '__main__':
    main()
