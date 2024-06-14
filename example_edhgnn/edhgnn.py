import sys
sys.path.append('../src')

import scipy
import torch
import torch.nn as nn
import numpy as np
from torch_sparse import transpose
from torch_geometric.utils import is_undirected
from utils import MLP, reorder_like, SimpleMLP


class DGIB(nn.Module):

    def __init__(self, gnn_t, gnn_s, extractor, discriminator, criterion_clf, criterion_recon, optimizer, learn_edge_att=True, final_r=0.7, decay_interval=10, decay_r=0.1):
        super().__init__()
        self.gnn_t = gnn_t
        self.gnn_s = gnn_s
        self.extractor = extractor
        self.discriminator = discriminator
        self.criterion_clf = criterion_clf
        self.criterion_recon = criterion_recon
        self.optimizer = optimizer
        self.device = next(self.parameters()).device

        self.learn_edge_att = learn_edge_att
        self.final_r = final_r
        self.decay_interval = decay_interval
        self.decay_r = decay_r

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
            # TODO: 思考：要不要截断为1？
            adj_original = torch.clamp(adj_original.to_dense(), max=1)  # 将大于1的元素截断为1，确保为0-1矩阵

            # TODO: 模型提点的方式1：修改损失函数为交叉熵损失
            # recon_loss = self.criterion_recon(adj_reconstructed, adj_original)
            # 计算逐元素平方差
            recon_loss += torch.sum((adj_reconstructed - adj_original) ** 2)

            # 更新emb_recon中当前图的起始索引
            start_idx += num_nodes

        # 计算平均重构损失
        recon_loss /= total_num_nodes

        # recon_loss *= 5

        joint_disc = self.discriminator(h_t, h_s)

        joint_disc_loss = 5 * -torch.mean(torch.log(joint_disc))

        r = self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r)
        info_loss = (att * torch.log(att/r + 1e-6) + (1-att) * torch.log((1-att)/(1-r+1e-6) + 1e-6)).mean()

        loss1 = pred_loss + recon_loss + joint_disc_loss

        loss_dict = {'loss1': loss1.item(), 'pred': pred_loss.item(), 'recon': recon_loss.item(), 'joint_disc': joint_disc_loss.item()}
        return loss1, loss_dict
    
    def __loss2__(self, h_t, h_s):
        joint_disc = self.discriminator(h_t, h_s)
        perm_indices = torch.randperm(h_s.shape[0])
        marginal_disc = self.discriminator(h_t, h_s[perm_indices])
        joint_disc_loss2 = 5 * torch.mean(-torch.log(1.0 - joint_disc))
        marginal_disc_loss2 = 5 * torch.mean(-torch.log(marginal_disc))
        loss2 = joint_disc_loss2 + marginal_disc_loss2
        loss_dict = {'loss2': loss2.item(), 'joint_disc2': joint_disc_loss2.item(), 'marginal_disc2': marginal_disc_loss2.item()}
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
    def get_r(decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att


class ExtractorMLP(nn.Module):

    def __init__(self, hidden_size, learn_edge_att):
        super().__init__()
        self.learn_edge_att = learn_edge_att

        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=0.5)
        else:
            self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=0.5)

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
        self.discriminator = SimpleMLP([h_dim * 2, h_dim * 4, h_dim, 1], dropout=0.3)

    def forward(self, h_t, h_s):
        h = torch.cat((h_t, h_s), dim = 1)
        discriminated_logits = self.discriminator(h)
        return discriminated_logits