# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/basic_gnn.html#GIN

import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, SAGPooling
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from .conv_layers import GINConv, GINEConv, RGCNConv


class GIN(nn.Module):
    def __init__(self, x_dim, edge_attr_dim, num_class, multi_label, model_config):
        super().__init__()

        self.n_layers = model_config['n_layers']
        hidden_size = model_config['hidden_size']
        self.edge_attr_dim = edge_attr_dim
        self.dropout_p = model_config['dropout_p']
        self.use_edge_attr = model_config.get('use_edge_attr', True)

        if model_config.get('atom_encoder', False):
            self.node_encoder = AtomEncoder(emb_dim=hidden_size)
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.edge_encoder = BondEncoder(emb_dim=hidden_size)
        else:
            self.node_encoder = Linear(x_dim, hidden_size)
            self.node_encoder2 = Linear(x_dim, hidden_size)
            self.node_encoder_recon = Linear(x_dim, hidden_size)
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.edge_encoder = Linear(edge_attr_dim, hidden_size)

        self.convs = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.convs_recon = nn.ModuleList()
        self.relu = nn.ReLU()
        self.pool = global_add_pool
        #self.pool = SAGPooling(in_channels=hidden_size, ratio=0.5)

        for _ in range(self.n_layers):
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.convs.append(GINEConv(GIN.MLP(hidden_size, hidden_size), edge_dim=hidden_size))
                self.convs2.append(GINEConv(GIN.MLP(hidden_size, hidden_size), edge_dim=hidden_size))
            else:
                self.convs.append(GINConv(GIN.MLP(hidden_size, hidden_size)))
                self.convs2.append(GINConv(GIN.MLP(hidden_size, hidden_size)))
        
        for _ in range(self.n_layers):
            self.convs_recon.append(RGCNConv(hidden_size, hidden_size, num_class))

        self.fc_out = nn.Sequential(nn.Linear(hidden_size, 1 if num_class == 2 and not multi_label else num_class))

    def forward(self, x, edge_index, batch, edge_attr=None, edge_atten=None, edge_type=None):
        x = self.node_encoder2(x)
        if edge_attr is not None and self.use_edge_attr:
            edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.n_layers):
            x = self.convs2[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x

    @staticmethod
    def MLP(in_channels: int, out_channels: int):
        return nn.Sequential(
            Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            Linear(out_channels, out_channels),
        )

    def get_attention_emb(self, x, edge_index, batch, edge_attr=None, edge_atten=None, edge_type=None):
        x = self.node_encoder(x)
        if edge_attr is not None and self.use_edge_attr:
            edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x
    
    def get_reconstruction_emb(self, x, edge_index, batch, edge_attr=None, edge_atten=None, edge_type=None):
        x = self.node_encoder_recon(x)
        if edge_attr is not None and self.use_edge_attr:
            edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.n_layers):
            x = self.convs_recon[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten, edge_type=edge_type)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x

    def get_pred_from_emb(self, emb, batch, edge_index):
        return self.fc_out(self.pool(emb, batch))
        #return self.fc_out(self.pool(emb, edge_index, batch=batch))
    
    def get_graph_emb_from_node_emb(self, emb, batch, edge_index):
        return self.pool(emb, batch)
        #return self.pool(emb, edge_index, batch=batch)