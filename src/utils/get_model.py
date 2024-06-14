import torch.nn as nn
import torch.nn.functional as F
from models import GIN, PNA, SPMotifNet, RGCN
from torch_geometric.nn import InstanceNorm


def get_model(x_dim, edge_attr_dim, num_class, multi_label, model_config, device, num_relations=None):
    if model_config['model_name'] == 'GIN':
        model = GIN(x_dim, edge_attr_dim, num_class, multi_label, model_config)
    elif model_config['model_name'] == 'PNA':
        model = PNA(x_dim, edge_attr_dim, num_class, multi_label, model_config)
    elif model_config['model_name'] == 'RGCN':
        model = RGCN(x_dim, edge_attr_dim, num_class, multi_label, model_config, num_relations)
    elif model_config['model_name'] == 'SPMotifNet':
        model = SPMotifNet(x_dim, edge_attr_dim, num_class, multi_label, model_config)
    else:
        raise ValueError('[ERROR] Unknown model name!')
    return model.to(device)


class CriterionClf(nn.Module):
    def __init__(self, num_class, multi_label):
        super(CriterionClf, self).__init__()
        self.num_class = num_class
        self.multi_label = multi_label
        print(f'[INFO] Using multi_label: {self.multi_label}')

    def forward(self, logits, targets):
        if self.num_class == 2 and not self.multi_label:
            loss = F.binary_cross_entropy_with_logits(logits, targets.float())
        elif self.num_class > 2 and not self.multi_label:
            loss = F.cross_entropy(logits, targets.long())
        else:
            is_labeled = targets == targets  # mask for labeled data
            loss = F.binary_cross_entropy_with_logits(logits[is_labeled], targets[is_labeled].float())
        return loss

class CriterionRecon(nn.Module):
    def __init__(self):
        super(CriterionRecon, self).__init__()

    def forward(self, adj_reconstructed, adj_original):
        loss = F.cross_entropy(adj_reconstructed, adj_original)
        return loss


def get_preds(logits, multi_label):
    if multi_label:
        print(logits.sigmoid())
        preds = (logits.sigmoid() > 0.5).float()
        print(preds)
    elif logits.shape[1] > 1:  # multi-class
        preds = logits.argmax(dim=1).float()
    else:  # binary
        preds = (logits.sigmoid() > 0.5).float()
    return preds


class BatchSequential(nn.Sequential):
    def forward(self, inputs, batch = None):
        for module in self._modules.values():
            if isinstance(module, (InstanceNorm)):
                inputs = module(inputs, batch)
            else:
                inputs = module(inputs)
        return inputs


class MLP(BatchSequential):
    def __init__(self, channels, dropout, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                m.append(InstanceNorm(channels[i]))
                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*m)

class SimpleMLP(BatchSequential):
    def __init__(self, channels, dropout, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                # 这里将InstanceNorm相关的部分移除，但是要加LayerNorm！！！
                m.append(nn.LayerNorm(channels[i]))
                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))
        m.append(nn.Sigmoid())
        super(SimpleMLP, self).__init__(*m)