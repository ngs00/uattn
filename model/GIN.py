import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import GlobalAttention
from torch_geometric.nn import Set2Set
from model.UNGlobalAttention import UNGlobalAttention


class GIN(nn.Module):
    def __init__(self, num_node_feats, dim_out, readout='mean'):
        super(GIN, self).__init__()
        self.gc1 = GINConv(nn.Linear(num_node_feats, 256))
        self.bn1 = nn.BatchNorm1d(256)
        self.gc2 = GINConv(nn.Linear(256, 256))
        self.bn2 = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(256, 196)
        self.bn3 = nn.BatchNorm1d(196)
        self.fc2 = nn.Linear(196, dim_out)
        self.readout = readout

        if self.readout == 'attn':
            self.readout_func = GlobalAttention(nn.Linear(256, 1))
        elif self.readout == 'lstm':
            self.readout_func = Set2Set(256, 8)
            self.fc1 = nn.Linear(512, 196)
            self.bn3 = nn.BatchNorm1d(196)
        elif self.readout == 'uattn':
            self.readout_func = UNGlobalAttention(nn.Linear(256, 1))

    def forward(self, g):
        h = F.relu(self.bn1(self.gc1(g.x, g.edge_index)))
        h = F.relu(self.bn2(self.gc2(h, g.edge_index)))

        if self.readout == 'mean':
            hg = global_mean_pool(h, g.batch)
        elif self.readout == 'max':
            hg = global_max_pool(h, g.batch)
        elif self.readout == 'sum':
            hg = global_add_pool(h, g.batch)
        elif self.readout == 'attn' or self.readout == 'lstm' or self.readout == 'uattn':
            hg = self.readout_func(h, g.batch)

        h = F.relu(self.bn3(self.fc1(hg)))
        out = self.fc2(h)

        return out, hg
