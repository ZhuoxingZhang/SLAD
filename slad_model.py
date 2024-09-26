import torch
import torch.nn.functional as F
from imblearn.over_sampling import SMOTE
from torch import nn
from torch.nn import ModuleList
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, GINConv

from resource_stat import stat_resource_usage


def balance_features_labels(features, labels, over_sample_scale_factor, sample_method):
    count_0 = torch.sum(labels == 0).item()
    count_1 = torch.sum(labels == 1).item()

    minority_label = 0 if count_0 < count_1 else 1
    minority_indices = torch.where(labels == minority_label)[0]

    if len(minority_indices) <= 1 or over_sample_scale_factor == 0:
        return features, labels

    if sample_method == "SMOTE":
        """SMOTE sampling"""
        n_neighbor = len(minority_indices) - 1 if len(
            minority_indices) - 1 < over_sample_scale_factor else over_sample_scale_factor
        smote = SMOTE(sampling_strategy={minority_label: len(minority_indices) * over_sample_scale_factor},
                      random_state=42, k_neighbors=n_neighbor)
        features, labels = smote.fit_resample(features.detach().cpu().numpy(), labels.detach().cpu().numpy())

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        features = torch.tensor(features, dtype=torch.float32, device=device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        return features, labels
    if sample_method == "copy":
        """copy minority"""
        selected_indices = minority_indices.repeat(over_sample_scale_factor)
        selected_features = features[selected_indices]
        features = torch.cat([features, selected_features], dim=0)
        added_labels = torch.ones(len(minority_indices) * over_sample_scale_factor, dtype=labels.dtype) * minority_label
        if torch.cuda.is_available():
            added_labels = added_labels.cuda()
        labels = torch.cat([labels, added_labels])
        return features, labels


class MultiLayerGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_sizes, activation_fuc):
        super(MultiLayerGCN, self).__init__()
        self.activation = activation_fuc
        self.conv_layers = torch.nn.ModuleList()
        self.conv_layers.append(GCNConv(num_features, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.conv_layers.append(GCNConv(hidden_sizes[i - 1], hidden_sizes[i]))

    def forward(self, x, edge_index):
        for conv in self.conv_layers:
            if self.activation == 'relu':
                x = F.relu(conv(x, edge_index))
            elif self.activation == 'leaky_relu':
                x = F.leaky_relu(conv(x, edge_index))
            elif self.activation == 'gelu':
                x = F.gelu(conv(x, edge_index))
        return x


class MultiLayerGIN(torch.nn.Module):
    def __init__(self, num_features, hidden_sizes, activation_fuc):
        super(MultiLayerGIN, self).__init__()
        self.activation = activation_fuc
        self.conv_layers = torch.nn.ModuleList()
        self.conv_layers.append(GINConv(nn.Sequential(nn.Linear(num_features, hidden_sizes[0]))))
        for i in range(1, len(hidden_sizes)):
            self.conv_layers.append(GINConv(nn.Sequential(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))))

    def forward(self, x, edge_index):
        for conv in self.conv_layers:
            if self.activation == 'relu':
                x = F.relu(conv(x, edge_index))
            elif self.activation == 'leaky_relu':
                x = F.leaky_relu(conv(x, edge_index))
            elif self.activation == 'gelu':
                x = F.gelu(conv(x, edge_index))
        return x


class MultiLayerGAT(nn.Module):
    def __init__(self, in_features, hidden_dims, activation_fuc, num_head=4, dropout=0.2):
        super(MultiLayerGAT, self).__init__()
        self.activation = activation_fuc
        self.gat_layers = ModuleList([
                                         GATConv(in_features, hidden_dims[0], heads=num_head, dropout=dropout)
                                     ] + [
                                         GATConv(hidden_dims[i - 1] * num_head, hidden_dims[i], heads=num_head,
                                                 dropout=dropout)
                                         for i in range(1, len(hidden_dims))
                                     ])

    def forward(self, x, edge_index):
        for gat in self.gat_layers:
            if self.activation == 'relu':
                x = F.relu(gat(x, edge_index))
            elif self.activation == 'leaky_relu':
                x = F.leaky_relu(gat(x, edge_index))
            elif self.activation == 'gelu':
                x = F.gelu(gat(x, edge_index))
        return x


class MultiLayerGTC(torch.nn.Module):
    def __init__(self, in_features, hidden_dims, num_head=1, dropout=0.0):
        super(MultiLayerGTC, self).__init__()
        self.gtc_layers = ModuleList([
                                         TransformerConv(in_features, hidden_dims[0], heads=num_head, dropout=dropout)
                                     ] + [
                                         TransformerConv(hidden_dims[i - 1] * num_head, hidden_dims[i], heads=num_head,
                                                         dropout=dropout)
                                         for i in range(1, len(hidden_dims))
                                     ])
        self.layerNormList = ModuleList([nn.LayerNorm(hidden_dims[i] * num_head) for i in range(len(hidden_dims))])

    def forward(self, x, edge_index):
        for i, gtc in enumerate(self.gtc_layers):
            x = self.layerNormList[i](gtc(x, edge_index))
        return x


class MLPReadout(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1, n_layers=1):
        super().__init__()
        fc_layers = [nn.Linear(input_dim // 2 ** n, input_dim // 2 ** (n + 1), bias=True) for n in range(n_layers)]
        fc_layers.append(nn.Linear(input_dim // 2 ** n_layers, output_dim, bias=True))
        self.fc_layers = nn.ModuleList(fc_layers)
        self.n_layers = n_layers
        self.dropout = dropout
        self.activation = nn.GELU()

    def forward(self, x):
        y = x
        for n in range(self.n_layers):
            y = F.dropout(y, self.dropout, training=self.training)
            y = self.fc_layers[n](y)
            y = self.activation(y)
        y = self.fc_layers[self.n_layers](y)
        return y


class MLP(torch.nn.Module):
    def __init__(self, mlp_input_dim, mlp_hidden_dim, mlp_output_dim, activation_fuc):
        super(MLP, self).__init__()
        self.activation = activation_fuc
        self.fc1 = torch.nn.Linear(mlp_input_dim, mlp_hidden_dim)
        self.fc2 = torch.nn.Linear(mlp_hidden_dim, mlp_output_dim)

    def forward(self, x):
        x = self.fc1(x)
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x)
        elif self.activation == 'gelu':
            x = F.gelu(x)
        x = self.fc2(x)
        return x


class SLADGNN(torch.nn.Module):
    """substructure-aware log anomaly detection gnn"""

    def __init__(self, gnn_head_num, gnn_dropout, activation_fuc, GNN_type, num_features, hidden_sizes,
                 feat_transform_dim, num_classes,
                 num_prototypes_per_class, mlp_input_dim,
                 mlp_output_dim=1):
        super(SLADGNN, self).__init__()
        self.dim_repre_struct = hidden_sizes[-1] * gnn_head_num
        if GNN_type == 'gcn':
            self.GNN = MultiLayerGCN(num_features, hidden_sizes, activation_fuc)
        elif GNN_type == 'gat':
            self.GNN = MultiLayerGAT(num_features, hidden_sizes, activation_fuc, num_head=gnn_head_num,
                                     dropout=gnn_dropout)
        elif GNN_type == 'gtc':
            self.GNN = MultiLayerGTC(num_features, hidden_sizes, num_head=gnn_head_num, dropout=gnn_dropout)
        elif GNN_type == 'gin':
            self.GNN = MultiLayerGIN(num_features, hidden_sizes, activation_fuc)
        else:
            raise ValueError("not supportive gnn.")
        self.prototype_layer = nn.Parameter(
            torch.randn(num_classes, num_prototypes_per_class, hidden_sizes[-1] * gnn_head_num))
        self.multihead_attn = nn.MultiheadAttention(self.dim_repre_struct, 1)
        self.feat_transformation = torch.nn.Linear(num_features,
                                                   feat_transform_dim)  # transform original node feature to hidden space
        self.GNN_4_soft_pruning = MultiLayerGTC(feat_transform_dim + hidden_sizes[-1] * gnn_head_num, hidden_sizes,
                                                num_head=gnn_head_num, dropout=gnn_dropout)
        self.mlp = MLPReadout(mlp_input_dim, mlp_output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data, mode, over_sample_scale_factor, sample_method, output):
        if mode == 'test':
            stat_resource_usage(output)
        x, edge_index, y = data.x, data.edge_index, data.y
        x = self.GNN(x, edge_index)
        if mode == 'test':
            stat_resource_usage(output)
        reshaped_prot_layer = self.prototype_layer.reshape(-1, self.prototype_layer.size(-1))
        node_repre_substruct = self.multihead_attn(x, reshaped_prot_layer, reshaped_prot_layer)[0]
        node_feat_transform = self.feat_transformation(data.x)  # transform original node feature to hidden space
        node_embed_with_repre_substruct = torch.cat((node_feat_transform, node_repre_substruct), dim=-1)
        x = self.GNN_4_soft_pruning(node_embed_with_repre_substruct, edge_index)
        """SMOTE oversampling"""
        if mode == "train":
            x, y = balance_features_labels(x, y, over_sample_scale_factor, sample_method)
        if mode == 'test':
            stat_resource_usage(output)
        x = self.mlp(x)
        x = self.sigmoid(x)
        return x.squeeze(), y.float()
