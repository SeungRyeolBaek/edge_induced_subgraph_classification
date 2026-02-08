# impl/ISNN_model.py

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor  # optional but recommended

from impl.IGNN_utils import projection_norm_inf


# -----------------------------
# Helpers
# -----------------------------
def _adj_matmul(adj, x):
    """
    Multiply adjacency (sparse) with dense matrix x.

    Supports:
      - torch_sparse.SparseTensor (recommended)
      - torch.sparse_coo_tensor
      - dense tensor fallback
    """
    if isinstance(adj, SparseTensor):
        return adj.matmul(x)
    if torch.is_tensor(adj) and adj.is_sparse:
        return torch.sparse.mm(adj, x)
    # dense fallback
    return adj @ x


# -----------------------------
# Attention module
# -----------------------------
class attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(attention, self).__init__()
        self.W = nn.Parameter(torch.rand(in_channels, out_channels, requires_grad=True))
        self.b = nn.Parameter(torch.rand(1, out_channels, requires_grad=True))
        self.p = nn.Parameter(torch.rand(out_channels, 1, requires_grad=True))
        self.initialize()

    def initialize(self):
        torch.nn.init.xavier_uniform_(self.W)
        torch.nn.init.xavier_uniform_(self.b)
        torch.nn.init.xavier_uniform_(self.p)

    def forward(self, x, y):
        # NOTE: keep original logic (tanh + shared projection)
        x = torch.tanh(x @ self.W + self.b) @ self.p
        y = torch.tanh(y @ self.W + self.b) @ self.p
        z = torch.cat([x, y], dim=1)
        return F.softmax(z, dim=1)


# -----------------------------
# Base models
# -----------------------------
class baseIGNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, dropout=0.5, normalize=True):
        super(baseIGNN, self).__init__()
        self.convs = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(out_channels, out_channels, bias=False, normalize=normalize))
            self.mlps.append(nn.Linear(in_channels, out_channels, bias=False))
        self.dropout = dropout
        self.initialize()

    def initialize(self):
        for conv in self.convs:
            conv.reset_parameters()
            # conv.lin exists in PyG GCNConv
            torch.nn.init.xavier_uniform_(conv.lin.weight)

        for mlp in self.mlps:
            mlp.reset_parameters()
            torch.nn.init.xavier_uniform_(mlp.weight)

    def forward(self, feature, edge_index, embedding, pat2sub=None):
        # NOTE: keep original logic
        x = embedding
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if pat2sub is not None:
                x = x + self.mlps[i](pat2sub.T @ feature)
            else:
                x = x + self.mlps[i](feature)
            x = F.relu(x)
        return x


class baseGNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, dropout=0.5, normalize=True):
        super(baseGNN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, out_channels, normalize=normalize))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(out_channels, out_channels, normalize=normalize))
        self.dropout = dropout
        self.initialize()

    def initialize(self):
        for conv in self.convs:
            conv.reset_parameters()
            torch.nn.init.xavier_uniform_(conv.lin.weight)

    def forward(self, feature, sparse_adj):
        """
        Bugfix only: original code never applied the last conv layer,
        and for num_layers=1 it returned inputs unchanged.
        This change applies all layers, keeping the usual convention
        (no dropout/relu after the last layer).
        """
        x = feature
        for i, conv in enumerate(self.convs):
            x = conv(x, sparse_adj)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# -----------------------------
# Models
# -----------------------------
class SubIGNN_v2(nn.Module):
    def __init__(
        self,
        pretrained_embeddings,
        out_channels,
        num_classes,
        num_nodes,
        projection_matrix,
        node_mask,
        loss_fn=None,
        gamma=0.01,
        kappa=0.95,
    ):
        super(SubIGNN_v2, self).__init__()
        self.num_classes = num_classes
        self.node_mask = node_mask
        self.projection_matrix = projection_matrix
        self.gamma = gamma
        self.kappa = kappa

        self.loss_fn = nn.BCEWithLogitsLoss() if loss_fn is None else loss_fn

        self.classifier = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, num_classes),
        )

        self.F = nn.Parameter(torch.rand(out_channels, out_channels, requires_grad=True))

        # IMPORTANT: keep pretrained embeddings as non-trainable buffer so device moves are safe.
        # If you actually want it trainable, change to nn.Parameter.
        if torch.is_tensor(pretrained_embeddings):
            self.register_buffer("pretrained_embeddings", pretrained_embeddings)
        else:
            # fallback (shouldn't happen)
            self.pretrained_embeddings = pretrained_embeddings

        self.embeddings = nn.Parameter(0.01 * torch.rand(num_nodes, out_channels, requires_grad=True))

        # proxy_embeddings is not a parameter; keep as buffer for device safety
        self.register_buffer(
            "proxy_embeddings",
            0.1 * torch.randn((num_nodes, out_channels), device=node_mask.device),
        )

        self.attention = attention(out_channels, out_channels)

    def reset_parameters(self):
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

        self.classifier.apply(init_weights)
        self.attention.initialize()
        self.F.data = torch.rand_like(self.F.data)
        self.embeddings.data = 0.01 * torch.rand_like(self.embeddings.data)
        self.proxy_embeddings.data = 0.1 * torch.randn_like(self.proxy_embeddings.data)

    def project(self, epsilon=1e-5):
        """
        Bugfix only: original could return None when norm<=1.
        Keep same intent: normalize if too large, otherwise return as-is.
        """
        W = self.F.T @ self.F
        n = torch.norm(W)
        if n > 1:
            W = W / (n + epsilon)
        return W

    def forward(self, features, sparse_adj, embeddings=None):
        kappa = self.kappa
        weight = self.project()
        if embeddings is None:
            embeddings = self.embeddings
        outputs = kappa * ((_adj_matmul(sparse_adj, embeddings)) @ weight) + self.pretrained_embeddings
        return outputs

    @torch.no_grad()
    def fixed_point_iteration(self, features, sparse_adj, inner_iters=10):
        embeddings = self.proxy_embeddings
        for _ in range(inner_iters):
            new_embeddings = self.forward(features, sparse_adj, embeddings)
            residual_error = torch.norm(new_embeddings - embeddings)
            embeddings = new_embeddings
            if residual_error < 1e-5:
                break
        self.proxy_embeddings = embeddings
        return residual_error.item()

    def classify(self, train_mask, embeddings=None):
        node_mask = self.node_mask
        projection_matrix = self.projection_matrix
        num_subgraph_nodes = (~node_mask).sum().item()

        if embeddings is None:
            embeddings = self.embeddings
        if train_mask is None:
            train_mask = torch.ones(num_subgraph_nodes, dtype=torch.bool, device=embeddings.device)

        node_embeddings = embeddings[node_mask]
        subgraph_embeddings_1 = projection_matrix(node_embeddings)
        subgraph_embeddings_2 = embeddings[~node_mask]

        att = self.attention(subgraph_embeddings_1, subgraph_embeddings_2)
        final_embedding = att[:, 0].reshape(-1, 1) * subgraph_embeddings_1 + att[:, 1].reshape(-1, 1) * subgraph_embeddings_2
        return self.classifier(final_embedding[train_mask])

    def loss(self, features, sparse_adj, targets, train_mask):
        proxy_embeddings = self.proxy_embeddings
        gamma = self.gamma

        hybrid_embeddings = self.forward(features, sparse_adj)
        clssifier_loss = self.loss_fn(self.classify(train_mask), targets)

        residual = self.embeddings - hybrid_embeddings
        proxy_residual = proxy_embeddings - self.forward(features, sparse_adj, embeddings=proxy_embeddings)

        residual_loss = 0.5 * (residual.pow(2).sum())
        proxy_residual_loss = 0.5 * (proxy_residual.pow(2).sum())

        graph_residual_loss = residual_loss
        graph_residual_approximate_loss = proxy_residual_loss

        loss = clssifier_loss + gamma * (graph_residual_loss - graph_residual_approximate_loss)
        return loss

    @torch.no_grad()
    def predict(self, features, sparse_adj, train_mask):
        return self.classify(train_mask)

    @torch.no_grad()
    def get_subgraph_embeddings(self):
        node_mask = self.node_mask
        projection_matrix = self.projection_matrix
        node_embeddings = self.embeddings[node_mask]
        subgraph_embeddings = projection_matrix(node_embeddings)
        return subgraph_embeddings


class SubIGNN_new(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_classes,
        num_nodes,
        projection_matrix,
        node_mask,
        num_layers=1,
        dropout=0.5,
        loss_fn=None,
        gamma=0.01,
        kappa=0.95,
    ):
        super(SubIGNN_new, self).__init__()
        self.dropout = dropout
        self.graph_convs = baseGNN(in_channels, out_channels, num_layers, dropout, normalize=False)
        self.num_classes = num_classes
        self.node_mask = node_mask
        self.projection_matrix = projection_matrix
        self.gamma = gamma
        self.kappa = kappa

        self.loss_fn = nn.BCEWithLogitsLoss() if loss_fn is None else loss_fn

        self.classifier = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, num_classes),
        )

        self.F = nn.Parameter(torch.rand(out_channels, out_channels, requires_grad=True))
        self.embeddings = nn.Parameter(0.01 * torch.rand(num_nodes, out_channels, requires_grad=True))

        self.register_buffer(
            "proxy_embeddings",
            0.1 * torch.randn((num_nodes, out_channels), device=node_mask.device),
        )
        self.attention = attention(out_channels, out_channels)

    def reset_parameters(self):
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

        self.classifier.apply(init_weights)
        self.graph_convs.initialize()
        self.attention.initialize()
        self.F.data = torch.rand_like(self.F.data)
        self.embeddings.data = 0.01 * torch.rand_like(self.embeddings.data)
        self.proxy_embeddings.data = 0.1 * torch.randn_like(self.proxy_embeddings.data)

    def project(self, epsilon=1e-5):
        # keep original behavior
        W = self.F.T @ self.F
        n = torch.norm(W)
        if n > 1:
            return W / (n + epsilon)
        return W

    def forward(self, features, sparse_adj, embeddings=None):
        kappa = self.kappa
        weight = self.project()
        if embeddings is None:
            embeddings = self.embeddings
        outputs = self.graph_convs(features, sparse_adj)
        outputs = kappa * ((_adj_matmul(sparse_adj, embeddings)) @ weight) + outputs
        return outputs

    @torch.no_grad()
    def fixed_point_iteration(self, features, sparse_adj, inner_iters=10):
        embeddings = self.proxy_embeddings
        for _ in range(inner_iters):
            new_embeddings = self.forward(features, sparse_adj, embeddings)
            residual_error = torch.norm(new_embeddings - embeddings)
            embeddings = new_embeddings
            if residual_error < 1e-5:
                break
        self.proxy_embeddings = embeddings
        return residual_error.item()

    def classify(self, train_mask, embeddings=None):
        node_mask = self.node_mask
        projection_matrix = self.projection_matrix
        num_subgraph_nodes = (~node_mask).sum().item()

        if embeddings is None:
            embeddings = self.embeddings
        if train_mask is None:
            train_mask = torch.ones(num_subgraph_nodes, dtype=torch.bool, device=embeddings.device)

        node_embeddings = embeddings[node_mask]
        subgraph_embeddings_1 = projection_matrix(node_embeddings)
        subgraph_embeddings_2 = embeddings[~node_mask]
        att = self.attention(subgraph_embeddings_1, subgraph_embeddings_2)
        final_embedding = att[:, 0].reshape(-1, 1) * subgraph_embeddings_1 + att[:, 1].reshape(-1, 1) * subgraph_embeddings_2
        return self.classifier(final_embedding[train_mask])

    def loss(self, features, sparse_adj, targets, train_mask):
        proxy_embeddings = self.proxy_embeddings
        gamma = self.gamma

        hybrid_embeddings = self.forward(features, sparse_adj)
        clssifier_loss = self.loss_fn(self.classify(train_mask), targets)

        residual = self.embeddings - hybrid_embeddings
        proxy_residual = proxy_embeddings - self.forward(features, sparse_adj, embeddings=proxy_embeddings)

        residual_loss = 0.5 * (residual.pow(2).sum())
        proxy_residual_loss = 0.5 * (proxy_residual.pow(2).sum())

        graph_residual_loss = residual_loss
        graph_residual_approximate_loss = proxy_residual_loss

        loss = clssifier_loss + gamma * (graph_residual_loss - graph_residual_approximate_loss)
        return loss

    @torch.no_grad()
    def predict(self, features, sparse_adj, train_mask):
        return self.classify(train_mask)

    @torch.no_grad()
    def get_subgraph_embeddings(self):
        node_mask = self.node_mask
        projection_matrix = self.projection_matrix
        node_embeddings = self.embeddings[node_mask]
        subgraph_embeddings = projection_matrix(node_embeddings)
        return subgraph_embeddings


class SoftEIGNN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_classes,
        num_nodes,
        projection_matrix,
        device,
        num_layers=1,
        dropout=0.5,
        loss_fn=None,
        gamma=0.01,
    ):
        super(SoftEIGNN, self).__init__()
        self.dropout = dropout
        self.graph_convs = baseGNN(in_channels, out_channels, num_layers, dropout, normalize=True)
        self.num_classes = num_classes
        self.projection_matrix = projection_matrix
        self.gamma = gamma

        self.loss_fn = nn.BCEWithLogitsLoss() if loss_fn is None else loss_fn

        self.classifier = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, num_classes),
        )

        self.F = nn.Parameter(torch.rand(out_channels, out_channels, requires_grad=True))
        self.embeddings = nn.Parameter(0.01 * torch.rand(num_nodes, out_channels, requires_grad=True))
        self.register_buffer("proxy_embeddings", 0.1 * torch.randn((num_nodes, out_channels), device=device))

    def project(self, epsilon=1e-5):
        # keep original behavior
        W = self.F.T @ self.F
        return W / (torch.norm(W) + epsilon)

    def forward(self, features, sparse_adj, embeddings=None, kappa=0.95):
        weight = self.project()
        if embeddings is None:
            embeddings = self.embeddings
        outputs = self.graph_convs(features, sparse_adj)
        outputs = kappa * ((_adj_matmul(sparse_adj, embeddings)) @ weight) + outputs
        return outputs

    @torch.no_grad()
    def fixed_point_iteration(self, features, sparse_adj, inner_iters=10):
        embeddings = self.proxy_embeddings
        for _ in range(inner_iters):
            new_embeddings = self.forward(features, sparse_adj, embeddings)
            residual_error = torch.norm(new_embeddings - embeddings)
            embeddings = new_embeddings
            if residual_error < 1e-5:
                break
        self.proxy_embeddings = embeddings
        return residual_error.item()

    def classify(self, train_mask, node_embeddings=None):
        if node_embeddings is None:
            node_embeddings = self.embeddings
        subgraph_embeddings = self.projection_matrix(node_embeddings)
        return self.classifier(subgraph_embeddings[train_mask])

    def loss(self, features, sparse_adj, targets, train_mask):
        proxy_embeddings = self.proxy_embeddings
        gamma = self.gamma

        clssifier_loss = self.loss_fn(self.classify(train_mask), targets)

        residual = self.embeddings - self.forward(features, sparse_adj)
        proxy_residual = proxy_embeddings - self.forward(features, sparse_adj, embeddings=proxy_embeddings)

        residual_loss = 0.5 * (residual.pow(2).sum())
        proxy_residual_loss = 0.5 * (proxy_residual.pow(2).sum())

        return clssifier_loss + gamma * (residual_loss - proxy_residual_loss)

    def predict(self, features, sparse_adj, train_mask):
        return self.classify(train_mask)


class SoftIGNN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_classes,
        num_nodes,
        projection_matrix,
        device,
        num_layers=1,
        dropout=0.5,
        loss_fn=None,
        gamma=0.01,
    ):
        super(SoftIGNN, self).__init__()
        self.dropout = dropout
        self.graph_convs = baseIGNN(in_channels, out_channels, num_layers, dropout, normalize=True)
        self.num_classes = num_classes
        self.projection_matrix = projection_matrix
        self.gamma = gamma

        self.loss_fn = nn.BCEWithLogitsLoss() if loss_fn is None else loss_fn

        self.classifier = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, num_classes),
        )

        self.F = nn.Parameter(torch.rand(out_channels, out_channels, requires_grad=True))
        self.embeddings = nn.Parameter(0.01 * torch.rand(num_nodes, out_channels, requires_grad=True))
        self.register_buffer("proxy_embeddings", 0.1 * torch.randn((num_nodes, out_channels), device=device))

    def project(self, kappa=0.95):
        # keep original behavior (in-place projection on conv weights)
        for conv in self.graph_convs.convs:
            conv.lin.weight.data = projection_norm_inf(conv.lin.weight.data, kappa)

    def forward(self, features, sparse_adj, embeddings=None, kappa=0.95):
        self.project(kappa=kappa)
        if embeddings is None:
            embeddings = self.embeddings
        outputs = self.graph_convs(features, sparse_adj, embeddings)
        return outputs

    @torch.no_grad()
    def fixed_point_iteration(self, features, sparse_adj, inner_iters=10):
        embeddings = self.proxy_embeddings
        for _ in range(inner_iters):
            new_embeddings = self.forward(features, sparse_adj, embeddings)
            residual_error = torch.norm(new_embeddings - embeddings)
            embeddings = new_embeddings
            if residual_error < 1e-5:
                break
        self.proxy_embeddings = embeddings
        return residual_error.item()

    def classify(self, train_mask, node_embeddings=None):
        if node_embeddings is None:
            node_embeddings = self.embeddings
        subgraph_embeddings = self.projection_matrix(node_embeddings)
        return self.classifier(subgraph_embeddings[train_mask])

    def loss(self, features, sparse_adj, targets, train_mask):
        proxy_embeddings = self.proxy_embeddings
        gamma = self.gamma

        clssifier_loss = self.loss_fn(self.classify(train_mask), targets)

        residual = self.embeddings - self.forward(features, sparse_adj)
        proxy_residual = proxy_embeddings - self.forward(features, sparse_adj, embeddings=proxy_embeddings)

        residual_loss = 0.5 * (residual.pow(2).sum())
        proxy_residual_loss = 0.5 * (proxy_residual.pow(2).sum())

        return clssifier_loss + gamma * (residual_loss - proxy_residual_loss)

    def predict(self, features, sparse_adj, train_mask):
        return self.classify(train_mask)


class VanillaGCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, projection_matrix, num_layers=1, dropout=0.5, loss_fn=None):
        super(VanillaGCN, self).__init__()
        self.dropout = dropout
        self.graph_convs = baseGNN(in_channels, out_channels, num_layers, dropout, normalize=True)
        self.num_classes = num_classes
        self.projection_matrix = projection_matrix

        self.loss_fn = nn.BCEWithLogitsLoss() if loss_fn is None else loss_fn

        self.classifier = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, num_classes),
        )

    def forward(self, features, sparse_adj):
        node_embeddings = self.graph_convs(features, sparse_adj)
        subgraph_embeddings = self.projection_matrix(node_embeddings)
        return subgraph_embeddings

    def classify(self, features, sparse_adj, train_mask):
        subgraph_embeddings = self.forward(features, sparse_adj)
        return self.classifier(subgraph_embeddings[train_mask])

    def loss(self, features, sparse_adj, targets, train_mask):
        clssifier_loss = self.loss_fn(self.classify(features, sparse_adj, train_mask), targets)
        return clssifier_loss

    @torch.no_grad()
    def predict(self, features, sparse_adj, train_mask):
        return self.classify(features=features, sparse_adj=sparse_adj, train_mask=train_mask)


class baseline(nn.Module):
    def __init__(self, out_channels, num_classes, num_nodes, projection_matrix, loss_fn=None):
        super(baseline, self).__init__()
        self.projection_matrix = projection_matrix
        self.embeddings = nn.Parameter(0.01 * torch.rand(num_nodes, out_channels, requires_grad=True))
        self.loss_fn = loss_fn

        self.classifier = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, num_classes),
        )

    def forward(self):
        embeddings = self.embeddings
        return self.projection_matrix(embeddings)

    def classify(self, train_mask):
        subgraph_embeddings = self.forward()
        return self.classifier(subgraph_embeddings[train_mask])

    def loss(self, features, sparse_adj, targets, train_mask):
        clssifier_loss = self.loss_fn(self.classify(train_mask), targets)
        return clssifier_loss

    @torch.no_grad()
    def predict(self, features, sparse_adj, train_mask):
        return self.classify(train_mask)
