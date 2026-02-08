# impl/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.norm import GraphNorm, GraphSizeNorm
from torch_geometric.nn.glob import global_mean_pool, global_add_pool, global_max_pool
from .utils import pad2batch

# ----------------------------
# small helpers (원본 그대로)
# ----------------------------
class Seq(nn.Module):
    def __init__(self, modlist):
        super().__init__()
        self.modlist = nn.ModuleList(modlist)

    def forward(self, *args, **kwargs):
        out = self.modlist[0](*args, **kwargs)
        for i in range(1, len(self.modlist)):
            out = self.modlist[i](out)
        return out


class MLP(nn.Module):
    def __init__(self,
                 input_channels: int,
                 hidden_channels: int,
                 output_channels: int,
                 num_layers: int,
                 dropout=0,
                 tail_activation=False,
                 activation=nn.ReLU(inplace=True),
                 gn=False):
        super().__init__()
        modlist = []
        if num_layers == 1:
            modlist.append(nn.Linear(input_channels, output_channels))
            if tail_activation:
                if gn:
                    modlist.append(GraphNorm(output_channels))
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout, inplace=True))
                modlist.append(activation)
        else:
            modlist.append(nn.Linear(input_channels, hidden_channels))
            for _ in range(num_layers - 2):
                if gn:
                    modlist.append(GraphNorm(hidden_channels))
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout, inplace=True))
                modlist.append(activation)
                modlist.append(nn.Linear(hidden_channels, hidden_channels))
            if gn:
                modlist.append(GraphNorm(hidden_channels))
            if dropout > 0:
                modlist.append(nn.Dropout(p=dropout, inplace=True))
            modlist.append(activation)
            modlist.append(nn.Linear(hidden_channels, output_channels))
            if tail_activation:
                if gn:
                    modlist.append(GraphNorm(output_channels))
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout, inplace=True))
                modlist.append(activation)

        self.seq = Seq(modlist)

    def forward(self, x):
        return self.seq(x)


def buildAdj(edge_index, edge_weight, n_node: int, aggr: str):
    """
    normalized adjacency (sparse COO)
    NOTE: returned sparse matrix is NOT coalesced by default.
    """
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, size=(n_node, n_node))
    deg = torch.sparse.sum(adj, dim=(1,)).to_dense().flatten()
    deg[deg < 0.5] += 1.0
    if aggr == "mean":
        deg = 1.0 / deg
        return torch.sparse_coo_tensor(edge_index, deg[edge_index[0]] * edge_weight, size=(n_node, n_node))
    elif aggr == "sum":
        return torch.sparse_coo_tensor(edge_index, edge_weight, size=(n_node, n_node))
    elif aggr == "gcn":
        deg = torch.pow(deg, -0.5)
        return torch.sparse_coo_tensor(edge_index, deg[edge_index[0]] * edge_weight * deg[edge_index[1]], size=(n_node, n_node))
    else:
        raise NotImplementedError


# ----------------------------
# GLASSConv (원본)
# ----------------------------
class GLASSConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation=nn.ReLU(inplace=True),
                 aggr="mean",
                 z_ratio=0.8,
                 dropout=0.2):
        super().__init__()
        self.trans_fns = nn.ModuleList([nn.Linear(in_channels, out_channels),
                                        nn.Linear(in_channels, out_channels)])
        self.comb_fns = nn.ModuleList([nn.Linear(in_channels + out_channels, out_channels),
                                       nn.Linear(in_channels + out_channels, out_channels)])
        self.adj = torch.sparse_coo_tensor(size=(0, 0))
        self.activation = activation
        self.aggr = aggr
        self.gn = GraphNorm(out_channels)
        self.z_ratio = z_ratio
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.trans_fns: m.reset_parameters()
        for m in self.comb_fns:  m.reset_parameters()
        self.gn.reset_parameters()
        self.adj = torch.sparse_coo_tensor(size=(0, 0))

    def forward(self, x_, edge_index, edge_weight, mask):
        if self.adj.shape[0] == 0:
            n_node = x_.shape[0]
            self.adj = buildAdj(edge_index, edge_weight, n_node, self.aggr)

        x1 = self.activation(self.trans_fns[1](x_))
        x0 = self.activation(self.trans_fns[0](x_))
        x = torch.where(mask, self.z_ratio * x1 + (1 - self.z_ratio) * x0,
                        self.z_ratio * x0 + (1 - self.z_ratio) * x1)

        x = self.adj @ x
        x = self.gn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = torch.cat((x, x_), dim=-1)
        x1 = self.comb_fns[1](x)
        x0 = self.comb_fns[0](x)
        x = torch.where(mask, self.z_ratio * x1 + (1 - self.z_ratio) * x0,
                        self.z_ratio * x0 + (1 - self.z_ratio) * x1)
        return x


# ----------------------------
# EmbZGConv (원본 구조 유지)
# ----------------------------
class EmbZGConv(nn.Module):
    def __init__(self,
                 hidden_channels,
                 output_channels,
                 num_layers,
                 max_deg,
                 dropout=0,
                 activation=nn.ReLU(),
                 conv=GLASSConv,
                 gn=True,
                 jk=False,
                 **kwargs):
        super().__init__()
        self.input_emb = nn.Embedding(max_deg + 1, hidden_channels, scale_grad_by_freq=False)
        self.emb_gn = GraphNorm(hidden_channels)

        self.convs = nn.ModuleList()
        self.jk = jk
        for _ in range(num_layers - 1):
            self.convs.append(conv(in_channels=hidden_channels,
                                   out_channels=hidden_channels,
                                   activation=activation,
                                   **kwargs))
        self.convs.append(conv(in_channels=hidden_channels,
                               out_channels=output_channels,
                               activation=activation,
                               **kwargs))

        self.activation = activation
        self.dropout = dropout

        if gn:
            self.gns = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.gns.append(GraphNorm(hidden_channels))
            if self.jk:
                self.gns.append(GraphNorm(output_channels + (num_layers - 1) * hidden_channels))
            else:
                self.gns.append(GraphNorm(output_channels))
        else:
            self.gns = None

        self.reset_parameters()

    def reset_parameters(self):
        self.input_emb.reset_parameters()
        self.emb_gn.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        if self.gns is not None:
            for gn in self.gns:
                gn.reset_parameters()

    def forward(self, x, edge_index, edge_weight, z=None):
        # z: node membership (n_node,) or (n_node,1) 0/1
        if z is None:
            mask = torch.zeros((x.shape[0], 1), device=x.device, dtype=torch.bool)
        else:
            # 안전: dtype이 float/int여도 ok
            mask = (z > 0.5).reshape(-1, 1)

        x = self.input_emb(x).reshape(x.shape[0], -1)
        x = self.emb_gn(x)

        xs = []
        x = F.dropout(x, p=self.dropout, training=self.training)
        for layer, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight, mask)
            xs.append(x)
            if self.gns is not None:
                x = self.gns[layer](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index, edge_weight, mask)
        xs.append(x)

        if self.jk:
            x = torch.cat(xs, dim=-1)
            if self.gns is not None:
                x = self.gns[-1](x)
            return x
        else:
            x = xs[-1]
            if self.gns is not None:
                x = self.gns[-1](x)
            return x


# ----------------------------
# Pool modules (원본)
# ----------------------------
class PoolModule(nn.Module):
    def __init__(self, pool_fn, trans_fn=None):
        super().__init__()
        self.pool_fn = pool_fn
        self.trans_fn = trans_fn

    def forward(self, x, batch):
        if self.trans_fn is not None:
            x = self.trans_fn(x)
        return self.pool_fn(x, batch)


class AddPool(PoolModule):
    def __init__(self, trans_fn=None):
        super().__init__(global_add_pool, trans_fn)


class MaxPool(PoolModule):
    def __init__(self, trans_fn=None):
        super().__init__(global_max_pool, trans_fn)


class MeanPool(PoolModule):
    def __init__(self, trans_fn=None):
        super().__init__(global_mean_pool, trans_fn)


class SizePool(AddPool):
    def __init__(self, trans_fn=None):
        super().__init__(trans_fn)

    def forward(self, x, batch):
        if x is not None and self.trans_fn is not None:
            x = self.trans_fn(x)
        x = GraphSizeNorm()(x, batch)
        return self.pool_fn(x, batch)


# ----------------------------
# GLASS wrapper (원본)
# ----------------------------
class GLASS(nn.Module):
    def __init__(self, conv: EmbZGConv, preds: nn.ModuleList, pools: nn.ModuleList):
        super().__init__()
        self.conv = conv
        self.preds = preds
        self.pools = pools

    def NodeEmb(self, x, edge_index, edge_weight, z=None):
        embs = []
        for k in range(x.shape[1]):
            emb = self.conv(x[:, k, :].reshape(x.shape[0], x.shape[-1]), edge_index, edge_weight, z)
            embs.append(emb.reshape(emb.shape[0], 1, emb.shape[-1]))
        emb = torch.cat(embs, dim=1)
        emb = torch.mean(emb, dim=1)
        return emb

    def Pool(self, emb, subG_node, pool):
        batch, pos = pad2batch(subG_node)
        emb = emb[pos]
        emb = pool(emb, batch)
        return emb

    def forward(self, x, edge_index, edge_weight, subG_node, z=None, id=0):
        emb = self.NodeEmb(x, edge_index, edge_weight, z)
        emb = self.Pool(emb, subG_node, self.pools[id])
        return self.preds[id](emb)


# ============================================================
# GLASS-E (edge-aware) implementation (REPLACE THIS BLOCK)
# ============================================================

# 아래 클래스/함수들은 너 프로젝트 models.py 안에 이미 있다고 가정:
# - GraphNorm, buildAdj, pad2batch
# - AddPool, MaxPool, MeanPool, SizePool
# - MLP

def _edge_key(u, v, n_node: int):
    mn = torch.minimum(u, v)
    mx = torch.maximum(u, v)
    return mn * n_node + mx


def _pad_edge_list_to_2d(subG_edge):
    """
    subG_edge:
      - (B, maxE, 2) or (maxE, 2). padding is -1
    return: (K, 2) valid edges only
    """
    if subG_edge is None:
        return None
    if not isinstance(subG_edge, torch.Tensor):
        return None
    if subG_edge.numel() == 0:
        return None
    e = subG_edge.reshape(-1, 2)
    valid = (e[:, 0] >= 0) & (e[:, 1] >= 0)
    e = e[valid]
    if e.numel() == 0:
        return None
    return e


def _node_mask_from_subG_edge(n_node: int, subG_edge, device):
    """
    build node mask from edge endpoints union (n_node,1) bool
    """
    mask = torch.zeros((n_node,), dtype=torch.bool, device=device)
    e = _pad_edge_list_to_2d(subG_edge)
    if e is None:
        return mask.reshape(-1, 1)
    u = e[:, 0].to(device)
    v = e[:, 1].to(device)
    mask[u] = True
    mask[v] = True
    return mask.reshape(-1, 1)


def _edge_membership_mask_from_subG_edge_cached(base_key, subG_edge, n_node: int):
    """
    base_key: (E_adj,) cached undirected key of base adjacency entries
    subG_edge: (B,maxE,2) or (maxE,2), padded -1
    return: (E_adj,) bool
    """
    e = _pad_edge_list_to_2d(subG_edge)
    if e is None:
        return torch.zeros((base_key.shape[0],), dtype=torch.bool, device=base_key.device)

    u = e[:, 0].to(base_key.device)
    v = e[:, 1].to(base_key.device)
    q = _edge_key(u, v, n_node)
    if q.numel() == 0:
        return torch.zeros((base_key.shape[0],), dtype=torch.bool, device=base_key.device)

    q = torch.unique(q)
    return torch.isin(base_key, q)


class EdgeGLASSConv(nn.Module):
    """
    KEEP LOGIC EXACTLY AS YOU WROTE (NO CHANGE)

    z supports:
      1) None -> node_mask False, edge_mask False
      2) Tensor (n_node,) or (n_node,1) -> node_mask, edge False
      3) Tensor (...,2) -> subG_edge, node_mask from edge endpoints
      4) tuple (subG_edge, node_z) -> edge from subG_edge, node_mask from node_z
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation=nn.ReLU(inplace=True),
                 aggr="mean",
                 z_ratio=0.8,
                 dropout=0.2):
        super().__init__()
        self.trans_fns = nn.ModuleList([
            nn.Linear(in_channels, out_channels),
            nn.Linear(in_channels, out_channels)
        ])
        self.comb_fns = nn.ModuleList([
            nn.Linear(in_channels + out_channels, out_channels),
            nn.Linear(in_channels + out_channels, out_channels)
        ])
        self.adj = torch.sparse_coo_tensor(size=(0, 0))
        self.adj_index = None   # (2, E_adj)
        self.adj_value = None   # (E_adj,)
        self.base_key = None    # (E_adj,)
        self.activation = activation
        self.aggr = aggr
        self.gn = GraphNorm(out_channels)
        self.z_ratio = z_ratio
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        for _ in self.trans_fns:
            _.reset_parameters()
        for _ in self.comb_fns:
            _.reset_parameters()
        self.gn.reset_parameters()
        self.adj = torch.sparse_coo_tensor(size=(0, 0))
        self.adj_index = None
        self.adj_value = None
        self.base_key = None

    def _prepare_adj(self, x_, edge_index, edge_weight):
        if self.adj.shape[0] == 0:
            n_node = x_.shape[0]
            self.adj = buildAdj(edge_index, edge_weight, n_node, self.aggr).coalesce()
            self.adj_index = self.adj.indices()
            self.adj_value = self.adj.values()
            row = self.adj_index[0]
            col = self.adj_index[1]
            self.base_key = _edge_key(row, col, n_node)

    def forward(self, x_, edge_index, edge_weight, z=None):
        self._prepare_adj(x_, edge_index, edge_weight)
        n_node = x_.shape[0]
        device = x_.device

        subG_edge = None
        node_z = None

        if isinstance(z, tuple) and len(z) == 2:
            subG_edge, node_z = z[0], z[1]
        elif isinstance(z, torch.Tensor):
            if z.dim() >= 2 and z.shape[-1] == 2:
                subG_edge = z
            else:
                node_z = z

        if node_z is None:
            if subG_edge is None:
                node_mask = (torch.zeros((n_node,), device=device) > 0.5).reshape(-1, 1)
            else:
                node_mask = _node_mask_from_subG_edge(n_node, subG_edge, device)
        else:
            node_mask = (node_z > 0.5).reshape(-1, 1)

        if subG_edge is None:
            edge_mask = torch.zeros((self.adj_index.shape[1],), dtype=torch.bool, device=device)
        else:
            edge_mask = _edge_membership_mask_from_subG_edge_cached(self.base_key, subG_edge, n_node)

        x1 = self.activation(self.trans_fns[1](x_))
        x0 = self.activation(self.trans_fns[0](x_))

        x_in   = self.z_ratio * x1 + (1 - self.z_ratio) * x0
        x_out  = self.z_ratio * x0 + (1 - self.z_ratio) * x1
        x_self = torch.where(node_mask, x_in, x_out)

        row = self.adj_index[0]
        col = self.adj_index[1]

        msg1 = x1[col]
        msg0 = x0[col]

        msg_in  = self.z_ratio * msg1 + (1 - self.z_ratio) * msg0
        msg_out = self.z_ratio * msg0 + (1 - self.z_ratio) * msg1
        msg = torch.where(edge_mask.reshape(-1, 1), msg_in, msg_out)

        msg = msg * self.adj_value.reshape(-1, 1)
        out = torch.zeros((n_node, msg.shape[1]), device=device, dtype=msg.dtype)
        out.index_add_(0, row, msg)

        out = self.gn(out)
        out = F.dropout(out, p=self.dropout, training=self.training)

        out = torch.cat((out, x_self), dim=-1)
        y1 = self.comb_fns[1](out)
        y0 = self.comb_fns[0](out)

        out_in  = self.z_ratio * y1 + (1 - self.z_ratio) * y0
        out_out = self.z_ratio * y0 + (1 - self.z_ratio) * y1
        out = torch.where(node_mask, out_in, out_out)
        return out


class EmbZGConvEdge(nn.Module):
    """
    same shell as EmbZGConv, but passes z through to EdgeGLASSConv
    """
    def __init__(self,
                 hidden_channels,
                 output_channels,
                 num_layers,
                 max_deg,
                 dropout=0,
                 activation=nn.ReLU(),
                 conv=EdgeGLASSConv,
                 gn=True,
                 jk=False,
                 **kwargs):
        super().__init__()
        self.input_emb = nn.Embedding(max_deg + 1, hidden_channels, scale_grad_by_freq=False)
        self.emb_gn = GraphNorm(hidden_channels)

        self.convs = nn.ModuleList()
        self.jk = jk

        for _ in range(num_layers - 1):
            self.convs.append(conv(in_channels=hidden_channels,
                                   out_channels=hidden_channels,
                                   activation=activation,
                                   **kwargs))
        self.convs.append(conv(in_channels=hidden_channels,
                               out_channels=output_channels,
                               activation=activation,
                               **kwargs))

        self.activation = activation
        self.dropout = dropout

        if gn:
            self.gns = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.gns.append(GraphNorm(hidden_channels))
            if self.jk:
                self.gns.append(GraphNorm(output_channels + (num_layers - 1) * hidden_channels))
            else:
                self.gns.append(GraphNorm(output_channels))
        else:
            self.gns = None

        self.reset_parameters()

    def reset_parameters(self):
        self.input_emb.reset_parameters()
        self.emb_gn.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        if self.gns is not None:
            for gn in self.gns:
                gn.reset_parameters()

    def forward(self, x, edge_index, edge_weight, z=None):
        x = self.input_emb(x).reshape(x.shape[0], -1)
        x = self.emb_gn(x)

        xs = []
        x = F.dropout(x, p=self.dropout, training=self.training)

        for layer, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight, z)
            xs.append(x)
            if self.gns is not None:
                x = self.gns[layer](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index, edge_weight, z)
        xs.append(x)

        if self.jk:
            x = torch.cat(xs, dim=-1)
            if self.gns is not None:
                x = self.gns[-1](x)
            return x
        else:
            x = xs[-1]
            if self.gns is not None:
                x = self.gns[-1](x)
            return x


class GLASSEdge(nn.Module):
    """
    GLASS wrapper compatible with your training loop.
    z can be subG_edge or (subG_edge,node_z)
    """
    def __init__(self, conv: EmbZGConvEdge, preds: nn.ModuleList, pools: nn.ModuleList):
        super().__init__()
        self.conv = conv
        self.preds = preds
        self.pools = pools

    def NodeEmb(self, x, edge_index, edge_weight, z=None):
        embs = []
        for k in range(x.shape[1]):
            emb = self.conv(x[:, k, :].reshape(x.shape[0], x.shape[-1]),
                            edge_index, edge_weight, z)
            embs.append(emb.reshape(emb.shape[0], 1, emb.shape[-1]))
        emb = torch.cat(embs, dim=1)
        emb = torch.mean(emb, dim=1)
        return emb

    def Pool(self, emb, subG_node, pool):
        batch, pos = pad2batch(subG_node)
        emb = emb[pos]
        emb = pool(emb, batch)
        return emb

    def forward(self, x, edge_index, edge_weight, subG_node, z=None, id=0):
        emb = self.NodeEmb(x, edge_index, edge_weight, z)
        emb = self.Pool(emb, subG_node, self.pools[id])
        return self.preds[id](emb)


def _make_pool(pool: str):
    pool = pool.lower()
    if pool == "sum":
        return AddPool()
    if pool == "max":
        return MaxPool()
    if pool == "mean":
        return MeanPool()
    if pool == "size":
        return SizePool()
    raise ValueError(f"unknown pool: {pool}")


def GLASS_E(hidden_dim: int,
            conv_layer: int,
            max_deg: int,
            output_channels: int,
            pool: str = "size",
            dropout: float = 0.3,
            jk: int = 1,
            z_ratio: float = 0.8,
            aggr: str = "mean",
            activation=nn.ELU(inplace=True),
            gn: bool = True):
    """
    모델 생성 엔트리. (실행 코드에서 그대로 models.GLASS_E(...) 호출)
    """
    use_jk = bool(jk)

    conv = EmbZGConvEdge(
        hidden_channels=hidden_dim,
        output_channels=hidden_dim,
        num_layers=conv_layer,
        max_deg=max_deg,
        dropout=dropout,
        activation=activation,
        conv=EdgeGLASSConv,
        gn=gn,
        jk=use_jk,
        z_ratio=z_ratio,
        aggr=aggr,
    )

    # ✅ FIX: JK면 conv output dim이 hidden_dim * conv_layer 로 커짐
    pred_in_dim = hidden_dim * conv_layer if use_jk else hidden_dim

    pools = nn.ModuleList([_make_pool(pool)])
    preds = nn.ModuleList([
        MLP(pred_in_dim, hidden_dim, output_channels,
            num_layers=2, dropout=dropout, activation=activation, gn=False)
    ])
    return GLASSEdge(conv=conv, preds=preds, pools=pools)
