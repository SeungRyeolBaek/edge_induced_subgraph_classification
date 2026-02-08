# edge_models.py (in impl/)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn.norm import GraphNorm, GraphSizeNorm
from torch_geometric.nn.glob import global_mean_pool, global_add_pool, global_max_pool
from .utils import pad2batch
import math

class Seq(nn.Module):
    ''' 
    An extension of nn.Sequential. 
    Args: 
        modlist an iterable of modules to add.
    '''
    def __init__(self, modlist):
        super().__init__()
        self.modlist = nn.ModuleList(modlist)

    def forward(self, *args, **kwargs):
        out = self.modlist[0](*args, **kwargs)
        for i in range(1, len(self.modlist)):
            out = self.modlist[i](out)
        return out

class MLP(nn.Module):
    '''
    Multi-Layer Perception.
    Args:
        tail_activation: whether to use activation function at the last layer.
        activation: activation function.
        gn: whether to use GraphNorm layer.
    '''
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
        self.seq = None
        if num_layers == 1:
            modlist.append(nn.Linear(input_channels, output_channels))
            if tail_activation:
                if gn:
                    modlist.append(GraphNorm(output_channels))
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout, inplace=True))
                modlist.append(activation)
            self.seq = Seq(modlist)
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
    '''
        Calculating the normalized adjacency matrix.
        Args:
            n_node: number of nodes in graph.
            aggr: the aggregation method, can be "mean", "sum" or "gcn".
        '''
    adj = torch.sparse_coo_tensor(edge_index,
                                  edge_weight,
                                  size=(n_node, n_node))
    deg = torch.sparse.sum(adj, dim=(1, )).to_dense().flatten()
    deg[deg < 0.5] += 1.0
    if aggr == "mean":
        deg = 1.0 / deg
        return torch.sparse_coo_tensor(edge_index,
                                       deg[edge_index[0]] * edge_weight,
                                       size=(n_node, n_node))
    elif aggr == "sum":
        return torch.sparse_coo_tensor(edge_index,
                                       edge_weight,
                                       size=(n_node, n_node))
    elif aggr == "gcn":
        deg = torch.pow(deg, -0.5)
        return torch.sparse_coo_tensor(edge_index,
                                       deg[edge_index[0]] * edge_weight *
                                       deg[edge_index[1]],
                                       size=(n_node, n_node))
    else:
        raise NotImplementedError

class GLASSConv(torch.nn.Module):
    '''
    A kind of message passing layer we use for GLASS.
    We use different parameters to transform the features of node with different labels individually, and mix them.
    Args:
        aggr: the aggregation method.
        z_ratio: the ratio to mix the transformed features.
    '''
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
        self.activation = activation
        self.aggr = aggr
        self.gn = GraphNorm(out_channels)
        self.z_ratio = z_ratio
        self.reset_parameters()
        self.dropout = dropout

    def reset_parameters(self):
        for _ in self.trans_fns:
            _.reset_parameters()
        for _ in self.comb_fns:
            _.reset_parameters()
        self.gn.reset_parameters()

    def forward(self, x_, edge_index, edge_weight, mask):
        if self.adj.shape[0] == 0:
            n_node = x_.shape[0]
            self.adj = buildAdj(edge_index, edge_weight, n_node, self.aggr)
        # transform node features with different parameters individually.
        x1 = self.activation(self.trans_fns[1](x_))
        x0 = self.activation(self.trans_fns[0](x_))
        # mix transformed feature.
        x = torch.where(mask, self.z_ratio * x1 + (1 - self.z_ratio) * x0,
                        self.z_ratio * x0 + (1 - self.z_ratio) * x1)
        # pass messages.
        x = self.adj @ x
        x = self.gn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.cat((x, x_), dim=-1)
        # transform node features with different parameters individually.
        x1 = self.comb_fns[1](x)
        x0 = self.comb_fns[0](x)
        # mix transformed feature.
        x = torch.where(mask, self.z_ratio * x1 + (1 - self.z_ratio) * x0,
                        self.z_ratio * x0 + (1 - self.z_ratio) * x1)
        return x


class EmbZGConv(nn.Module):
    '''
    combination of some GLASSConv layers, normalization layers, dropout layers, and activation function.
    Args:
        max_deg: the max integer in input node features.
        conv: the message passing layer we use.
        gn: whether to use GraphNorm.
        jk: whether to use Jumping Knowledge Network.
    '''
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
        self.input_emb = nn.Embedding(max_deg + 1,
                                      hidden_channels,
                                      scale_grad_by_freq=False)
        self.emb_gn = GraphNorm(hidden_channels)
        self.convs = nn.ModuleList()
        self.jk = jk
        for _ in range(num_layers - 1):
            self.convs.append(
                conv(in_channels=hidden_channels,
                     out_channels=hidden_channels,
                     activation=activation,
                     **kwargs))
        self.convs.append(
            conv(in_channels=hidden_channels,
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
                self.gns.append(
                    GraphNorm(output_channels +
                              (num_layers - 1) * hidden_channels))
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
        if not (self.gns is None):
            for gn in self.gns:
                gn.reset_parameters()

    def forward(self, x, edge_index, edge_weight, z=None):
        # z is the node label.
        if z is None:
            mask = (torch.zeros(
                (x.shape[0]), device=x.device) < 0.5).reshape(-1, 1)
        else:
            mask = (z > 0.5).reshape(-1, 1)
        # convert integer input to vector node features.
        x = self.input_emb(x).reshape(x.shape[0], -1)
        x = self.emb_gn(x)
        xs = []
        x = F.dropout(x, p=self.dropout, training=self.training)
        # pass messages at each layer.
        for layer, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight, mask)
            xs.append(x)
            if not (self.gns is None):
                x = self.gns[layer](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight, mask)
        xs.append(x)

        if self.jk:
            x = torch.cat(xs, dim=-1)
            if not (self.gns is None):
                x = self.gns[-1](x)
            return x
        else:
            x = xs[-1]
            if not (self.gns is None):
                x = self.gns[-1](x)
            return x


class PoolModule(nn.Module):
    '''
    Modules used for pooling node embeddings to produce subgraph embeddings.
    Args: 
        trans_fn: module to transfer node embeddings.
        pool_fn: module to pool node embeddings like global_add_pool.
    '''
    def __init__(self, pool_fn, trans_fn=None):
        super().__init__()
        self.pool_fn = pool_fn
        self.trans_fn = trans_fn

    def forward(self, x, batch):
        # The j-th element in batch vector is i if node j is in the i-th subgraph.
        # for example [0,1,0,0,1,1,2,2] means nodes 0,2,3 in subgraph 0, nodes 1,4,5 in subgraph 1, and nodes 6,7 in subgraph 2.
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
        if x is not None:
            if self.trans_fn is not None:
                x = self.trans_fn(x)
        x = GraphSizeNorm()(x, batch)
        return self.pool_fn(x, batch)


class GLASS(nn.Module):
    '''
    GLASS model: combine message passing layers and mlps and pooling layers.
    Args:
        preds and pools are ModuleList containing the same number of MLPs and Pooling layers.
        preds[id] and pools[id] is used to predict the id-th target. Can be used for SSL.
    '''
    def __init__(self, conv: EmbZGConv, preds: nn.ModuleList,
                 pools: nn.ModuleList):
        super().__init__()
        self.conv = conv
        self.preds = preds
        self.pools = pools

    def NodeEmb(self, x, edge_index, edge_weight, z=None):
        embs = []
        for _ in range(x.shape[1]):
            emb = self.conv(x[:, _, :].reshape(x.shape[0], x.shape[-1]),
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


# models used for producing node embeddings.


class MyGCNConv(torch.nn.Module):
    '''
    A kind of message passing layer we use for pretrained GNNs.
    Args:
        aggr: the aggregation method.
    '''
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation=nn.ReLU(inplace=True),
                 aggr="mean"):
        super().__init__()
        self.trans_fn = nn.Linear(in_channels, out_channels)
        self.comb_fn = nn.Linear(in_channels + out_channels, out_channels)
        self.adj = torch.sparse_coo_tensor(size=(0, 0))
        self.activation = activation
        self.aggr = aggr
        self.gn = GraphNorm(out_channels)

    def reset_parameters(self):
        self.trans_fn.reset_parameters()
        self.comb_fn.reset_parameters()
        self.gn.reset_parameters()

    def forward(self, x_, edge_index, edge_weight):
        if self.adj.shape[0] == 0:
            n_node = x_.shape[0]
            self.adj = buildAdj(edge_index, edge_weight, n_node, self.aggr)
        x = self.trans_fn(x_)
        x = self.activation(x)
        x = self.adj @ x
        x = self.gn(x)
        x = torch.cat((x, x_), dim=-1)
        x = self.comb_fn(x)
        return x

class EmbGConv(torch.nn.Module):
    '''
    combination of some message passing layers, normalization layers, dropout layers, and activation function.
    Args:
        max_deg: the max integer in input node features.
        conv: the message passing layer we use.
        gn: whether to use GraphNorm.
        jk: whether to use Jumping Knowledge Network.
    '''
    def __init__(self,
                 input_channels: int,
                 hidden_channels: int,
                 output_channels: int,
                 num_layers: int,
                 max_deg: int,
                 dropout=0,
                 activation=nn.ReLU(inplace=True),
                 conv=GCNConv,
                 gn=True,
                 jk=False,
                 **kwargs):
        super().__init__()
        self.input_emb = nn.Embedding(max_deg + 1, hidden_channels)
        self.convs = nn.ModuleList()
        self.jk = jk
        if num_layers > 1:
            self.convs.append(
                conv(in_channels=input_channels,
                     out_channels=hidden_channels,
                     **kwargs))
            for _ in range(num_layers - 2):
                self.convs.append(
                    conv(in_channels=hidden_channels,
                         out_channels=hidden_channels,
                         **kwargs))
            self.convs.append(
                conv(in_channels=hidden_channels,
                     out_channels=output_channels,
                     **kwargs))
        else:
            self.convs.append(
                conv(in_channels=input_channels,
                     out_channels=output_channels,
                     **kwargs))
        self.activation = activation
        self.dropout = dropout
        if gn:
            self.gns = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.gns.append(GraphNorm(hidden_channels))
        else:
            self.gns = None
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if not (self.gns is None):
            for gn in self.gns:
                gn.reset_parameters()

    def forward(self, x, edge_index, edge_weight, z=None):
        xs = []
        x = F.dropout(self.input_emb(x.reshape(-1)),
                      p=self.dropout,
                      training=self.training)
        for layer, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            if not (self.gns is None):
                x = self.gns[layer](x)
            xs.append(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(self.convs[-1](x, edge_index, edge_weight))
        if self.jk:
            return torch.cat(xs, dim=-1)
        else:
            return xs[-1]

class EdgeGNN(nn.Module):
    '''
    EdgeGNN model: combine message passing layers and mlps and pooling layers to do link prediction task.
    Args:
        preds and pools are ModuleList containing the same number of MLPs and Pooling layers.
        preds[id] and pools[id] is used to predict the id-th target. Can be used for SSL.
    '''
    def __init__(self, conv, preds: nn.ModuleList, pools: nn.ModuleList):
        super().__init__()
        self.conv = conv
        self.preds = preds
        self.pools = pools

    def NodeEmb(self, x, edge_index, edge_weight, z=None):
        embs = []
        for _ in range(x.shape[1]):
            emb = self.conv(x[:, _, :].reshape(x.shape[0], x.shape[-1]),
                            edge_index, edge_weight, z)
            embs.append(emb.reshape(emb.shape[0], 1, emb.shape[-1]))
        emb = torch.cat(embs, dim=1)
        emb = torch.mean(emb, dim=1)
        return emb

    def Pool(self, emb, subG_node, pool):
        emb = emb[subG_node]
        emb = torch.mean(emb, dim=1)
        return emb

    def forward(self, x, edge_index, edge_weight, subG_node, z=None, id=0):
        emb = self.NodeEmb(x, edge_index, edge_weight, z)
        emb = self.Pool(emb, subG_node, self.pools[id])
        return self.preds[id](emb)

# ============================
# EdgeModel: base + segregated copies with layer-wise mixing
# ============================

def _squeeze_x(x: torch.Tensor) -> torch.Tensor:
    """
    Accepts:
      - (N,) long/int  -> (N,1)
      - (N,1)         -> (N,1)
      - (N,F)         -> (N,F)
      - (N,1,F)       -> (N,F)
    """
    if x is None:
        raise ValueError("x is None")
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be torch.Tensor, got {type(x)}")
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim == 2:
        return x
    if x.ndim == 3 and x.shape[1] == 1:
        return x.squeeze(1)
    return x


def _pad_edge_list_to_2d(subG_edge: torch.Tensor):
    """
    subG_edge:
      - (B, maxE, 2) or (maxE, 2) padded with -1
    return:
      - (K,2) valid edges (base node ids)
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


def _pad_node_list_to_1d(subG_node: torch.Tensor):
    """
    subG_node:
      - (B, maxN) padded -1
      - or (maxN,)
      - OR sometimes comes with extra singleton dims like:
          (B, 1, 1), (B, maxN, 1), (B, 1, maxN), (1, B, maxN) ...
    return:
      - list of 1D tensors per subgraph (base node ids)
    """
    if subG_node is None or (not isinstance(subG_node, torch.Tensor)) or subG_node.numel() == 0:
        return None

    # ---- normalize shapes by squeezing singleton dims ----
    # common bad case: (B,1,1) -> (B,)
    # (B,maxN,1) -> (B,maxN)
    # (B,1,maxN) -> (B,maxN)
    if subG_node.ndim >= 3:
        # squeeze ONLY singleton dims, keep batch dimension if possible
        subG_node = subG_node.squeeze(-1)
        subG_node = subG_node.squeeze(-1)
        subG_node = subG_node.squeeze()

    # now handle canonical forms
    if subG_node.ndim == 1:
        n = subG_node[subG_node >= 0]
        return [n]

    if subG_node.ndim == 2:
        out = []
        for i in range(subG_node.shape[0]):
            n = subG_node[i]
            out.append(n[n >= 0])
        return out

    raise ValueError(f"subG_node must be (B,maxN) or (maxN,), got {tuple(subG_node.shape)}")

def scatter_mean_to_base(copy_h: torch.Tensor, copy2orig: torch.Tensor, N: int):
    """
    copy_h:   (M,H)
    copy2orig:(M,) base node ids
    return:   (N,H) mean over copies (0 if no copies)
    """
    device = copy_h.device
    H = copy_h.shape[1]
    out = torch.zeros((N, H), device=device, dtype=copy_h.dtype)
    cnt = torch.zeros((N, 1), device=device, dtype=copy_h.dtype)

    out.index_add_(0, copy2orig, copy_h)
    ones = torch.ones((copy2orig.shape[0], 1), device=device, dtype=copy_h.dtype)
    cnt.index_add_(0, copy2orig, ones)
    out = out / torch.clamp(cnt, min=1.0)
    return out


def _build_gcn_adj(edge_index: torch.Tensor, edge_weight: torch.Tensor, n_node: int):
    """
    Build normalized adjacency like GCN: D^{-1/2} A D^{-1/2}.
    Returns:
      adj_index: (2,E)
      adj_value: (E,)
    """
    adj = buildAdj(edge_index, edge_weight, n_node, aggr="gcn").coalesce()
    return adj.indices(), adj.values()


def _build_segregated_batch(subG_node: torch.Tensor,
                            subG_edge: torch.Tensor,
                            base_x: torch.Tensor):
    """
    Build a disjoint union graph of segregated copies for the CURRENT BATCH.

    Nodes:
      - for each subgraph i: one copy node per valid base node in subG_node[i]
        (so copies are "node-segregated": same base node can appear multiple times across subgraphs)
    Edges:
      - use subG_edge[i] edges that are fully inside subG_node[i]
      - mapped to local indices within that subgraph copy-block

    Returns dict:
      local_x      : (M, F) (copied features from base_x[orig])
      local_ei     : (2, E_local) (COO)
      local_ew     : (E_local,) (ones)
      copy2orig    : (M,) base node id for each copy
      copy_ptr     : (B+1,) prefix ptr for each subgraph block
    """
    nodes_list = _pad_node_list_to_1d(subG_node)
    if nodes_list is None:
        raise ValueError("subG_node is required for EdgeModel (to build segregated copies).")

    B = len(nodes_list)
    device = base_x.device

    sizes = [int(n.numel()) for n in nodes_list]
    ptr = torch.zeros((B + 1,), device=device, dtype=torch.long)
    if B > 0:
        ptr[1:] = torch.cumsum(torch.tensor(sizes, device=device, dtype=torch.long), dim=0)
    M = int(ptr[-1].item())

    if M == 0:
        local_x = base_x.new_zeros((0, base_x.shape[1]))
        local_ei = torch.empty((2, 0), device=device, dtype=torch.long)
        local_ew = torch.empty((0,), device=device, dtype=torch.float32)
        copy2orig = torch.empty((0,), device=device, dtype=torch.long)
        return {
            "local_x": local_x,
            "local_ei": local_ei,
            "local_ew": local_ew,
            "copy2orig": copy2orig,
            "copy_ptr": ptr,
        }

    copy2orig = torch.empty((M,), device=device, dtype=torch.long)
    for i, n in enumerate(nodes_list):
        s = int(ptr[i].item())
        e = int(ptr[i + 1].item())
        copy2orig[s:e] = n.to(device=device, dtype=torch.long)

    local_x = base_x[copy2orig]

    edge_chunks = []
    if subG_edge is None:
        local_ei = torch.empty((2, 0), device=device, dtype=torch.long)
        local_ew = torch.empty((0,), device=device, dtype=torch.float32)
        return {
            "local_x": local_x,
            "local_ei": local_ei,
            "local_ew": local_ew,
            "copy2orig": copy2orig,
            "copy_ptr": ptr,
        }

    if subG_edge.ndim == 2 and subG_edge.shape[-1] == 2 and B == 1:
        subG_edge_b = subG_edge.unsqueeze(0)
    else:
        subG_edge_b = subG_edge
    if subG_edge_b.ndim != 3 or subG_edge_b.shape[0] != B or subG_edge_b.shape[-1] != 2:
        local_ei = torch.empty((2, 0), device=device, dtype=torch.long)
        local_ew = torch.empty((0,), device=device, dtype=torch.float32)
        return {
            "local_x": local_x,
            "local_ei": local_ei,
            "local_ew": local_ew,
            "copy2orig": copy2orig,
            "copy_ptr": ptr,
        }

    for i in range(B):
        s = int(ptr[i].item())
        n = nodes_list[i].to(device=device, dtype=torch.long)
        if n.numel() == 0:
            continue

        mapping = {int(n[j].item()): j for j in range(n.numel())}

        e_i = subG_edge_b[i]
        if e_i.numel() == 0:
            continue
        e2 = _pad_edge_list_to_2d(e_i)
        if e2 is None:
            continue
        uu = e2[:, 0].to(device=device, dtype=torch.long)
        vv = e2[:, 1].to(device=device, dtype=torch.long)

        src = []
        dst = []
        for k in range(int(e2.shape[0])):
            u = int(uu[k].item())
            v = int(vv[k].item())
            if (u in mapping) and (v in mapping):
                lu = mapping[u] + s
                lv = mapping[v] + s
                src.append(lu); dst.append(lv)
                src.append(lv); dst.append(lu)

        if len(src) > 0:
            ei_local = torch.tensor([src, dst], device=device, dtype=torch.long)
            edge_chunks.append(ei_local)

    if len(edge_chunks) == 0:
        local_ei = torch.empty((2, 0), device=device, dtype=torch.long)
    else:
        local_ei = torch.cat(edge_chunks, dim=1)

    local_ew = torch.ones((local_ei.shape[1],), device=device, dtype=torch.float32)
    return {
        "local_x": local_x,
        "local_ei": local_ei,
        "local_ew": local_ew,
        "copy2orig": copy2orig,
        "copy_ptr": ptr,
    }


class EdgeConv(nn.Module):
    """
    One layer:
      - base graph message passing (GCN-style normalized adjacency, cached)
      - local graph message passing (either:
            (a) pre-normalized local adjacency in pack, or
            (b) raw local edges and normalize on the fly fallback)
      - layer-wise mixing:
          local <- alpha * local + (1-alpha) * base[orig]
          base  <- alpha * base  + (1-alpha) * mean(local copies per orig)
    """
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        H = int(hidden_dim)
        self.base_lin = nn.Linear(H, H)
        self.local_lin = nn.Linear(H, H)
        self.base_gn = GraphNorm(H)
        self.local_gn = GraphNorm(H)
        self.dropout = float(dropout)

        self._base_adj_index = None
        self._base_adj_value = None

    def reset_parameters(self):
        self.base_lin.reset_parameters()
        self.local_lin.reset_parameters()
        self.base_gn.reset_parameters()
        self.local_gn.reset_parameters()
        self._base_adj_index = None
        self._base_adj_value = None

    def _maybe_build_base_adj(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        if self._base_adj_index is None:
            n = int(x.shape[0])
            idx, val = _build_gcn_adj(edge_index, edge_weight, n)
            self._base_adj_index = idx
            self._base_adj_value = val

    def forward(self,
                base_x: torch.Tensor,
                base_edge_index: torch.Tensor,
                base_edge_weight: torch.Tensor,
                local_x: torch.Tensor,
                # raw local (fallback)
                local_edge_index: torch.Tensor = None,
                local_edge_weight: torch.Tensor = None,
                # pre-normalized local (preferred)
                local_adj_index: torch.Tensor = None,
                local_adj_value: torch.Tensor = None,
                copy2orig: torch.Tensor = None,
                alpha: float = 0.8):
        a = float(alpha)
        if not (0.0 <= a <= 1.0):
            raise ValueError(f"alpha must be in [0,1], got {a}")

        # ---- base MP (cached adj) ----
        self._maybe_build_base_adj(base_x, base_edge_index, base_edge_weight)
        row = self._base_adj_index[0]
        col = self._base_adj_index[1]
        w = self._base_adj_value

        bx = self.base_lin(base_x)
        msg = bx[col] * w.reshape(-1, 1)
        out = torch.zeros_like(bx)
        out.index_add_(0, row, msg)
        out = self.base_gn(out)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        base_h = out

        # ---- local MP ----
        if local_x is None or local_x.numel() == 0:
            local_h = local_x
        else:
            lx = self.local_lin(local_x)

            # preferred: use pre-normalized adjacency
            if local_adj_index is not None and local_adj_value is not None and local_adj_index.numel() > 0:
                rL = local_adj_index[0]
                cL = local_adj_index[1]
                vL = local_adj_value
                msgL = lx[cL] * vL.reshape(-1, 1)
                outL = torch.zeros_like(lx)
                outL.index_add_(0, rL, msgL)
                local_h = outL
            else:
                # fallback: normalize on-the-fly using raw local edges
                if local_edge_index is None or local_edge_index.numel() == 0:
                    local_h = lx
                else:
                    nL = int(local_x.shape[0])
                    idxL, valL = _build_gcn_adj(local_edge_index, local_edge_weight, nL)
                    rL = idxL[0]; cL = idxL[1]
                    msgL = lx[cL] * valL.reshape(-1, 1)
                    outL = torch.zeros_like(lx)
                    outL.index_add_(0, rL, msgL)
                    local_h = outL

            local_h = self.local_gn(local_h)
            local_h = F.relu(local_h)
            local_h = F.dropout(local_h, p=self.dropout, training=self.training)

        # ---- mixing ----
        if local_h is not None and local_h.numel() > 0:
            local_mixed = a * local_h + (1.0 - a) * base_h[copy2orig]
            msg_to_base = scatter_mean_to_base(local_mixed, copy2orig, N=int(base_h.shape[0]))
        else:
            local_mixed = local_h
            msg_to_base = torch.zeros_like(base_h)

        base_mixed = a * base_h + (1.0 - a) * msg_to_base
        return base_mixed, local_mixed


class EdgeModel(nn.Module):
    """
    base graph + segregated copy graph.
    IMPORTANT:
      - If z is a dict (prebuilt pack), do NOT build segregated graph inside forward.
      - Otherwise z is treated as subG_edge and we build pack internally (legacy).
    """
    def __init__(self,
                 in_channels: int,
                 hidden_dim: int,
                 out_channels: int,
                 num_layers: int,
                 dropout: float = 0.2,
                 alpha: float = 0.8):
        super().__init__()
        H = int(hidden_dim)
        L = int(num_layers)
        if L <= 0:
            raise ValueError("num_layers must be >= 1")

        self.alpha = float(alpha)
        self.dropout = float(dropout)

        self.base_in = nn.Linear(int(in_channels), H)
        self.local_in = nn.Linear(int(in_channels), H)

        self.layers = nn.ModuleList([EdgeConv(H, dropout=self.dropout) for _ in range(L)])
        self.pred = nn.Linear(H, int(out_channels))

    def reset_parameters(self):
        self.base_in.reset_parameters()
        self.local_in.reset_parameters()
        for l in self.layers:
            l.reset_parameters()
        self.pred.reset_parameters()

    def _pool_local_to_subgraph(self, local_h: torch.Tensor, copy_ptr: torch.Tensor):
        """
        Mean pool over local copy nodes inside each subgraph block.
        copy_ptr: (B+1,) prefix sums
        returns: (B,H)
        """
        B = int(copy_ptr.numel() - 1)
        if B <= 0:
            return local_h.new_zeros((0, local_h.shape[1]))

        H = int(local_h.shape[1])
        out = local_h.new_zeros((B, H))
        for i in range(B):
            s = int(copy_ptr[i].item())
            e = int(copy_ptr[i + 1].item())
            if e > s:
                out[i] = local_h[s:e].mean(dim=0)
            else:
                out[i] = 0.0
        return out

    def forward(self, x, edge_index, edge_weight, subG_node, z=None, id=0):
        """
        Inputs:
          - x, edge_index, edge_weight: base graph
          - subG_node: (B,maxN) padded -1  (only used in legacy path)
          - z:
              (A) dict pack (preferred, cached in dataloader stage):
                    {
                      "local_x0": (M,F) float,
                      "copy2orig": (M,),
                      "copy_ptr": (B+1,),
                      "local_adj_index": (2,El) normalized,
                      "local_adj_value": (El,),
                      # optional fallbacks:
                      "local_ei": (2,El_raw),
                      "local_ew": (El_raw,)
                    }
              (B) Tensor subG_edge (legacy): (B,maxE,2) padded -1
        """
        if not (0.5 < float(self.alpha) <= 1.0):
            raise ValueError(f"EdgeModel.alpha must be in (0.5,1], got {self.alpha}")

        x0 = _squeeze_x(x)
        if x0.dtype in (torch.int32, torch.int64, torch.int16, torch.uint8):
            x0 = x0.to(torch.float32)

        base_x = self.base_in(x0)
        base_x = F.relu(base_x)
        base_x = F.dropout(base_x, p=self.dropout, training=self.training)

        # ---- pack: cached path vs legacy path ----
        if isinstance(z, dict):
            pack = z
            local_x0 = pack["local_x0"]
            copy2orig = pack["copy2orig"]
            copy_ptr = pack["copy_ptr"]
            local_adj_index = pack.get("local_adj_index", None)
            local_adj_value = pack.get("local_adj_value", None)
            local_ei = pack.get("local_ei", None)
            local_ew = pack.get("local_ew", None)
        else:
            # legacy: build per forward (kept for compatibility)
            pack = _build_segregated_batch(subG_node=subG_node, subG_edge=z, base_x=x0)
            local_x0 = pack["local_x"]
            copy2orig = pack["copy2orig"]
            copy_ptr = pack["copy_ptr"]
            local_ei = pack["local_ei"]
            local_ew = pack["local_ew"]
            # build normalized local adjacency once in this forward (legacy)
            nL = int(local_x0.shape[0])
            if nL > 0 and local_ei.numel() > 0:
                adj = buildAdj(local_ei, local_ew, nL, aggr="gcn").coalesce()
                local_adj_index = adj.indices()
                local_adj_value = adj.values()
            else:
                local_adj_index = torch.empty((2, 0), device=x0.device, dtype=torch.long)
                local_adj_value = torch.empty((0,), device=x0.device, dtype=torch.float32)

        local_x = self.local_in(local_x0)
        local_x = F.relu(local_x)
        local_x = F.dropout(local_x, p=self.dropout, training=self.training)

        for layer in self.layers:
            base_x, local_x = layer(
                base_x, edge_index, edge_weight,
                local_x,
                local_edge_index=local_ei,
                local_edge_weight=local_ew,
                local_adj_index=local_adj_index,
                local_adj_value=local_adj_value,
                copy2orig=copy2orig,
                alpha=float(self.alpha),
            )

        sub_emb = self._pool_local_to_subgraph(local_x, copy_ptr)
        out = self.pred(sub_emb)
        return out