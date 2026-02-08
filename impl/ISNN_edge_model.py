# impl/ISNN_edge_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor

# ============================================================
# helpers
# ============================================================
def _adj_matmul(adj, x):
    if isinstance(adj, SparseTensor):
        return adj.matmul(x)
    if torch.is_tensor(adj) and adj.is_sparse:
        return torch.sparse.mm(adj, x)
    return adj @ x


def _squeeze_x(x: torch.Tensor) -> torch.Tensor:
    # baseG.x: (N,1,F) or (N,F)
    if x.ndim == 3 and x.shape[1] == 1:
        return x.squeeze(1)
    if x.ndim == 2:
        return x
    raise ValueError(f"Expected x shape (N,F) or (N,1,F), got {tuple(x.shape)}")


def _pad_edge_list_to_2d(subg_edge):
    """
    Normalize subg_edge to list-per-subgraph format.

    Input supported:
      - Tensor (S, maxE, 2) with padding -1
      - list/tuple of (Ei,2) tensors/lists
      - Tensor (maxE,2) (single subgraph) -> treated as [that]
      - Tensor (K,2) (already valid) -> treated as [that]

    Return:
      - list[Tensor(Ei,2)] where each Tensor is valid edges only (no -1)
    """
    if subg_edge is None:
        return None

    # Case 1) list/tuple already
    if isinstance(subg_edge, (list, tuple)):
        out = []
        for e in subg_edge:
            if e is None:
                out.append(torch.empty((0, 2), dtype=torch.long))
                continue
            if not isinstance(e, torch.Tensor):
                e = torch.tensor(e, dtype=torch.long)
            e = e.reshape(-1, 2)
            if e.numel() == 0:
                out.append(torch.empty((0, 2), dtype=torch.long))
                continue
            valid = (e[:, 0] >= 0) & (e[:, 1] >= 0)
            out.append(e[valid].long())
        return out

    # Case 2) tensor
    if not isinstance(subg_edge, torch.Tensor):
        subg_edge = torch.tensor(subg_edge, dtype=torch.long)

    # (S, maxE, 2)
    if subg_edge.ndim == 3 and subg_edge.shape[-1] == 2:
        S = int(subg_edge.shape[0])
        out = []
        for i in range(S):
            e = subg_edge[i].reshape(-1, 2)
            if e.numel() == 0:
                out.append(torch.empty((0, 2), dtype=torch.long))
                continue
            valid = (e[:, 0] >= 0) & (e[:, 1] >= 0)
            out.append(e[valid].long())
        return out

    # (maxE, 2) single-subgraph
    if subg_edge.ndim == 2 and subg_edge.shape[-1] == 2:
        e = subg_edge.reshape(-1, 2)
        if e.numel() == 0:
            return [torch.empty((0, 2), dtype=torch.long)]
        valid = (e[:, 0] >= 0) & (e[:, 1] >= 0)
        return [e[valid].long()]

    raise ValueError(f"Unsupported subg_edge shape: {tuple(subg_edge.shape)}")


def _build_norm_adj_from_edge_index(edge_index, edge_weight, num_nodes: int, normalize: bool):
    adj = SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        value=edge_weight,
        sparse_sizes=(num_nodes, num_nodes),
    )
    if not normalize:
        return adj
    row, col, val = adj.coo()
    deg = torch.zeros((num_nodes,), device=row.device, dtype=torch.float32)
    deg.scatter_add_(0, row, torch.ones_like(val, dtype=torch.float32))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
    val = deg_inv_sqrt[row] * val * deg_inv_sqrt[col]
    return SparseTensor(row=row, col=col, value=val, sparse_sizes=(num_nodes, num_nodes))


# ============================================================
# attention (same as ISNN)
# ============================================================
class attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.W = nn.Parameter(torch.rand(in_channels, out_channels, requires_grad=True))
        self.b = nn.Parameter(torch.rand(1, out_channels, requires_grad=True))
        self.p = nn.Parameter(torch.rand(out_channels, 1, requires_grad=True))
        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.b)
        nn.init.xavier_uniform_(self.p)

    def forward(self, x, y):
        x = torch.tanh(x @ self.W + self.b) @ self.p
        y = torch.tanh(y @ self.W + self.b) @ self.p
        z = torch.cat([x, y], dim=1)
        return F.softmax(z, dim=1)


# ============================================================
# backbones
# ============================================================
class baseGNN(nn.Module):
    """
    GCN stack on SparseTensor adjacency (PyG GCNConv supports SparseTensor)
    """
    def __init__(self, in_channels, out_channels, num_layers=2, dropout=0.5, normalize=False):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, out_channels, normalize=normalize))
        for _ in range(max(0, num_layers - 1)):
            self.convs.append(GCNConv(out_channels, out_channels, normalize=normalize))
        self.initialize()

    def initialize(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj):
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, adj)
            if i != len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class IGNNBlock(nn.Module):
    """
    Minimal IGNN-style fixed point layer:
      h <- kappa * (A @ h) @ W + phi(x)
    """
    def __init__(self, in_channels, hidden_channels, num_nodes, num_layers=2, dropout=0.5, kappa=0.95):
        super().__init__()
        self.kappa = float(kappa)
        self.phi = baseGNN(in_channels, hidden_channels, num_layers=num_layers, dropout=dropout, normalize=False)
        self.F = nn.Parameter(torch.rand(hidden_channels, hidden_channels))
        self.emb = nn.Parameter(0.01 * torch.rand(num_nodes, hidden_channels))
        self.register_buffer("proxy", 0.1 * torch.randn(num_nodes, hidden_channels))

    def project(self, eps=1e-5):
        W = self.F.T @ self.F
        n = torch.norm(W)
        if n > 1:
            W = W / (n + eps)
        return W

    def forward(self, x, adj, emb=None):
        if emb is None:
            emb = self.emb
        W = self.project()
        base = self.phi(x, adj)
        return self.kappa * (_adj_matmul(adj, emb) @ W) + base

    @torch.no_grad()
    def fixed_point(self, x, adj, inner_iters=10, tol=1e-5):
        h = self.proxy
        err = torch.tensor(float("inf"), device=h.device)
        for _ in range(int(inner_iters)):
            new_h = self.forward(x, adj, emb=h)
            err = torch.norm(new_h - h)
            h = new_h
            if err < tol:
                break
        self.proxy = h
        return float(err.item())


# ============================================================
# EDGE-INDUCED local hybrid builder (segregated copies + supernode)
# ============================================================
@torch.no_grad()
def build_local_hybrid_disjoint(
    base_x: torch.Tensor,
    base_pos: torch.Tensor,
    subg_edge,                 # (S,maxE,2) or list[Ei,2]
    device,
    add_self_loops: bool = True,
    normalize_adj: bool = True,
):
    """
    Build ONE disjoint-union local-hybrid graph for all subgraphs.

    Returned dict:
      local_x: (M, F)
      local_adj: SparseTensor(M,M)
      copy2orig: (num_copies,) original node id for each copy node (0..N-1)
      copy_ptr: (S+1,)
      super_ids: (S,)
      local_node_is_copy: (M,) bool
      num_copies, num_local_nodes, num_subgraphs
    """
    x = _squeeze_x(base_x).to(device)
    if x.ndim != 2:
        raise ValueError(f"base_x must become (N,F) after squeeze, got {tuple(x.shape)}")
    if x.shape[1] == 0:
        x = torch.ones((x.shape[0], 1), device=device, dtype=torch.float32)
   
    pos = base_pos.to(device)
    S = int(pos.shape[0])
    Fdim = int(x.shape[1])

    subg_edge = _pad_edge_list_to_2d(subg_edge)
    if isinstance(subg_edge, torch.Tensor):
        raise ValueError("subg_edge as Tensor must be list-per-subgraph or (S,maxE,2) before flattening.")

    # build copies
    copy_ptr = [0]
    copy2orig = []
    local_feats = []
    for i in range(S):
        nodes = pos[i]
        nodes = nodes[nodes >= 0].long()
        copy2orig.append(nodes)
        local_feats.append(x[nodes])
        copy_ptr.append(copy_ptr[-1] + int(nodes.numel()))

    copy_ptr = torch.tensor(copy_ptr, dtype=torch.long, device=device)  # (S+1,)
    if len(copy2orig) == 0:
        raise ValueError("No subgraphs found (pos empty).")

    copy2orig = torch.cat(copy2orig, dim=0)   # (num_copies,)
    copies_x = torch.cat(local_feats, dim=0)  # (num_copies,F)
    num_copies = int(copies_x.shape[0])

    # supernodes (S)
    super_x = torch.zeros((S, Fdim), device=device, dtype=copies_x.dtype)
    for i in range(S):
        a, b = int(copy_ptr[i].item()), int(copy_ptr[i + 1].item())
        if b > a:
            super_x[i] = copies_x[a:b].mean(dim=0)

    local_x = torch.cat([copies_x, super_x], dim=0)  # (M,F)
    M = int(local_x.shape[0])
    super_ids = torch.arange(num_copies, num_copies + S, device=device, dtype=torch.long)

    # edges
    rows = []
    cols = []

    # induced edges among copies
    for i in range(S):
        a, b = int(copy_ptr[i].item()), int(copy_ptr[i + 1].item())
        if b <= a:
            continue

        orig_nodes = copy2orig[a:b]  # (ni,)

        sorted_vals, sorted_idx = torch.sort(orig_nodes)
        sorted_vals = sorted_vals.contiguous()
        local_ids = torch.arange(a, b, device=device, dtype=torch.long)[sorted_idx]

        e = subg_edge[i].to(device)
        if e.numel() == 0:
            continue
        u = e[:, 0].long().contiguous()
        v = e[:, 1].long().contiguous()

        iu = torch.searchsorted(sorted_vals, u)
        iv = torch.searchsorted(sorted_vals, v)

        ok = (
            (iu < sorted_vals.numel()) &
            (iv < sorted_vals.numel()) &
            (sorted_vals[iu] == u) &
            (sorted_vals[iv] == v)
        )
        if ok.any():
            lu = local_ids[iu[ok]]
            lv = local_ids[iv[ok]]
            rows.append(lu)
            cols.append(lv)

    # membership edges: copy -> supernode
    for i in range(S):
        a, b = int(copy_ptr[i].item()), int(copy_ptr[i + 1].item())
        if b <= a:
            continue
        cids = torch.arange(a, b, device=device, dtype=torch.long)
        sid = super_ids[i].expand_as(cids)
        rows.append(cids)
        cols.append(sid)

    if len(rows) == 0:
        edge_index = torch.empty((2, 0), device=device, dtype=torch.long)
        edge_weight = torch.empty((0,), device=device, dtype=torch.float32)
    else:
        row = torch.cat(rows, dim=0)
        col = torch.cat(cols, dim=0)
        edge_index = torch.stack([row, col], dim=0)
        edge_weight = torch.ones(edge_index.shape[1], device=device, dtype=torch.float32)

    if add_self_loops:
        self_loops = torch.arange(M, device=device, dtype=torch.long)
        sl = torch.stack([self_loops, self_loops], dim=0)
        edge_index = torch.cat([edge_index, sl], dim=1)
        edge_weight = torch.cat([edge_weight, torch.ones(M, device=device)], dim=0)

    local_adj = _build_norm_adj_from_edge_index(edge_index, edge_weight, num_nodes=M, normalize=normalize_adj)
    local_node_is_copy = torch.zeros((M,), device=device, dtype=torch.bool)
    local_node_is_copy[:num_copies] = True

    return {
        "local_x": local_x,
        "local_adj": local_adj,
        "copy2orig": copy2orig,
        "copy_ptr": copy_ptr,
        "super_ids": super_ids,
        "local_node_is_copy": local_node_is_copy,
        "num_copies": num_copies,
        "num_local_nodes": M,
        "num_subgraphs": S,
    }


# ============================================================
# pooling / aggregation utils
# ============================================================
@torch.no_grad()
def pool_subgraph_from_local(local_h, copy_ptr, super_ids, use_attention=False, att_module=None):
    """
    local_h: (M,H)
    copy_ptr: (S+1,)
    super_ids: (S,)
    returns: (S,H)
    """
    S = int(super_ids.numel())
    H = int(local_h.shape[1])
    sub_from_copies = torch.zeros((S, H), device=local_h.device, dtype=local_h.dtype)
    for i in range(S):
        a, b = int(copy_ptr[i].item()), int(copy_ptr[i + 1].item())
        if b > a:
            sub_from_copies[i] = local_h[a:b].mean(dim=0)

    sub_from_super = local_h[super_ids]  # (S,H)

    if not use_attention:
        return sub_from_super

    if att_module is None:
        raise ValueError("use_attention=True requires att_module")
    att = att_module(sub_from_copies, sub_from_super)  # (S,2)
    return att[:, 0:1] * sub_from_copies + att[:, 1:2] * sub_from_super


@torch.no_grad()
def scatter_mean_to_base(copy_h, copy2orig, N: int):
    """
    copy_h: (num_copies,H)
    copy2orig: (num_copies,)
    return: (N,H)
    """
    H = int(copy_h.shape[1])
    out = torch.zeros((N, H), device=copy_h.device, dtype=copy_h.dtype)
    cnt = torch.zeros((N,), device=copy_h.device, dtype=torch.float32)
    out.index_add_(0, copy2orig, copy_h)
    ones = torch.ones((copy_h.shape[0],), device=copy_h.device, dtype=torch.float32)
    cnt.index_add_(0, copy2orig, ones)
    cnt = cnt.clamp(min=1.0).view(-1, 1)
    return out / cnt


# ============================================================
# heads
# ============================================================
class MLPHead(nn.Module):
    def __init__(self, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)
