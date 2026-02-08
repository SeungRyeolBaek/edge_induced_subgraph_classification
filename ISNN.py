# ISNNTest_node.py
import argparse
import os
import random
import yaml
import json
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler

import optuna

from torch_geometric.utils import add_self_loops, degree
from torch_sparse import SparseTensor
import time
from impl import metrics, config
from impl.ISNN_model import SubIGNN_new, VanillaGCN
from edge_dataset import node_datasets


def set_seed(seed: int):
    print(f"seed {seed}", flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# KNN adjacency (original-style)
# ----------------------------
def remove_redundant_edges(edge_index, edge_weight):
    """
    Remove redundant bidirectional edges based on in-degree.
    If (a->b) and (b->a) exist, keep only one:
      - keep the one whose source has higher in-degree
      - if tie, arbitrarily drop (a->b)
    """
    if edge_index.numel() == 0:
        return edge_index, edge_weight

    num_nodes = int(edge_index.max().item() + 1)
    in_degree = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    ones = torch.ones_like(edge_index[1], dtype=torch.long, device=edge_index.device)
    in_degree.scatter_add_(0, edge_index[1], ones)

    edge_list = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    edge_set = set(edge_list)
    to_remove = set()

    for (src, tgt) in edge_set:
        if (tgt, src) in edge_set:
            if in_degree[src] > in_degree[tgt]:
                to_remove.add((src, tgt))
            elif in_degree[src] < in_degree[tgt]:
                to_remove.add((tgt, src))
            else:
                to_remove.add((src, tgt))

    mask = torch.tensor([(s, t) not in to_remove for (s, t) in edge_list],
                        dtype=torch.bool, device=edge_index.device)
    return edge_index[:, mask], edge_weight[mask]


def get_knn_adj(embeddings, k, distance_metric='cosine', normalize_rows=True, binary=True):
    """
    Generate k-NN adjacency (directed) from embeddings, then remove redundant bidirectional edges.
    NOTE: original code used an offset m=100 (skip top-100 closest); we preserve it.
    """
    if isinstance(embeddings, torch.Tensor):
        emb_np = embeddings.detach().float().cpu().numpy()
    else:
        emb_np = np.asarray(embeddings, dtype=np.float32)

    num_nodes = emb_np.shape[0]
    if num_nodes <= 1 or k <= 0:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,), dtype=torch.float32)

    if distance_metric == 'euclidean':
        dist_matrix = np.linalg.norm(emb_np[:, None] - emb_np[None, :], axis=2)
    elif distance_metric == 'cosine':
        denom = np.linalg.norm(emb_np, axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        normed = emb_np / denom
        dist_matrix = 1.0 - np.dot(normed, normed.T)
    else:
        raise ValueError("Unsupported distance metric. Use 'euclidean' or 'cosine'.")

    m = 100
    kk = min(k, max(0, num_nodes - 1 - m))
    if kk <= 0:
        kk = min(k, num_nodes - 1)
        neighbors = np.argsort(dist_matrix, axis=1)[:, 1:kk + 1]
        distances = np.sort(dist_matrix, axis=1)[:, 1:kk + 1]
    else:
        neighbors = np.argsort(dist_matrix, axis=1)[:, m:m + kk]
        distances = np.sort(dist_matrix, axis=1)[:, m:m + kk]

    if normalize_rows and distances.size > 0:
        row_sums = distances.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        distances = distances / row_sums

    row_indices = np.repeat(np.arange(num_nodes), neighbors.shape[1])
    col_indices = neighbors.reshape(-1)

    if binary:
        edge_weights = np.ones_like(col_indices, dtype=np.float32)
    else:
        edge_weights = distances.reshape(-1).astype(np.float32)

    edge_index = torch.tensor(np.stack([row_indices, col_indices], axis=0), dtype=torch.long)
    edge_weight = torch.tensor(edge_weights, dtype=torch.float32)

    edge_index, edge_weight = remove_redundant_edges(edge_index, edge_weight)
    return edge_index, edge_weight


class SubgraphProjection(nn.Module):
    """
    Sparse matrix P (num_subgraphs x num_nodes), P[i, v]=1 if v in subgraph i
    forward: P @ node_embeddings -> subgraph_embeddings
    """
    def __init__(self, device, baseG, normalize=False):
        super().__init__()
        self.num_rows = baseG.pos.shape[0]
        self.num_cols = baseG.x.shape[0]
        self.normalize = normalize

        row_indices = []
        col_indices = []

        for row_idx in range(self.num_rows):
            valid_nodes = baseG.pos[row_idx][baseG.pos[row_idx] != -1].tolist()
            row_indices.extend([row_idx] * len(valid_nodes))
            col_indices.extend(valid_nodes)

        row_indices = torch.tensor(row_indices, dtype=torch.long)
        col_indices = torch.tensor(col_indices, dtype=torch.long)
        values = torch.ones(len(row_indices), dtype=torch.float32)

        P = SparseTensor(
            row=row_indices,
            col=col_indices,
            value=values,
            sparse_sizes=(self.num_rows, self.num_cols),
        )

        if self.normalize:
            row_sums = P.sum(dim=1)
            row_sums_inv = 1.0 / (row_sums + 1e-8)
            norm_factors = row_sums_inv[row_indices]
            values = values * norm_factors
            P = SparseTensor(
                row=row_indices,
                col=col_indices,
                value=values,
                sparse_sizes=(self.num_rows, self.num_cols),
            )

        self.projection_matrix = P.to(device)

    def forward(self, input_matrix):
        return self.projection_matrix.matmul(input_matrix)


def build_hybrid_graph(baseG, device, subgraph_edge_index=None, subgraph_edge_weight=None, normalize=True):
    """
    Hybrid graph:
      - original nodes: 0..N-1
      - subgraph supernodes: N..N+S-1
      - membership edges: node -> supernode
      - optional supernode-supernode edges
    """
    N = baseG.x.shape[0]
    S = baseG.pos.shape[0]

    src_nodes = []
    dst_nodes = []
    subgraph_features = []

    x = baseG.x.to(device)
    if x.ndim == 3:
        x = x.squeeze(1)  # (N,F)
    if x.ndim != 2:
        raise ValueError(f"Expected base node features 2D after squeeze, got {tuple(x.shape)}")

    ei = baseG.edge_index.to(device)

    for i in range(S):
        nodes = baseG.pos[i][baseG.pos[i] != -1].tolist()
        src_nodes.extend(nodes)
        dst_nodes.extend([N + i] * len(nodes))
        subgraph_features.append(torch.mean(x[nodes], dim=0))  # (F,)

    edge_index_hybrid = torch.cat(
        [ei, torch.tensor([src_nodes, dst_nodes], dtype=torch.long, device=device)],
        dim=1
    )

    edge_weight_hybrid = torch.cat(
        [
            torch.ones(ei.shape[1], device=device),
            torch.ones(len(src_nodes), device=device),
        ],
        dim=0
    )

    if subgraph_edge_index is not None:
        sub_ei = subgraph_edge_index.to(device)
        if subgraph_edge_weight is None:
            sub_ew = torch.ones(sub_ei.shape[1], device=device)
        else:
            sub_ew = subgraph_edge_weight.to(device)

        edge_index_hybrid = torch.cat([edge_index_hybrid, sub_ei], dim=1)
        edge_weight_hybrid = torch.cat([edge_weight_hybrid, sub_ew], dim=0)

    edge_index_hybrid, edge_weight_hybrid = add_self_loops(
        edge_index_hybrid, edge_weight_hybrid, num_nodes=N + S
    )

    adj = SparseTensor(
        row=edge_index_hybrid[0],
        col=edge_index_hybrid[1],
        value=edge_weight_hybrid,
        sparse_sizes=(N + S, N + S)
    )

    if normalize:
        row, col, value = adj.coo()
        deg = degree(row, num_nodes=N + S, dtype=torch.float32)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        value = deg_inv_sqrt[row] * value * deg_inv_sqrt[col]
        adj = SparseTensor(row=row, col=col, value=value, sparse_sizes=(N + S, N + S))

    feature = torch.cat([x, torch.stack(subgraph_features).to(device)], dim=0)  # (N+S,F)
    node_mask = torch.tensor([True] * N + [False] * S, device=device)
    return feature, adj, node_mask


def train_one_epoch(model, optimizer, features, adj, baseG, train_idx, loss_fn):
    model.train()
    optimizer.zero_grad()
    logits = model.classify(train_mask=train_idx)
    y = baseG.y[train_idx]
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def eval_split(model, features, adj, baseG, idx, score_fn, loss_fn=None):
    model.eval()
    logits = model.classify(train_mask=idx)
    y = baseG.y[idx]

    loss = None
    if loss_fn is not None:
        loss = float(loss_fn(logits, y).item())

    pred_np = logits.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    score = float(score_fn(pred_np, y_np))
    return score, loss


def build_isnn(params, baseG, projection, node_mask, device, output_channels, loss_fn):
    hidden_dim = int(params["hidden_dim"])
    conv_layer = int(params.get("conv_layer", 8))
    dropout = float(params["dropout"])
    gamma = float(params.get("gamma", 0.01))
    kappa = float(params.get("kappa", 0.95))

    in_channels = int(baseG.x.shape[-1])
    num_nodes_total = baseG.x.shape[0] + baseG.pos.shape[0]

    gnn = SubIGNN_new(
        in_channels=in_channels,
        out_channels=hidden_dim,
        num_classes=int(output_channels),
        num_nodes=int(num_nodes_total),
        projection_matrix=projection,
        node_mask=node_mask,
        num_layers=conv_layer,
        dropout=dropout,
        loss_fn=loss_fn,
        gamma=gamma,
        kappa=kappa,
    ).to(device)

    return gnn


def pretrain_and_get_subgraph_knn_edges(
    baseG, device, output_channels,
    hidden_dim, conv_layer, dropout,
    lr, weight_decay,
    pretrain_epochs, knn_k,
):
    """
    Force original --pretrain behavior:
      1) VanillaGCN pretrains for subgraph classification for pretrain_epochs
      2) Use trained embeddings -> get_knn_adj() -> subgraph-supernode edges
    Returns: (subgraph_edge_index, subgraph_edge_weight) with node ids in [0..S-1] (no offset yet)
    """
    x = baseG.x
    if x.ndim == 3:
        x = x.squeeze(1)
    x = x.to(device)

    projection = SubgraphProjection(device, baseG, normalize=False).to(device)

    pre = VanillaGCN(
        in_channels=int(x.shape[1]),
        out_channels=int(hidden_dim),
        num_classes=int(output_channels),
        num_layers=int(conv_layer),
        projection_matrix=projection,
        loss_fn=None,
        dropout=float(dropout),
    ).to(device)

    opt = Adam(pre.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    train_idx = (baseG.mask == 0).nonzero(as_tuple=True)[0].to(device)

    if output_channels == 1 and baseG.y.ndim == 1:
        y_all = baseG.y.float().to(device)
        crit = nn.BCEWithLogitsLoss()

        def _loss(logits):
            return crit(logits.view(-1), y_all[train_idx].view(-1))

    elif baseG.y.ndim > 1:
        y_all = baseG.y.float().to(device)
        crit = nn.BCEWithLogitsLoss()

        def _loss(logits):
            return crit(logits, y_all[train_idx])

    else:
        y_all = baseG.y.long().to(device)
        crit = nn.CrossEntropyLoss()

        def _loss(logits):
            return crit(logits, y_all[train_idx])

    for ep in range(1, int(pretrain_epochs) + 1):
        pre.train()
        opt.zero_grad()

        out = pre(x, baseG.edge_index.to(device))

        if out.shape[0] == baseG.pos.shape[0] and out.shape[-1] == int(output_channels):
            logits = out[train_idx]
        else:
            if hasattr(pre, "classifier"):
                logits = pre.classifier(out)[train_idx]
            elif hasattr(pre, "cls"):
                logits = pre.cls(out)[train_idx]
            else:
                if not hasattr(pre, "_tmp_head"):
                    pre._tmp_head = nn.Linear(out.shape[1], int(output_channels)).to(device)
                logits = pre._tmp_head(out)[train_idx]

        loss = _loss(logits)
        loss.backward()
        opt.step()

        if ep % 20 == 0 or ep == int(pretrain_epochs):
            print(f"[pretrain] ep {ep}/{pretrain_epochs} loss {float(loss.item()):.4f}", flush=True)

    pre.eval()
    with torch.no_grad():
        emb = pre(x, baseG.edge_index.to(device)).detach()

    sub_ei, sub_ew = get_knn_adj(
        emb, int(knn_k),
        distance_metric="cosine",
        normalize_rows=True,
        binary=True
    )
    return sub_ei.to(device), sub_ew.to(device)


def run_once(
    params, seed, max_epoch, baseG, device,
    output_channels, score_fn, loss_fn,
    pretrain_epochs=200, val_start=200,
    log_every=50,
):
    set_seed(seed)

    train_sub = (baseG.mask == 0).nonzero(as_tuple=True)[0].to(device)
    val_sub   = (baseG.mask == 1).nonzero(as_tuple=True)[0].to(device)
    test_sub  = (baseG.mask == 2).nonzero(as_tuple=True)[0].to(device)

    lr = float(params["lr"])
    wd = float(params.get("weight_decay", 0.0))
    inner_iters = int(params.get("inner_iters", 5))

    hidden_dim = int(params["hidden_dim"])
    conv_layer = int(params.get("conv_layer", 8))
    dropout = float(params.get("dropout", 0.0))

    # ---- always pretrain (original --pretrain behavior) ----
    knn = int(params.get("knn", 30))
    sub_ei, sub_ew = pretrain_and_get_subgraph_knn_edges(
        baseG=baseG,
        device=device,
        output_channels=output_channels,
        hidden_dim=hidden_dim,
        conv_layer=conv_layer,
        dropout=dropout,
        lr=lr,
        weight_decay=wd,
        pretrain_epochs=int(pretrain_epochs),
        knn_k=knn,
    )

    # supernode offset
    N = baseG.x.shape[0]
    sub_ei = sub_ei + N

    # build hybrid graph
    features, adj, node_mask = build_hybrid_graph(
        baseG, device,
        subgraph_edge_index=sub_ei,
        subgraph_edge_weight=sub_ew,
        normalize=True
    )

    projection = SubgraphProjection(device, baseG, normalize=False).to(device)
    gnn = build_isnn(params, baseG, projection, node_mask, device, output_channels, loss_fn)

    optimizer = Adam(gnn.parameters(), lr=lr, weight_decay=wd)
    scd = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=5e-5)

    best_val = -1.0
    tst_score = -1.0
    early_stop = 0

    log_every = max(1, int(log_every))
    max_epoch = int(max_epoch)
    val_start = int(val_start)

    trn_time = []

    for ep in range(1, max_epoch + 1):
        # fixed point update
        gnn.fixed_point_iteration(features, adj, inner_iters=inner_iters)

        # train step timing (너가 보여준 스타일)
        t1 = time.time()
        loss_val = train_one_epoch(gnn, optimizer, features, adj, baseG, train_sub, loss_fn)
        trn_time.append(time.time() - t1)

        scd.step(loss_val)

        if ep >= val_start:
            score, _ = eval_split(gnn, features, adj, baseG, val_sub, score_fn, loss_fn=None)
            score_tst, _ = eval_split(gnn, features, adj, baseG, test_sub, score_fn, loss_fn=None)

            if score > best_val:
                early_stop = 0
                best_val = score
                tst_score = max(tst_score, score_tst)  
                print(f"iter {ep} loss {loss_val:.4f} val {best_val:.4f} tst {tst_score:.4f}", flush=True)

            elif score >= best_val - 1e-5:
                tst_score = max(tst_score, score_tst)
                # print(f"iter {ep} loss {loss_val:.4f} val {best_val:.4f} tst {score_tst:.4f}", flush=True)

            else:
                early_stop += 1
                if ep % log_every == 0:
                    print(f"iter {ep} loss {loss_val:.4f} val {score:.4f} tst {score_tst:.4f}", flush=True)
        if early_stop > 100:
            break

    # AUROC는 학습 도중 절대 출력/계산하지 않고, 여기서 딱 1번만 계산
    tst_auroc, _ = eval_split(gnn, features, adj, baseG, test_sub, metrics.auroc, loss_fn=None)

    print(
        f"end: epoch {ep}, train time {sum(trn_time):.2f} s, val {best_val:.3f}, tst {tst_score:.3f}, auroc {tst_auroc:.3f}",
        flush=True
    )
    return best_val, tst_score, tst_auroc

def objective(
    trial, baseG, device, output_channels, score_fn, loss_fn,
    fixed_hidden_dim: int, pretrain_epochs: int, val_start: int,
    hpo_epochs: int, log_every: int,
):
    params = {
        "hidden_dim": int(fixed_hidden_dim),
        "conv_layer": 8,
        "dropout": trial.suggest_float("dropout", 0.0, 0.6),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 1e-4),
        "gamma": trial.suggest_float("gamma", 1e-4, 5e-2, log=True),
        "kappa": trial.suggest_float("kappa", 0.85, 0.99),
        "inner_iters": trial.suggest_int("inner_iters", 1, 10),
        "switch_epoch": -1,
        "knn": trial.suggest_int("knn", 5, 50),
    }

    try:
        val_score, tst_score, _ = run_once(
            params=params,
            seed=0,
            max_epoch=int(hpo_epochs),
            baseG=baseG,
            device=device,
            output_channels=output_channels,
            score_fn=score_fn,
            loss_fn=loss_fn,
            pretrain_epochs=int(pretrain_epochs),
            val_start=int(val_start),
            log_every=int(log_every),
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()
        raise

    print(f"[trial] val {val_score:.4f} tst {tst_score:.4f} params {params}", flush=True)
    return val_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="DocRED")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--optruns", type=int, default=100)
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument("--use_seed", action="store_true")
    p.add_argument("--config_dir", type=str, default="config/node/ISNN")
    p.add_argument("--path", type=str, default="./Emb/node/")
    p.add_argument("--hidden_dim", type=int, default=64)

    # original schedule knobs
    p.add_argument("--pretrain_epochs", type=int, default=200)
    p.add_argument("--val_start", type=int, default=200)
    p.add_argument("--final_epochs", type=int, default=1500)

    # HPO / logging knobs
    p.add_argument("--hpo_epochs", type=int, default=800)
    p.add_argument("--log_every", type=int, default=50)

    return p.parse_args()


def main():
    args = parse_args()
    config.set_device(args.device)
    device = config.device

    if args.use_seed:
        set_seed(0)

    print(f"Loading dataset {args.dataset}...", flush=True)
    baseG = node_datasets.load_dataset(args.dataset)
    baseG = baseG.to(device)

    # ----------------------------
    # Inject pretrained node features from pt (DO NOT REMOVE)
    # ----------------------------
    d = args.dataset
    if d == "Connectome":
        pt_name = "connectome"
    elif d == "DocRED":
        pt_name = "docred"
    elif d == "VisualGenome":
        pt_name = "vg"
    else:
        pt_name = d.lower()

    emb_file = os.path.join(args.path, f"{pt_name}_{args.hidden_dim}.pt")
    if not os.path.exists(emb_file):
        raise FileNotFoundError(f"node embedding pt not found: {emb_file}")

    emb = torch.load(emb_file, map_location="cpu")
    if isinstance(emb, nn.Embedding):
        emb = emb.weight.detach()
    if not isinstance(emb, torch.Tensor):
        raise TypeError(f"torch.load returned {type(emb)}, expected Tensor or nn.Embedding")

    emb = emb.float()
    if emb.ndim != 2:
        raise ValueError(f"Expected emb shape (N,D), got {tuple(emb.shape)}")

    N = baseG.x.shape[0]
    if emb.shape[0] != N:
        raise ValueError(f"Embedding N mismatch: baseG has {N} nodes but emb has {emb.shape[0]}")

    baseG.x = emb.unsqueeze(1).to(device)
    print(f"[feat] baseG.x <- {emb_file}  shape={tuple(baseG.x.shape)}", flush=True)

    # ----------------------------
    # Task type -> loss_fn/score_fn/output_channels
    # ----------------------------
    unique_labels = baseG.y.unique()
    if len(unique_labels) == 2 and baseG.y.ndim == 1:
        baseG.y = baseG.y.to(torch.float)

        def loss_fn(x, y):
            return nn.BCEWithLogitsLoss()(x.flatten(), y.flatten())

        output_channels = 1
        score_fn = metrics.binaryf1

    elif baseG.y.ndim > 1:
        baseG.y = baseG.y.to(torch.float)

        def loss_fn(x, y):
            return nn.BCEWithLogitsLoss()(x, y)

        output_channels = baseG.y.shape[1]
        score_fn = metrics.binaryf1

    else:
        baseG.y = baseG.y.to(torch.int64)
        loss_fn = nn.CrossEntropyLoss()
        output_channels = int(len(unique_labels))
        score_fn = metrics.microf1

    os.makedirs(args.config_dir, exist_ok=True)
    config_path = os.path.join(args.config_dir, f"{args.dataset}_isnn.yml")

    # -------- HPO or load config --------
    if not os.path.exists(config_path):
        print("=" * 40, flush=True)
        print(f"Config not found. Starting Optuna Optimization ({args.optruns} trials)...", flush=True)
        print("=" * 40, flush=True)

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda t: objective(
                t, baseG, device, output_channels, score_fn, loss_fn,
                args.hidden_dim, args.pretrain_epochs, args.val_start,
                args.hpo_epochs, args.log_every
            ),
            n_trials=args.optruns
        )

        best_params = study.best_trial.params
        best_params["hidden_dim"] = args.hidden_dim
        best_params["conv_layer"] = 8
        best_params["switch_epoch"] = -1

        with open(config_path, "w") as f:
            yaml.dump(best_params, f)

        print(f"Best trial params: {study.best_trial.params}", flush=True)
        print(f"Best Val Score: {study.best_value:.4f}", flush=True)
        print(f"Saved best config to {config_path}", flush=True)

        params = best_params
    else:
        print(f"Loading config from {config_path}", flush=True)
        with open(config_path) as f:
            params = yaml.safe_load(f)
        params["hidden_dim"] = int(params.get("hidden_dim", args.hidden_dim))
        params["conv_layer"] = int(params.get("conv_layer", 8))
        params["switch_epoch"] = -1

    # -------- Final evaluation --------
    print(f"\nRunning Final Evaluation on Test Set ({args.repeat} repeats)...", flush=True)
    print("Params:", params, flush=True)

    tst_scores = []
    tst_aurocs = []
    for r in range(args.repeat):
        print(f"\n--- Repeat {r}/{args.repeat} ---", flush=True)
        seed = (1 << r) - 1
        val_s, tst_s, tst_a = run_once(
            params=params,
            seed=seed,
            max_epoch=args.final_epochs,
            baseG=baseG,
            device=device,
            output_channels=output_channels,
            score_fn=score_fn,
            loss_fn=loss_fn,
            pretrain_epochs=args.pretrain_epochs,
            val_start=args.val_start,
            log_every=args.log_every,
        )
        tst_scores.append(tst_s)
        tst_aurocs.append(tst_a)

    tst_scores = np.array(tst_scores, dtype=float)
    mean = float(np.mean(tst_scores))
    err = float(np.std(tst_scores) / np.sqrt(len(tst_scores)))

    tst_aurocs = np.array(tst_aurocs, dtype=float)
    mean_auc = float(np.mean(tst_aurocs))
    err_auc = float(np.std(tst_aurocs) / np.sqrt(len(tst_aurocs)))

    print("\n" + "=" * 40, flush=True)
    print(f"Final Results over {args.repeat} runs:", flush=True)
    print(f"Average Test Score: {mean:.3f} error {err:.3f}", flush=True)
    print(f"Average Test AUROC: {mean_auc:.3f} error {err_auc:.3f}", flush=True)
    print("=" * 40, flush=True)

    out = {
        "dataset": args.dataset,
        "model": "isnn",
        "params": params,
        "test_scores": tst_scores.tolist(),
        "test_aurocs": tst_aurocs.tolist(),
        "mean": mean,
        "error": err,
        "mean_auroc": mean_auc,
        "error_auroc": err_auc,
        "emb_file": emb_file,
        "final_epochs": int(args.final_epochs),
        "pretrain_epochs": int(args.pretrain_epochs),
        "val_start": int(args.val_start),
        "hpo_epochs": int(args.hpo_epochs),
        "log_every": int(args.log_every),
    }
    with open(f"{args.dataset}_isnn_results.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
