# test_edge_model_union.py
# Ablation runner for EdgeModel (UNION local graph per minibatch; NO segregated copies).
# Goal: match original runner's overall compute pattern while removing CPU bottlenecks.
#
# Key changes vs previous "union_fast":
#   - Build union_nodes / membership / union_edges / normalized adjacency ALL on GPU
#   - Avoid GPU->CPU->GPU roundtrips for local_adj_index/value
#   - Keep per-instance raw cache on CPU (optionally pinned), but move batch tensors once

import argparse
import json
import os
import random
import time
import yaml

import numpy as np
import torch
import torch.nn as nn
import optuna
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from impl import edge_models as models, metrics, config
from edge_dataset import SubGDataset, datasets


# ----------------------------
# utils
# ----------------------------
def set_seed(seed: int):
    print("seed ", seed, flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def repeat_seed(r: int) -> int:
    rr = int(r)
    if rr < 0:
        raise ValueError("repeat index r must be >= 0")
    return (1 << rr) - 1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="DocRED")
    p.add_argument("--optruns", type=int, default=100)
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--use_seed", action="store_true")

    # node feature options
    p.add_argument("--use_deg", action="store_true")
    p.add_argument("--use_one", action="store_true")
    p.add_argument("--use_nodeid", action="store_true")
    p.add_argument("--path", type=str, default="./Emb/node/")  # for nodeid emb

    # model fixed
    p.add_argument("--hidden_dim", type=int, default=64)

    # train regime
    p.add_argument("--max_epoch", type=int, default=500)
    p.add_argument("--val_start", type=int, default=100)
    p.add_argument("--log_every", type=int, default=10)

    # optuna objective: multi-seed
    p.add_argument("--obj_repeats", type=int, default=3)

    # config (KEEP SAME AS ORIGINAL)
    p.add_argument("--config_dir", type=str, default="config/edge/EdgeModel")

    # (optional) memory safety: store local_x0 as fp16 on gpu
    p.add_argument("--preproc_fp16", action="store_true")

    # (optional) pin raw cache to speed H2D
    p.add_argument("--pin_raw", action="store_true")

    return p.parse_args()


args = parse_args()
config.set_device(args.device)

baseG = None
trn_dataset = None
val_dataset = None
tst_dataset = None

output_channels = 1
score_fn = None
loss_fn = None

_BASE_EW = None
_NODEID_CPU_CACHE = {}


def _dataset_prefix(name: str) -> str:
    emb_name_map = {
        "VisualGenome": "vg",
        "DocRED": "docred",
        "Connectome": "connectome",
    }
    return emb_name_map.get(name, name.lower())


def _get_nodeid_emb_cpu(dataset_name: str, hidden_dim: int) -> torch.Tensor:
    key = (_dataset_prefix(dataset_name), int(hidden_dim))
    if key in _NODEID_CPU_CACHE:
        return _NODEID_CPU_CACHE[key]

    prefix, hd = key
    emb_file = os.path.join(args.path, f"{prefix}_{hd}.pt")
    if not os.path.exists(emb_file):
        raise FileNotFoundError(f"nodeid embedding missing: {emb_file}")

    print(f"[emb] load ONCE {emb_file} (cpu cache)", flush=True)
    emb = torch.load(emb_file, map_location=torch.device("cpu"))
    if isinstance(emb, dict) and "x" in emb:
        emb = emb["x"]
    if isinstance(emb, nn.Embedding):
        emb = emb.weight.detach()

    emb = emb.detach().to(torch.float32)
    if emb.ndim == 3 and emb.shape[1] == 1:
        emb = emb.squeeze(1)

    _NODEID_CPU_CACHE[key] = emb
    return emb


def base_edge_weight_ones() -> torch.Tensor:
    global _BASE_EW
    if (
        _BASE_EW is None
        or _BASE_EW.device != baseG.edge_index.device
        or _BASE_EW.numel() != baseG.edge_index.shape[1]
    ):
        ei = baseG.edge_index
        _BASE_EW = torch.ones((ei.shape[1],), device=ei.device, dtype=torch.float32)
    return _BASE_EW


def split_and_features():
    global baseG, trn_dataset, val_dataset, tst_dataset
    global output_channels, score_fn, loss_fn
    global _BASE_EW

    baseG = datasets.load_dataset(args.dataset)

    unique_labels = baseG.y.unique()
    if len(unique_labels) == 2 and baseG.y.ndim == 1:
        baseG.y = baseG.y.to(torch.float)
        output_channels = 1
        score_fn = metrics.binaryf1

        def _loss(x, y):
            return BCEWithLogitsLoss()(x.reshape(-1), y.reshape(-1))

        loss_fn = _loss

    elif baseG.y.ndim > 1:
        baseG.y = baseG.y.to(torch.float)
        output_channels = int(baseG.y.shape[1])
        score_fn = metrics.binaryf1

        def _loss(x, y):
            return BCEWithLogitsLoss()(x, y)

        loss_fn = _loss

    else:
        baseG.y = baseG.y.to(torch.int64)
        output_channels = int(len(unique_labels))
        score_fn = metrics.microf1
        loss_fn = CrossEntropyLoss()

    # features
    if args.use_deg:
        baseG.setDegreeFeature()
    elif args.use_one:
        baseG.setOneFeature()
    elif args.use_nodeid:
        baseG.setNodeIdFeature()
    else:
        baseG.setNodeIdFeature()

    if args.use_nodeid:
        emb_cpu = _get_nodeid_emb_cpu(args.dataset, int(args.hidden_dim))
        baseG.x = emb_cpu.to(config.device, non_blocking=True)
    else:
        x = baseG.x
        if isinstance(x, torch.Tensor) and x.ndim == 1:
            x = x.reshape(-1, 1)
        baseG.x = x.to(config.device).to(torch.float32)

    baseG.to(config.device)

    trn_dataset = SubGDataset.GDataset(*baseG.get_split("train"))
    val_dataset = SubGDataset.GDataset(*baseG.get_split("valid"))
    tst_dataset = SubGDataset.GDataset(*baseG.get_split("test"))

    _BASE_EW = None


# ----------------------------
# raw batch parsing helpers
# ----------------------------
@torch.no_grad()
def _extract_batch_fields(batch):
    try:
        _, _, _, pos, subG_edge, y = batch
    except ValueError:
        _, _, _, pos, subG_edge, _, y = batch
    return pos, subG_edge, y


def _pad_node_list_to_1d(subG_node: torch.Tensor):
    if subG_node is None or (not isinstance(subG_node, torch.Tensor)) or subG_node.numel() == 0:
        return None
    if subG_node.ndim >= 3:
        subG_node = subG_node.squeeze(-1).squeeze(-1).squeeze()
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


def _pad_edge_list_to_2d(subG_edge: torch.Tensor):
    if subG_edge is None or (not isinstance(subG_edge, torch.Tensor)) or subG_edge.numel() == 0:
        return None
    e = subG_edge.reshape(-1, 2)
    valid = (e[:, 0] >= 0) & (e[:, 1] >= 0)
    e = e[valid]
    if e.numel() == 0:
        return None
    return e


# ----------------------------
# UNION model wrapper
# ----------------------------
class UnionEdgeModel(nn.Module):
    """
    base graph + ONE union local graph per minibatch.
    Pool per subgraph using membership indices.

    pack dict:
      local_x0: (M,F) on device
      copy2orig: (M,) base node ids (union_nodes) on device
      local_adj_index/value: normalized (2,El)/(El,) on device
      memb_idx: (K,) indices into union nodes [0..M-1] on device
      memb_gid: (K,) subgraph id for each membership entry on device
      B: int
    """
    def __init__(self, in_channels: int, hidden_dim: int, out_channels: int,
                 num_layers: int, dropout: float, alpha: float):
        super().__init__()
        H = int(hidden_dim)
        L = int(num_layers)
        if L <= 0:
            raise ValueError("num_layers must be >= 1")

        self.alpha = float(alpha)
        self.dropout = float(dropout)

        self.base_in = nn.Linear(int(in_channels), H)
        self.local_in = nn.Linear(int(in_channels), H)
        self.layers = nn.ModuleList([models.EdgeConv(H, dropout=self.dropout) for _ in range(L)])
        self.pred = nn.Linear(H, int(out_channels))

    def _pool_by_membership_fast(self, local_h: torch.Tensor, memb_idx: torch.Tensor, memb_gid: torch.Tensor, B: int):
        H = int(local_h.shape[1])
        out = local_h.new_zeros((B, H))
        if memb_idx.numel() == 0:
            return out
        out.index_add_(0, memb_gid, local_h[memb_idx])
        cnt = local_h.new_zeros((B, 1))
        ones = local_h.new_ones((memb_gid.numel(), 1))
        cnt.index_add_(0, memb_gid, ones)
        return out / cnt.clamp_min(1.0)

    def forward(self, x, edge_index, edge_weight, pack: dict):
        if not (0.5 < float(self.alpha) <= 1.0):
            raise ValueError(f"alpha must be in (0.5,1], got {self.alpha}")

        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.dtype in (torch.int32, torch.int64, torch.int16, torch.uint8):
            x = x.to(torch.float32)

        base_x = self.base_in(x)
        base_x = torch.relu(base_x)
        base_x = torch.dropout(base_x, p=self.dropout, train=self.training)

        local_x0 = pack["local_x0"]
        copy2orig = pack["copy2orig"]
        local_adj_index = pack["local_adj_index"]
        local_adj_value = pack["local_adj_value"]
        memb_idx = pack["memb_idx"]
        memb_gid = pack["memb_gid"]
        B = int(pack["B"])

        local_x = self.local_in(local_x0)
        local_x = torch.relu(local_x)
        local_x = torch.dropout(local_x, p=self.dropout, train=self.training)

        for layer in self.layers:
            base_x, local_x = layer(
                base_x, edge_index, edge_weight,
                local_x,
                local_edge_index=None,
                local_edge_weight=None,
                local_adj_index=local_adj_index,
                local_adj_value=local_adj_value,
                copy2orig=copy2orig,
                alpha=float(self.alpha),
            )

        sub_emb = self._pool_by_membership_fast(local_x, memb_idx, memb_gid, B)
        return self.pred(sub_emb)


def build_model(params):
    hidden = int(params["hidden_dim"])
    dropout = float(params["dropout"])
    layers = int(params["num_layers"])
    alpha = float(params["alpha"])

    x = baseG.x
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    in_dim = int(x.shape[1])

    return UnionEdgeModel(
        in_channels=in_dim,
        hidden_dim=hidden,
        out_channels=output_channels,
        num_layers=layers,
        dropout=dropout,
        alpha=alpha,
    ).to(config.device)


# ----------------------------
# Preprocess: raw nodes/edges per instance (CPU)
# ----------------------------
_PREPROC_RAW = {"train": None, "valid": None, "test": None}


@torch.no_grad()
def preprocess_split_to_raw(dataset, split_name: str):
    loader = SubGDataset.GDataloader(dataset, batch_size=1, shuffle=False, drop_last=False)
    raw = []
    ys = []

    t0 = time.time()
    for it, batch in enumerate(loader):
        pos, subG_edge, y = _extract_batch_fields(batch)

        nodes_list = _pad_node_list_to_1d(pos.cpu())
        nodes = nodes_list[0].to(torch.long).cpu() if nodes_list is not None else torch.empty((0,), dtype=torch.long)

        if subG_edge is None:
            edges = torch.empty((0, 2), dtype=torch.long)
        else:
            e2 = _pad_edge_list_to_2d(subG_edge.cpu())
            edges = e2.to(torch.long).cpu() if e2 is not None else torch.empty((0, 2), dtype=torch.long)

        if args.pin_raw:
            nodes = nodes.pin_memory()
            edges = edges.pin_memory()

        raw.append({"nodes": nodes, "edges": edges})
        ys.append(y.detach().cpu())

        if (it + 1) % 5000 == 0:
            print(f"[preproc_raw:{split_name}] {it+1} done", flush=True)

    dt = time.time() - t0
    print(f"[preproc_raw:{split_name}] done: {len(raw)} instances, time {dt:.2f}s", flush=True)
    return raw, ys


@torch.no_grad()
def preprocess_all_splits_once():
    global _PREPROC_RAW
    if _PREPROC_RAW["train"] is not None:
        return
    print("=" * 60, flush=True)
    print("[preproc_raw] caching raw nodes/edges per instance (CPU) ...", flush=True)
    print("=" * 60, flush=True)
    _PREPROC_RAW["train"] = preprocess_split_to_raw(trn_dataset, "train")
    _PREPROC_RAW["valid"] = preprocess_split_to_raw(val_dataset, "valid")
    _PREPROC_RAW["test"] = preprocess_split_to_raw(tst_dataset, "test")


# ----------------------------
# Build UNION pack per minibatch (GPU path)
# ----------------------------
@torch.no_grad()
def build_union_pack_from_instances(raw_list, ys_list, indices):
    """
    Build union pack on GPU to avoid CPU bottleneck.
    """
    device = config.device
    B = int(len(indices))
    if B <= 0:
        raise ValueError("empty batch indices")

    # ---- gather nodes/edges (CPU tensors) ----
    nodes_cpu = [raw_list[i]["nodes"] for i in indices]
    edges_cpu = [raw_list[i]["edges"] for i in indices]

    # ---- move concatenated nodes once ----
    sizes = torch.tensor([int(n.numel()) for n in nodes_cpu], dtype=torch.long)
    K = int(sizes.sum().item())

    if K == 0:
        union_nodes = torch.empty((0,), device=device, dtype=torch.long)
        memb_idx = torch.empty((0,), device=device, dtype=torch.long)
        memb_gid = torch.empty((0,), device=device, dtype=torch.long)
    else:
        all_nodes = torch.cat(nodes_cpu, dim=0).to(device=device, dtype=torch.long, non_blocking=True)  # (K,)
        union_nodes = torch.unique(all_nodes, sorted=True)  # (M,)
        # membership mapping via searchsorted on GPU
        memb_idx = torch.searchsorted(union_nodes, all_nodes)
        memb_gid = torch.repeat_interleave(torch.arange(B, device=device, dtype=torch.long), sizes.to(device))

    M = int(union_nodes.numel())

    # ---- edges: move once, map via searchsorted on GPU, dedup on GPU ----
    if M == 0:
        local_adj_index = torch.empty((2, 0), device=device, dtype=torch.long)
        local_adj_value = torch.empty((0,), device=device, dtype=torch.float32)
    else:
        e_sizes = torch.tensor([int(e.shape[0]) for e in edges_cpu], dtype=torch.long)
        E = int(e_sizes.sum().item())

        if E == 0:
            local_adj_index = torch.empty((2, 0), device=device, dtype=torch.long)
            local_adj_value = torch.empty((0,), device=device, dtype=torch.float32)
        else:
            all_e = torch.cat(edges_cpu, dim=0).to(device=device, dtype=torch.long, non_blocking=True)  # (E,2)
            u = all_e[:, 0]
            v = all_e[:, 1]
            iu = torch.searchsorted(union_nodes, u)
            iv = torch.searchsorted(union_nodes, v)

            # optional safety: if any u/v not found (shouldn't happen)
            ok = (iu >= 0) & (iu < M) & (iv >= 0) & (iv < M) & (union_nodes[iu] == u) & (union_nodes[iv] == v)
            if not bool(ok.all().item()):
                iu = iu[ok]
                iv = iv[ok]

            if iu.numel() == 0:
                local_adj_index = torch.empty((2, 0), device=device, dtype=torch.long)
                local_adj_value = torch.empty((0,), device=device, dtype=torch.float32)
            else:
                e_dir = torch.stack([iu, iv], dim=1)
                e_rev = torch.stack([iv, iu], dim=1)
                e2 = torch.cat([e_dir, e_rev], dim=0)
                e2 = torch.unique(e2, dim=0)  # dedup directed edges
                local_ei = e2.t().contiguous()
                local_ew = torch.ones((local_ei.shape[1],), device=device, dtype=torch.float32)

                adj = models.buildAdj(local_ei, local_ew, M, aggr="gcn").coalesce()
                local_adj_index = adj.indices()
                local_adj_value = adj.values()

    # ---- local_x0 from baseG.x (already on device) ----
    if M == 0:
        local_x0 = torch.empty((0, baseG.x.shape[1]), device=device, dtype=torch.float32)
        copy2orig = torch.empty((0,), device=device, dtype=torch.long)
    else:
        copy2orig = union_nodes  # base node ids
        local_x0 = baseG.x[copy2orig]
        if local_x0.ndim == 1:
            local_x0 = local_x0.reshape(-1, 1)
        local_x0 = local_x0.to(torch.float32)
        if args.preproc_fp16:
            local_x0 = local_x0.to(torch.float16)

    # ---- labels ----
    y_list = [ys_list[i] for i in indices]
    if isinstance(y_list[0], torch.Tensor) and y_list[0].ndim > 0:
        y_cpu = torch.cat(y_list, dim=0)
    else:
        y_cpu = torch.stack(y_list, dim=0)
    y = y_cpu.to(device=device, non_blocking=True)

    pack = {
        "local_x0": local_x0,
        "copy2orig": copy2orig,
        "local_adj_index": local_adj_index,
        "local_adj_value": local_adj_value,
        "memb_idx": memb_idx,
        "memb_gid": memb_gid,
        "B": B,
    }
    return pack, y


# ----------------------------
# train/eval
# ----------------------------
def train_epoch_union(model, optimizer, raw_list, ys_list, batch_size, loss_fn):
    model.train()
    n = len(raw_list)
    order = list(range(n))
    random.shuffle(order)

    x = baseG.x
    edge_index = baseG.edge_index
    edge_weight = base_edge_weight_ones()

    total_loss = 0.0
    nb = 0

    for s in range(0, n, batch_size):
        idxs = order[s:s + batch_size]
        if not idxs:
            continue

        pack, y = build_union_pack_from_instances(raw_list, ys_list, idxs)

        optimizer.zero_grad(set_to_none=True)
        out = model(x, edge_index, edge_weight, pack=pack)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        nb += 1

    return total_loss / max(1, nb)


@torch.no_grad()
def eval_union(model, raw_list, ys_list, batch_size, score_fn, loss_fn=None):
    model.eval()

    x = baseG.x
    edge_index = baseG.edge_index
    edge_weight = base_edge_weight_ones()

    outs = []
    ys = []

    n = len(raw_list)
    for s in range(0, n, batch_size):
        idxs = list(range(s, min(n, s + batch_size)))
        pack, y = build_union_pack_from_instances(raw_list, ys_list, idxs)
        out = model(x, edge_index, edge_weight, pack=pack)
        outs.append(out.detach())
        ys.append(y.detach())

    out_all = torch.cat(outs, dim=0) if outs else torch.empty((0, output_channels), device=config.device)
    y_all = torch.cat(ys, dim=0) if ys else torch.empty((0,), device=config.device)

    score = score_fn(out_all.detach().cpu(), y_all.detach().cpu())
    if loss_fn is None:
        return float(score), None

    loss = loss_fn(out_all, y_all)
    return float(score), float(loss.item())


def train_and_eval(params, seed, max_epoch=None):
    set_seed(int(seed))

    gnn = build_model(params)

    batch_size = int(params["batch_size"])
    lr = float(params["lr"])
    resi = float(params["resi"])

    (trn_raw, trn_ys) = _PREPROC_RAW["train"]
    (val_raw, val_ys) = _PREPROC_RAW["valid"]
    (tst_raw, tst_ys) = _PREPROC_RAW["test"]

    optimizer = Adam(gnn.parameters(), lr=lr)
    scd = lr_scheduler.ReduceLROnPlateau(optimizer, factor=resi, min_lr=5e-5)

    val_score = 0.0
    tst_score = 0.0
    early_stop = 0
    trn_time = []

    max_epoch = int(args.max_epoch if max_epoch is None else max_epoch)
    val_start = int(args.val_start)

    num_div = max(1.0, float(max(1, len(tst_raw) // max(1, batch_size))))

    for i in range(max_epoch):
        t1 = time.time()
        loss = train_epoch_union(gnn, optimizer, trn_raw, trn_ys, batch_size, loss_fn)
        trn_time.append(time.time() - t1)
        scd.step(loss)

        if i >= val_start / num_div:
            score, _ = eval_union(gnn, val_raw, val_ys, batch_size, score_fn, loss_fn=loss_fn)

            if score > val_score:
                early_stop = 0
                val_score = score
                score_tst, _ = eval_union(gnn, tst_raw, tst_ys, batch_size, score_fn, loss_fn=loss_fn)
                tst_score = max(score_tst, tst_score)
                print(f"iter {i} loss {loss:.4f} val {val_score:.4f} tst {tst_score:.4f}", flush=True)

            elif score >= val_score - 1e-5:
                score_tst, _ = eval_union(gnn, tst_raw, tst_ys, batch_size, score_fn, loss_fn=loss_fn)
                tst_score = max(score_tst, tst_score)
                if i % int(args.log_every) == 0:
                    print(f"iter {i} loss {loss:.4f} val {score:.4f} tst {score_tst:.4f}", flush=True)

            else:
                early_stop += 1
                if i % int(args.log_every) == 0:
                    tst_curr, _ = eval_union(gnn, tst_raw, tst_ys, batch_size, score_fn, loss_fn=loss_fn)
                    print(f"iter {i} loss {loss:.4f} val {score:.4f} tst {tst_curr:.4f}", flush=True)

        if val_score >= 1 - 1e-5:
            early_stop += 1
        if early_stop > 100 / num_div:
            break

    print(
        f"end: epoch {i+1}, train time {sum(trn_time):.2f} s, val {val_score:.3f}, tst {tst_score:.3f}",
        flush=True,
    )
    return float(val_score), float(tst_score)


# ----------------------------
# optuna objective (multi-seed)
# ----------------------------
def objective(trial):
    params = {
        "hidden_dim": int(args.hidden_dim),
        "num_layers": trial.suggest_int("num_layers", 1, 8),
        "dropout": trial.suggest_float("dropout", 0.0, 0.6),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_int("batch_size", 24, 210),
        "resi": trial.suggest_float("resi", 0.3, 0.9),
        "alpha": trial.suggest_float("alpha", 0.500001, 1.0),
    }

    k = int(args.obj_repeats)
    if k <= 0:
        raise ValueError("--obj_repeats must be >= 1")

    scores = []
    for r in range(k):
        seed = repeat_seed(r)  # 0, 1, 3, 7, ...
        try:
            _, tst_score = train_and_eval(params, seed=seed, max_epoch=min(300, int(args.max_epoch)))
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()
            raise
        scores.append(float(tst_score))

    mean_score = float(np.mean(scores))
    trial.set_user_attr("scores", scores)
    trial.set_user_attr("mean", mean_score)
    return mean_score


# ----------------------------
# main (KEEP SAME config path behavior)
# ----------------------------
def main():
    if args.use_seed:
        set_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    split_and_features()
    preprocess_all_splits_once()

    os.makedirs(args.config_dir, exist_ok=True)
    cfg_path = os.path.join(args.config_dir, f"{args.dataset}.yml")

    if not os.path.exists(cfg_path):
        print("=" * 40, flush=True)
        print(f"Config not found. Starting Optuna Optimization ({args.optruns} trials)...", flush=True)
        print("=" * 40, flush=True)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=int(args.optruns))

        best_params = dict(study.best_trial.params)
        best_params["hidden_dim"] = int(args.hidden_dim)
        best_params["obj_repeats"] = int(args.obj_repeats)

        with open(cfg_path, "w") as f:
            yaml.dump(best_params, f)

        print(f"Best trial params: {study.best_trial.params}", flush=True)
        print(f"Best score (mean test over {args.obj_repeats} seeds): {study.best_value:.4f}", flush=True)
        print(f"Saved best config to {cfg_path}", flush=True)
        params = best_params
    else:
        print(f"Loading config from {cfg_path}", flush=True)
        with open(cfg_path) as f:
            params = yaml.safe_load(f)
        params["hidden_dim"] = int(params.get("hidden_dim", args.hidden_dim))
        if "alpha" not in params:
            params["alpha"] = 0.8

    print(f"\n[UNION ABLATION] Running Final Evaluation on Test Set ({args.repeat} repeats)...", flush=True)
    print("Params:", params, flush=True)

    tst_scores = []
    for r in range(int(args.repeat)):
        print(f"\n--- Repeat {r}/{args.repeat} ---", flush=True)
        seed = repeat_seed(r) 
        _, tst = train_and_eval(params, seed=seed, max_epoch=int(args.max_epoch))
        tst_scores.append(tst)

    tst_scores = np.array(tst_scores, dtype=float)
    mean = float(np.mean(tst_scores))
    err = float(np.std(tst_scores) / np.sqrt(len(tst_scores)))

    print("\n" + "=" * 40, flush=True)
    print(f"[UNION ABLATION] Final Results over {args.repeat} runs:", flush=True)
    print(f"Average Test Score: {mean:.3f} error {err:.3f}", flush=True)
    print("=" * 40, flush=True)

    out = {
        "dataset": args.dataset,
        "model": "EdgeModel_UNION_LOCAL_GPU",
        "params": params,
        "test_scores": tst_scores.tolist(),
        "mean": mean,
        "error": err,
        "max_epoch": int(args.max_epoch),
        "val_start": int(args.val_start),
        "obj_repeats": int(args.obj_repeats),
        "preproc_fp16": bool(args.preproc_fp16),
        "pin_raw": bool(args.pin_raw),
    }

    with open(f"{args.dataset}_edge_model_union_results.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
