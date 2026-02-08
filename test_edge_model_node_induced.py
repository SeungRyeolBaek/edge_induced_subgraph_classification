# test_edge_model_node_induced.py
# Train/Eval EdgeModel on NODE-INDUCED local edges inside each segregated subgraph copy.
# Variant 2: keep segregated copies (batch separability), but construct local edges by inducing from base graph E.
#
# CHANGE (ALWAYS-ON DISK CACHE):
# - Precompute (nodes, induced local edges, y) per instance ONCE per split and save to disk (.pt).
# - Next runs load from disk immediately. No CLI option needed.
#
# Runs with your fixed commands:
#   nohup python3 -u test_edge_model_node_induced.py --dataset DocRED --device 2 --repeat 10 --use_seed --use_nodeid > edge_node_induced_DocRED.log 2>&1 &
#   nohup python3 -u test_edge_model_node_induced.py --dataset VisualGenome --device 1 --repeat 10 --use_seed --use_nodeid > edge_node_induced_VisualGenome.log 2>&1 &
#   nohup python3 -u test_edge_model_node_induced.py --dataset Connectome --device 0 --repeat 10 --use_seed --use_nodeid > edge_node_induced_Connectome.log 2>&1 &

import argparse
import json
import os
import random
import time
import yaml
import hashlib

import numpy as np
import torch
import torch.nn as nn
import optuna
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from tqdm import tqdm

from impl import edge_models as models, metrics, config
from edge_dataset import SubGDataset, datasets

# ----------------------------
# GLOBALS
# ----------------------------

# CSR cache for base graph (GPU)
_BASE_ROW_PTR = None  # (N+1,) int64
_BASE_COL = None      # (E,) int64
_BASE_IS_BIDIR = None

# CSR cache for base graph (CPU)
_BASE_ROW_PTR_CPU = None  # (N+1,) int64 on CPU
_BASE_COL_CPU = None      # (E,) int64 on CPU

baseG = None
trn_dataset = None
val_dataset = None
tst_dataset = None

output_channels = 1
score_fn = None
loss_fn = None

_BASE_EW = None
_NODEID_CPU_CACHE = {}

# stamp-trick buffers (GPU) (kept for compatibility; not used in cached path)
_STAMP_MARK = None  # (N,) int32
_STAMP_POS = None   # (N,) int64
_STAMP_EPOCH = 0

# ALWAYS-ON disk cache for induced raw
_RAW_CACHE_DIR = "cache/raw_node_induced_v2"
_PREPROC_RAW = {"train": None, "valid": None, "test": None}


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

    # config (MUST match original for fair comparison)
    p.add_argument("--config_dir", type=str, default="config/edge/EdgeModel")

    # (optional) local_x0 stored as fp16 ON GPU (saves memory/bw)
    p.add_argument("--preproc_fp16", action="store_true")

    # (optional) pin raw node cache to speed H2D
    p.add_argument("--pin_raw", action="store_true")

    # (optional) reduce overhead: dedup directed edges per-subgraph
    p.add_argument("--dedup_local_ei", action="store_true")

    return p.parse_args()


args = parse_args()
config.set_device(args.device)


def _dataset_prefix(name: str) -> str:
    emb_name_map = {
        "VisualGenome": "vg",
        "DocRED": "docred",
        "Connectome": "connectome",
    }
    return emb_name_map.get(name, name.lower())


def _get_nodeid_emb_cpu(dataset_name: str, hidden_dim: int) -> torch.Tensor:
    """
    Load nodeid embedding ONCE into CPU cache and return a CPU float32 Tensor (N,F).
    """
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
    if _BASE_EW is None or _BASE_EW.device != baseG.edge_index.device or _BASE_EW.numel() != baseG.edge_index.shape[1]:
        ei = baseG.edge_index
        _BASE_EW = torch.ones((ei.shape[1],), device=ei.device, dtype=torch.float32)
    return _BASE_EW


def _estimate_base_is_bidir(edge_index_cpu: torch.Tensor, N: int, max_check: int = 200_000) -> bool:
    """
    Heuristic: sample up to max_check directed edges, test if reverse exists.
    Uses 64-bit key = u*N + v.
    """
    E = int(edge_index_cpu.shape[1])
    if E == 0:
        return True

    S = int(min(max_check, E))
    u = edge_index_cpu[0, :S].to(torch.int64)
    v = edge_index_cpu[1, :S].to(torch.int64)
    key = u * int(N) + v
    key_sorted, _ = torch.sort(key)

    rev = v * int(N) + u
    pos = torch.searchsorted(key_sorted, rev)
    ok = (pos < key_sorted.numel()) & (key_sorted[pos] == rev)
    ratio = float(ok.to(torch.float32).mean().item())
    return ratio >= 0.95


# ----------------------------
# induce local edges (CPU)
# ----------------------------
@torch.no_grad()
def _induce_edges_cpu(nodes_cpu: torch.Tensor) -> torch.Tensor:
    """
    nodes_cpu: (m,) base node ids on CPU (long).
    returns edges_local_cpu: (El, 2) local indices on CPU (long).

    Policy:
      - Use base CSR on CPU: (_BASE_ROW_PTR_CPU, _BASE_COL_CPU)
      - Keep directed edges as present in base CSR.
      - If base is not bidirectional, add reverse edges to emulate undirected local graph.
      - Optionally dedup via torch.unique (CPU, done ONCE at preprocess).
    """
    if nodes_cpu is None or (not isinstance(nodes_cpu, torch.Tensor)) or nodes_cpu.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long)

    if _BASE_ROW_PTR_CPU is None or _BASE_COL_CPU is None:
        raise RuntimeError("CPU CSR not built. Call split_and_features() first.")

    nodes_cpu = nodes_cpu.to(torch.long).cpu()
    m = int(nodes_cpu.numel())
    if m == 0:
        return torch.empty((0, 2), dtype=torch.long)

    # map base_id -> local_id
    node_list = nodes_cpu.tolist()
    pos = {nid: i for i, nid in enumerate(node_list)}

    src = []
    dst = []

    rowptr = _BASE_ROW_PTR_CPU
    col = _BASE_COL_CPU

    for u_base in node_list:
        a = int(rowptr[u_base].item())
        b = int(rowptr[u_base + 1].item())
        if b <= a:
            continue
        nbrs = col[a:b]
        for v_base in nbrs.tolist():
            j = pos.get(v_base, None)
            if j is not None:
                src.append(pos[u_base])
                dst.append(j)

    if len(src) == 0:
        return torch.empty((0, 2), dtype=torch.long)

    e = torch.tensor(list(zip(src, dst)), dtype=torch.long)

    if not bool(_BASE_IS_BIDIR):
        e = torch.cat([e, e[:, [1, 0]]], dim=0)

    if args.dedup_local_ei:
        e = torch.unique(e, dim=0)

    return e


# ----------------------------
# base CSR build (ONCE)
# ----------------------------
def _build_base_csr_once():
    """
    Build CSR (rowptr, col) for baseG.edge_index.
    Keep BOTH CPU CSR (for preprocessing induced edges without GPU sync)
    and GPU CSR (optional).
    """
    global _BASE_ROW_PTR, _BASE_COL, _BASE_IS_BIDIR
    global _BASE_ROW_PTR_CPU, _BASE_COL_CPU

    device = config.device
    N = int(baseG.x.shape[0])
    ei = baseG.edge_index

    ei_cpu = ei.detach().to("cpu", non_blocking=False).to(torch.int64)
    u = ei_cpu[0]
    v = ei_cpu[1]
    E = int(u.numel())

    if E == 0:
        rowptr = torch.zeros((N + 1,), dtype=torch.int64)
        col = torch.empty((0,), dtype=torch.int64)
        _BASE_ROW_PTR_CPU = rowptr
        _BASE_COL_CPU = col
        _BASE_ROW_PTR = rowptr.to(device)
        _BASE_COL = col.to(device)
        _BASE_IS_BIDIR = True
        print("[csr] base graph has no edges.", flush=True)
        return

    _BASE_IS_BIDIR = _estimate_base_is_bidir(ei_cpu, N=N)
    print(f"[csr] base bidirectional (heuristic): {_BASE_IS_BIDIR}", flush=True)

    perm = torch.argsort(u)
    u_sorted = u[perm]
    v_sorted = v[perm]

    deg = torch.bincount(u_sorted, minlength=N).to(torch.int64)
    rowptr = torch.zeros((N + 1,), dtype=torch.int64)
    rowptr[1:] = torch.cumsum(deg, dim=0)
    col = v_sorted.contiguous()

    _BASE_ROW_PTR_CPU = rowptr
    _BASE_COL_CPU = col

    _BASE_ROW_PTR = rowptr.to(device=device, non_blocking=True)
    _BASE_COL = col.to(device=device, non_blocking=True)


# ----------------------------
# split + features
# ----------------------------
def split_and_features():
    global baseG, trn_dataset, val_dataset, tst_dataset
    global output_channels, score_fn, loss_fn
    global _BASE_EW
    global _BASE_ROW_PTR, _BASE_COL, _BASE_IS_BIDIR
    global _BASE_ROW_PTR_CPU, _BASE_COL_CPU

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

    # reset CSR then build
    _BASE_ROW_PTR = None
    _BASE_COL = None
    _BASE_IS_BIDIR = None
    _BASE_ROW_PTR_CPU = None
    _BASE_COL_CPU = None
    _build_base_csr_once()


# ----------------------------
# model
# ----------------------------
def build_model(params):
    hidden = int(params["hidden_dim"])
    dropout = float(params["dropout"])
    layers = int(params["num_layers"])
    alpha = float(params["alpha"])

    x = baseG.x
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    in_dim = int(x.shape[1])

    model = models.EdgeModel(
        in_channels=in_dim,
        hidden_dim=hidden,
        out_channels=output_channels,
        num_layers=layers,
        dropout=dropout,
        alpha=alpha,
    ).to(config.device)
    return model


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


# ----------------------------
# ALWAYS-ON DISK CACHE: paths + (load/build/save)
# ----------------------------
def _ensure_raw_cache_dir():
    os.makedirs(_RAW_CACHE_DIR, exist_ok=True)


def _edgeindex_signature_cpu(ei: torch.Tensor, max_edges: int = 200_000) -> str:
    """
    Fast signature to avoid stale cache if base edges changed.
    Hash: (shape + first S edges).
    """
    ei_cpu = ei.detach().to("cpu", non_blocking=False).to(torch.int64)
    E = int(ei_cpu.shape[1])
    S = int(min(max_edges, E))
    samp = ei_cpu[:, :S].contiguous()
    h = hashlib.sha1()
    h.update(str(tuple(ei_cpu.shape)).encode("utf-8"))
    h.update(samp.numpy().tobytes())
    return h.hexdigest()[:16]


def _raw_cache_path(split_name: str) -> str:
    """
    Cache key depends on:
      - dataset
      - base edge_index signature
      - base bidirectional heuristic (affects whether we add reverse)
      - dedup_local_ei (affects edges)
    """
    ds = str(args.dataset)
    sig = _edgeindex_signature_cpu(baseG.edge_index)
    bidir = "bidir1" if bool(_BASE_IS_BIDIR) else "bidir0"
    dedup = "dedup1" if bool(args.dedup_local_ei) else "dedup0"
    fname = f"{ds}_{split_name}_{sig}_{bidir}_{dedup}.pt"
    return os.path.join(_RAW_CACHE_DIR, fname)


@torch.no_grad()
def _try_load_raw_cache(split_name: str):
    path = _raw_cache_path(split_name)
    if not os.path.exists(path):
        return None
    t0 = time.time()
    obj = torch.load(path, map_location="cpu")
    raw = obj["raw"]
    ys = obj["ys"]
    # (optional) re-pin after load
    if args.pin_raw:
        for it in raw:
            it["nodes"] = it["nodes"].pin_memory()
            it["edges"] = it["edges"].pin_memory()
    dt = time.time() - t0
    print(f"[raw_cache:{split_name}] loaded {len(raw)} instances from {path} ({dt:.2f}s)", flush=True)
    return (raw, ys)


@torch.no_grad()
def _build_and_save_raw_cache(dataset, split_name: str):
    """
    Build per-instance:
      - nodes: CPU long
      - edges: CPU long (El,2) local indices induced from base CSR (CPU)
      - y: CPU tensor
    Save to disk.
    """
    path = _raw_cache_path(split_name)
    print(f"[raw_cache:{split_name}] building (cache miss) -> {path}", flush=True)

    loader = SubGDataset.GDataloader(dataset, batch_size=1, shuffle=False, drop_last=False)
    raw = []
    ys = []

    t0 = time.time()
    for it, batch in enumerate(loader):
        pos, _subG_edge, y = _extract_batch_fields(batch)

        nodes_list = _pad_node_list_to_1d(pos.cpu())
        nodes = nodes_list[0].to(torch.long).cpu() if nodes_list is not None else torch.empty((0,), dtype=torch.long)

        edges = _induce_edges_cpu(nodes)  # (El,2) on CPU

        if args.pin_raw:
            nodes = nodes.pin_memory()
            edges = edges.pin_memory()

        raw.append({"nodes": nodes, "edges": edges})
        ys.append(y.detach().cpu())

        if (it + 1) % 5000 == 0:
            print(f"[raw_cache:{split_name}] {it+1} done", flush=True)

    dt = time.time() - t0
    print(f"[raw_cache:{split_name}] built: {len(raw)} instances ({dt:.2f}s)", flush=True)

    tmp = path + ".tmp"
    torch.save({"raw": raw, "ys": ys}, tmp)
    os.replace(tmp, path)
    print(f"[raw_cache:{split_name}] saved -> {path}", flush=True)
    return (raw, ys)


@torch.no_grad()
def preprocess_all_splits_once():
    """
    ALWAYS:
      - Try load disk cache per split.
      - If missing, build once and save.
    """
    global _PREPROC_RAW
    if _PREPROC_RAW["train"] is not None:
        return

    _ensure_raw_cache_dir()
    print("=" * 60, flush=True)
    print("[raw_cache] ALWAYS-ON: load if exists else build+save", flush=True)
    print("=" * 60, flush=True)

    # train
    got = _try_load_raw_cache("train")
    if got is None:
        got = _build_and_save_raw_cache(trn_dataset, "train")
    _PREPROC_RAW["train"] = got

    # valid
    got = _try_load_raw_cache("valid")
    if got is None:
        got = _build_and_save_raw_cache(val_dataset, "valid")
    _PREPROC_RAW["valid"] = got

    # test
    got = _try_load_raw_cache("test")
    if got is None:
        got = _build_and_save_raw_cache(tst_dataset, "test")
    _PREPROC_RAW["test"] = got


# ----------------------------
# Build BLOCK-DIAGONAL pack per minibatch (GPU path, buildAdj ONCE)
# ----------------------------
@torch.no_grad()
def build_batch_pack_from_instances(raw_list, ys_list, indices):
    """
    Variant 2, segregated copies (cached edges):
      - concat nodes (copy2orig = concatenated base node ids)
      - concat cached local edges (already local indices), shift by offsets
      - buildAdj ONCE with Mtot
    """
    device = config.device
    base_x = baseG.x
    if base_x.ndim == 1:
        base_x = base_x.reshape(-1, 1)

    B = int(len(indices))
    items = [raw_list[i] for i in indices]
    nodes_cpu_list = [it["nodes"] for it in items]
    edges_cpu_list = [it["edges"] for it in items]

    sizes_cpu = torch.tensor([int(n.numel()) for n in nodes_cpu_list], dtype=torch.long)
    Mtot = int(sizes_cpu.sum().item())

    # labels
    y_list = [ys_list[i] for i in indices]
    if isinstance(y_list[0], torch.Tensor) and y_list[0].ndim > 0:
        y_cpu = torch.cat(y_list, dim=0)
    else:
        y_cpu = torch.stack(y_list, dim=0)
    y = y_cpu.to(device=device, non_blocking=True)

    if Mtot == 0:
        z = {
            "local_x0": torch.empty((0, base_x.shape[1]), device=device, dtype=torch.float32),
            "copy2orig": torch.empty((0,), device=device, dtype=torch.long),
            "copy_ptr": torch.zeros((B + 1,), device=device, dtype=torch.long),
            "local_adj_index": torch.empty((2, 0), device=device, dtype=torch.long),
            "local_adj_value": torch.empty((0,), device=device, dtype=torch.float32),
        }
        return z, y

    # concat nodes once -> GPU
    all_nodes = torch.cat(nodes_cpu_list, dim=0).to(device=device, dtype=torch.long, non_blocking=True)
    copy2orig = all_nodes

    # gather local features
    local_x0 = base_x[copy2orig].to(torch.float32)
    if args.preproc_fp16:
        local_x0 = local_x0.to(torch.float16)

    # offsets on CPU
    offsets_cpu = torch.zeros((B,), dtype=torch.long)
    if B > 1:
        offsets_cpu[1:] = torch.cumsum(sizes_cpu[:-1], dim=0)

    # shift & concat edges
    all_ei_parts = []
    for bi in range(B):
        e = edges_cpu_list[bi]  # (El,2) CPU
        if e is None or (not isinstance(e, torch.Tensor)) or e.numel() == 0:
            continue
        off = int(offsets_cpu[bi].item())
        ei = e.t().contiguous()  # (2,El)
        if off != 0:
            ei = ei + off
        all_ei_parts.append(ei)

    if all_ei_parts:
        all_ei_cpu = torch.cat(all_ei_parts, dim=1)  # CPU (2,Eall)
        all_ei = all_ei_cpu.to(device=device, dtype=torch.long, non_blocking=True)
        all_ew = torch.ones((all_ei.shape[1],), device=device, dtype=torch.float32)
        adj = models.buildAdj(all_ei, all_ew, Mtot, aggr="gcn").coalesce()
        local_adj_index = adj.indices()
        local_adj_value = adj.values()
    else:
        local_adj_index = torch.empty((2, 0), device=device, dtype=torch.long)
        local_adj_value = torch.empty((0,), device=device, dtype=torch.float32)

    # copy_ptr on GPU
    sizes = sizes_cpu.to(device=device, non_blocking=True)
    copy_ptr = torch.zeros((B + 1,), device=device, dtype=torch.long)
    copy_ptr[1:] = torch.cumsum(sizes, dim=0)

    z = {
        "local_x0": local_x0,
        "copy2orig": copy2orig,
        "copy_ptr": copy_ptr,
        "local_adj_index": local_adj_index,
        "local_adj_value": local_adj_value,
    }
    return z, y


# ----------------------------
# train/eval
# ----------------------------
def train_epoch_fast(model, optimizer, raw_list, ys_list, batch_size, loss_fn):
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

        z, y = build_batch_pack_from_instances(raw_list, ys_list, idxs)

        optimizer.zero_grad(set_to_none=True)
        out = model(x, edge_index, edge_weight, subG_node=None, z=z)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        nb += 1

    return total_loss / max(1, nb)


@torch.no_grad()
def eval_fast(model, raw_list, ys_list, batch_size, score_fn, loss_fn=None):
    model.eval()

    x = baseG.x
    edge_index = baseG.edge_index
    edge_weight = base_edge_weight_ones()

    outs = []
    ys = []

    n = len(raw_list)
    for s in range(0, n, batch_size):
        idxs = list(range(s, min(n, s + batch_size)))
        z, y = build_batch_pack_from_instances(raw_list, ys_list, idxs)
        out = model(x, edge_index, edge_weight, subG_node=None, z=z)
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
        loss = train_epoch_fast(gnn, optimizer, trn_raw, trn_ys, batch_size, loss_fn)
        trn_time.append(time.time() - t1)
        scd.step(loss)

        if i >= val_start / num_div:
            score, _ = eval_fast(gnn, val_raw, val_ys, batch_size, score_fn, loss_fn=loss_fn)

            if score > val_score:
                early_stop = 0
                val_score = score
                score_tst, _ = eval_fast(gnn, tst_raw, tst_ys, batch_size, score_fn, loss_fn=loss_fn)
                tst_score = max(score_tst, tst_score)
                print(f"iter {i} loss {loss:.4f} val {val_score:.4f} tst {tst_score:.4f}", flush=True)

            elif score >= val_score - 1e-5:
                score_tst, _ = eval_fast(gnn, tst_raw, tst_ys, batch_size, score_fn, loss_fn=loss_fn)
                tst_score = max(score_tst, tst_score)
                if i % int(args.log_every) == 0:
                    print(f"iter {i} loss {loss:.4f} val {score:.4f} tst {score_tst:.4f}", flush=True)

            else:
                early_stop += 1
                if i % int(args.log_every) == 0:
                    tst_curr, _ = eval_fast(gnn, tst_raw, tst_ys, batch_size, score_fn, loss_fn=loss_fn)
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

    print(f"\n[NODE-INDUCED] Running Final Evaluation on Test Set ({args.repeat} repeats)...", flush=True)
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
    print(f"[NODE-INDUCED] Final Results over {args.repeat} runs:", flush=True)
    print(f"Average Test Score: {mean:.3f} error {err:.3f}", flush=True)
    print("=" * 40, flush=True)

    out = {
        "dataset": args.dataset,
        "model": "EdgeModel_NODE_INDUCED",
        "params": params,
        "test_scores": tst_scores.tolist(),
        "mean": mean,
        "error": err,
        "max_epoch": int(args.max_epoch),
        "val_start": int(args.val_start),
        "obj_repeats": int(args.obj_repeats),
        "preproc_fp16": bool(args.preproc_fp16),
        "pin_raw": bool(args.pin_raw),
        "dedup_local_ei": bool(args.dedup_local_ei),
        "base_is_bidir": bool(_BASE_IS_BIDIR),
        "raw_cache_dir": _RAW_CACHE_DIR,
    }
    with open(f"{args.dataset}_edge_model_node_induced_results.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
