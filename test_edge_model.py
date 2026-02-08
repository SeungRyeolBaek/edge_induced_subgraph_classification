# test_edge_model.py
# Train/Eval EdgeModel on edge-induced subgraph dataset (jsonl-based loaders)
# - base graph MP + segregated copy graph MP
# - layer-wise feature mixing with alpha in (0.5, 1]
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

    # config
    p.add_argument("--config_dir", type=str, default="config/edge/EdgeModel")

    # (optional) memory safety: store preproc local_x0 as fp16
    p.add_argument("--preproc_fp16", action="store_true")

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

    # if nodeid: overwrite x with cached embedding (fixed hidden_dim)
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

    # reset base ew cache
    _BASE_EW = None


# ----------------------------
# build model
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

_PREPROC = {
    "train": None,
    "valid": None,
    "test": None,
}


@torch.no_grad()
def _extract_batch_fields(batch):
    """
    GDataloader 배치 포맷이 가끔 달라서 기존 try/except 그대로 유지.
    """
    try:
        _, _, _, pos, subG_edge, y = batch
    except ValueError:
        _, _, _, pos, subG_edge, _, y = batch
    return pos, subG_edge, y


@torch.no_grad()
def preprocess_split_to_memory(dataset, split_name: str):
    """
    split 전체를 한 번 훑으면서:
      - 각 subgraph instance마다 _build_segregated_batch를 "단일 subgraph"로 실행
      - local adj normalize까지 끝낸 pack을 CPU로 저장
    """
    loader = SubGDataset.GDataloader(dataset, batch_size=1, shuffle=False, drop_last=False)
    packs = []
    ys = []

    base_x = baseG.x
    if base_x.ndim == 1:
        base_x = base_x.reshape(-1, 1)

    t0 = time.time()
    for it, batch in enumerate(loader):
        pos, subG_edge, y = _extract_batch_fields(batch)

        # pos: (1,maxN) or weird -> models._build_segregated_batch 내부에서 처리
        subG_node = pos.to(config.device)
        subG_edge = subG_edge.to(config.device) if subG_edge is not None else None

        pack0 = models._build_segregated_batch(subG_node=subG_node, subG_edge=subG_edge, base_x=base_x)

        local_x0 = pack0["local_x"]          # (Mi,F) on device
        local_ei = pack0["local_ei"]         # (2,Eraw) on device
        local_ew = pack0["local_ew"]         # (Eraw,) ones on device
        copy2orig = pack0["copy2orig"]       # (Mi,) on device
        copy_ptr_full = pack0["copy_ptr"]    # (B+1,) but here B should be 1

        Mi = int(local_x0.shape[0])
        if Mi > 0 and local_ei.numel() > 0:
            adj = models.buildAdj(local_ei, local_ew, Mi, aggr="gcn").coalesce()
            local_adj_index = adj.indices()
            local_adj_value = adj.values()
        else:
            local_adj_index = torch.empty((2, 0), device=config.device, dtype=torch.long)
            local_adj_value = torch.empty((0,), device=config.device, dtype=torch.float32)

        # per-instance ptr은 (2,)로 통일: [0, Mi]
        copy_ptr = torch.tensor([0, Mi], device=config.device, dtype=torch.long)

        # store CPU
        x_cpu = local_x0.detach().cpu()
        if args.preproc_fp16:
            x_cpu = x_cpu.to(torch.float16)

        pack_cpu = {
            "local_x0": x_cpu,  # (Mi,F)
            "copy2orig": copy2orig.detach().cpu().to(torch.long),
            "copy_ptr": copy_ptr.detach().cpu().to(torch.long),  # (2,)
            "local_adj_index": local_adj_index.detach().cpu().to(torch.long),
            "local_adj_value": local_adj_value.detach().cpu().to(torch.float32),
        }
        packs.append(pack_cpu)
        ys.append(y.detach().cpu())

        if (it + 1) % 5000 == 0:
            print(f"[preproc:{split_name}] {it+1} done", flush=True)

    dt = time.time() - t0
    print(f"[preproc:{split_name}] done: {len(packs)} instances, time {dt:.2f}s", flush=True)
    return packs, ys


@torch.no_grad()
def preprocess_all_splits_once():
    global _PREPROC
    if _PREPROC["train"] is not None:
        return

    print("=" * 60, flush=True)
    print("[preproc] building per-subgraph packs ONCE into memory ...", flush=True)
    print("=" * 60, flush=True)

    _PREPROC["train"] = preprocess_split_to_memory(trn_dataset, "train")
    _PREPROC["valid"] = preprocess_split_to_memory(val_dataset, "valid")
    _PREPROC["test"] = preprocess_split_to_memory(tst_dataset, "test")


def _to_device_pack(pack_cpu):
    # local_x0 might be fp16 on cpu -> keep fp16 on gpu (fine) or cast to fp32 if you want
    out = {}
    for k, v in pack_cpu.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(config.device, non_blocking=True)
        else:
            out[k] = v
    return out


@torch.no_grad()
def build_batch_from_instances(packs_cpu, ys_cpu, indices):
    """
    indices: list[int] for instances in this batch
    returns:
      pack (dict on device) for EdgeModel forward
      y (tensor on device)  (B,*) labels
    Batch pack 구성:
      local_x0 = concat
      local_adj_index/value: per-instance index에 offset을 더해 concat (block-diagonal)
      copy2orig = concat (base node id 그대로)
      copy_ptr: (B+1,) [0, M1, M1+M2, ...]
    """
    # gather
    xs = []
    c2o = []
    adj_i = []
    adj_v = []
    sizes = []

    off = 0
    for idx in indices:
        p = packs_cpu[idx]
        x0 = p["local_x0"]          # (Mi,F) CPU
        mi = int(x0.shape[0])
        sizes.append(mi)

        xs.append(x0)
        c2o.append(p["copy2orig"])  # (Mi,)
        # adj with offset
        ai = p["local_adj_index"]   # (2,Ei)
        av = p["local_adj_value"]   # (Ei,)
        if ai.numel() > 0:
            ai_off = ai.clone()
            ai_off[0] += off
            ai_off[1] += off
            adj_i.append(ai_off)
            adj_v.append(av)
        off += mi

    M = int(sum(sizes))
    B = int(len(indices))

    if M == 0:
        # empty batch (shouldn't happen normally)
        pack = {
            "local_x0": torch.empty((0, baseG.x.shape[1]), device=config.device, dtype=torch.float32),
            "copy2orig": torch.empty((0,), device=config.device, dtype=torch.long),
            "copy_ptr": torch.zeros((B + 1,), device=config.device, dtype=torch.long),
            "local_adj_index": torch.empty((2, 0), device=config.device, dtype=torch.long),
            "local_adj_value": torch.empty((0,), device=config.device, dtype=torch.float32),
        }
    else:
        # concat on CPU then move once
        x_cpu = torch.cat(xs, dim=0) if xs else torch.empty((0, baseG.x.shape[1]))
        c2o_cpu = torch.cat(c2o, dim=0) if c2o else torch.empty((0,), dtype=torch.long)

        if adj_i:
            ai_cpu = torch.cat(adj_i, dim=1)
            av_cpu = torch.cat(adj_v, dim=0)
        else:
            ai_cpu = torch.empty((2, 0), dtype=torch.long)
            av_cpu = torch.empty((0,), dtype=torch.float32)

        ptr = torch.zeros((B + 1,), dtype=torch.long)
        if B > 0:
            ptr[1:] = torch.cumsum(torch.tensor(sizes, dtype=torch.long), dim=0)

        pack_cpu = {
            "local_x0": x_cpu,
            "copy2orig": c2o_cpu,
            "copy_ptr": ptr,
            "local_adj_index": ai_cpu,
            "local_adj_value": av_cpu,
        }
        pack = _to_device_pack(pack_cpu)

    y_list = [ys_cpu[i] for i in indices]
    y_cpu = torch.cat(y_list, dim=0) if isinstance(y_list[0], torch.Tensor) and y_list[0].ndim > 0 else torch.stack(y_list, dim=0)
    y = y_cpu.to(config.device, non_blocking=True)

    return pack, y


# ----------------------------
# train/eval using preproc instances
# ----------------------------
def train_epoch_preproc(model, optimizer, packs_cpu, ys_cpu, batch_size, loss_fn):
    model.train()
    n = len(packs_cpu)
    order = list(range(n))
    random.shuffle(order)

    x = baseG.x
    edge_index = baseG.edge_index
    edge_weight = base_edge_weight_ones()

    total_loss = 0.0
    nb = 0

    for s in range(0, n, batch_size):
        idxs = order[s:s + batch_size]
        if len(idxs) == 0:
            continue

        pack, y = build_batch_from_instances(packs_cpu, ys_cpu, idxs)

        optimizer.zero_grad(set_to_none=True)
        out = model(x, edge_index, edge_weight, subG_node=None, z=pack)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        nb += 1

    return total_loss / max(1, nb)


@torch.no_grad()
def eval_preproc(model, packs_cpu, ys_cpu, batch_size, score_fn, loss_fn=None):
    model.eval()

    x = baseG.x
    edge_index = baseG.edge_index
    edge_weight = base_edge_weight_ones()

    outs = []
    ys = []

    n = len(packs_cpu)
    for s in range(0, n, batch_size):
        idxs = list(range(s, min(n, s + batch_size)))
        pack, y = build_batch_from_instances(packs_cpu, ys_cpu, idxs)
        out = model(x, edge_index, edge_weight, subG_node=None, z=pack)
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

    (trn_packs, trn_ys) = _PREPROC["train"]
    (val_packs, val_ys) = _PREPROC["valid"]
    (tst_packs, tst_ys) = _PREPROC["test"]

    optimizer = Adam(gnn.parameters(), lr=lr)
    scd = lr_scheduler.ReduceLROnPlateau(optimizer, factor=resi, min_lr=5e-5)

    val_score = 0.0
    tst_score = 0.0
    early_stop = 0
    trn_time = []

    max_epoch = int(args.max_epoch if max_epoch is None else max_epoch)
    val_start = int(args.val_start)

    num_div = max(1.0, float(max(1, len(tst_packs) // max(1, batch_size))))

    for i in range(max_epoch):
        t1 = time.time()
        loss = train_epoch_preproc(gnn, optimizer, trn_packs, trn_ys, batch_size, loss_fn)
        trn_time.append(time.time() - t1)
        scd.step(loss)

        if i >= val_start / num_div:
            score, _ = eval_preproc(gnn, val_packs, val_ys, batch_size, score_fn, loss_fn=loss_fn)

            if score > val_score:
                early_stop = 0
                val_score = score
                score_tst, _ = eval_preproc(gnn, tst_packs, tst_ys, batch_size, score_fn, loss_fn=loss_fn)
                tst_score = max(score_tst, tst_score)
                print(f"iter {i} loss {loss:.4f} val {val_score:.4f} tst {tst_score:.4f}", flush=True)

            elif score >= val_score - 1e-5:
                score_tst, _ = eval_preproc(gnn, tst_packs, tst_ys, batch_size, score_fn, loss_fn=loss_fn)
                tst_score = max(score_tst, tst_score)
                if i % int(args.log_every) == 0:
                    print(f"iter {i} loss {loss:.4f} val {score:.4f} tst {score_tst:.4f}", flush=True)

            else:
                early_stop += 1
                if i % int(args.log_every) == 0:
                    tst_curr, _ = eval_preproc(gnn, tst_packs, tst_ys, batch_size, score_fn, loss_fn=loss_fn)
                    print(f"iter {i} loss {loss:.4f} val {score:.4f} tst {tst_curr:.4f}", flush=True)

        if val_score >= 1 - 1e-5:
            early_stop += 1
        if early_stop > 100 / num_div:
            break

    # AUROC는 학습 도중 절대 출력/계산하지 않고, 여기서 딱 1번만 계산
    tst_auroc, _ = eval_preproc(gnn, tst_packs, tst_ys, batch_size, metrics.auroc, loss_fn=None)

    print(
        f"end: epoch {i+1}, train time {sum(trn_time):.2f} s, val {val_score:.3f}, tst {tst_score:.3f}, auroc {tst_auroc:.3f}",
        flush=True,
    )
    return float(val_score), float(tst_score), float(tst_auroc)


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
            val_score, tst_score, tst_auroc = train_and_eval(params, seed=seed, max_epoch=min(300, int(args.max_epoch)))
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
# main
# ----------------------------
def main():
    if args.use_seed:
        set_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 1) dataset + base features/embeddings: ONCE
    split_and_features()

    # 2) preprocess per-subgraph packs: ONCE (in-memory)
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
        print(f"Best score (mean val over {args.obj_repeats} seeds): {study.best_value:.4f}", flush=True)
        print(f"Saved best config to {cfg_path}", flush=True)
        params = best_params
    else:
        print(f"Loading config from {cfg_path}", flush=True)
        with open(cfg_path) as f:
            params = yaml.safe_load(f)
        params["hidden_dim"] = int(params.get("hidden_dim", args.hidden_dim))
        if "alpha" not in params:
            params["alpha"] = 0.8

    print(f"\nRunning Final Evaluation on Test Set ({args.repeat} repeats)...", flush=True)
    print("Params:", params, flush=True)

    tst_scores = []
    tst_aurocs = []
    for r in range(int(args.repeat)):
        print(f"\n--- Repeat {r}/{args.repeat} ---", flush=True)
        seed = repeat_seed(r) 
        _, tst, auc = train_and_eval(params, seed=seed, max_epoch=int(args.max_epoch))
        tst_scores.append(tst)
        tst_aurocs.append(auc)

    tst_scores = np.array(tst_scores, dtype=float)
    tst_aurocs = np.array(tst_aurocs, dtype=float)

    mean = float(np.mean(tst_scores))
    err = float(np.std(tst_scores) / np.sqrt(len(tst_scores)))

    auc_mean = float(np.mean(tst_aurocs))
    auc_err = float(np.std(tst_aurocs) / np.sqrt(len(tst_aurocs)))

    print("\n" + "=" * 40, flush=True)
    print(f"Final Results over {args.repeat} runs:", flush=True)
    print(f"Average Test Score: {mean:.3f} error {err:.3f}", flush=True)
    print(f"Average Test AUROC: {auc_mean:.3f} error {auc_err:.3f}", flush=True)
    print("=" * 40, flush=True)

    out = {
        "dataset": args.dataset,
        "model": "EdgeModel",
        "params": params,
        "test_scores": tst_scores.tolist(),
        "mean": mean,
        "error": err,
        "test_aurocs": tst_aurocs.tolist(),
        "auroc_mean": auc_mean,
        "auroc_error": auc_err,
        "max_epoch": int(args.max_epoch),
        "val_start": int(args.val_start),
        "obj_repeats": int(args.obj_repeats),
        "preproc_fp16": bool(args.preproc_fp16),
    }
    with open(f"{args.dataset}_edge_model_results.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
