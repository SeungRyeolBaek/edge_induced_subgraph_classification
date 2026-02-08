# GNNseg.py
# - edge_subgraph.jsonl loader
# - Optuna HPO (val micro-f1 / binaryf1)
# - Final repeats (test score + test AUROC)
# - AUROC is computed via impl.metrics.auroc for binary/multiclass/multilabel
# - Keeps the same CLI flags you are already using

import argparse
import functools
import json
import os
import random
import time
import copy

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import Adam, lr_scheduler
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader as pygDataloader
from torch_geometric.nn import GCNConv, GraphNorm, global_add_pool, global_mean_pool

# use impl.metrics (numpy-based f1/auroc)
from impl import metrics


class config:
    device = torch.device("cpu")

    @staticmethod
    def set_device(device_idx: int):
        if torch.cuda.is_available():
            config.device = torch.device(f"cuda:{device_idx}")
        else:
            config.device = torch.device("cpu")


class GsDataset(InMemoryDataset):
    """
    Each element is a PyG Data graph (one edge-induced subgraph instance).
    Stored on CPU; batches are moved to GPU in the train loop.
    """
    def __init__(self, datalist):
        self.datalist = datalist
        super().__init__()
        self.data, self.slices = self.collate(self.datalist)

    def __len__(self):
        return len(self.datalist)


class GConv(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        num_layers: int,
        dropout=0.0,
        activation=nn.ELU(inplace=True),
        conv=GCNConv,
        **kwargs,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.gns = nn.ModuleList()

        if num_layers > 1:
            self.convs.append(conv(in_channels=input_channels, out_channels=hidden_channels, **kwargs))
            for _ in range(num_layers - 2):
                self.convs.append(conv(in_channels=hidden_channels, out_channels=hidden_channels, **kwargs))
            self.convs.append(conv(in_channels=hidden_channels, out_channels=output_channels, **kwargs))
            self.gns = nn.ModuleList([GraphNorm(hidden_channels) for _ in range(num_layers - 1)])
        else:
            self.convs.append(conv(in_channels=input_channels, out_channels=output_channels, **kwargs))
            self.gns = nn.ModuleList()

        self.activation = activation
        self.dropout = float(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()
        for gn in self.gns:
            gn.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        xs = []
        if len(self.convs) == 1:
            out = self.convs[0](x, edge_index, edge_weight)
            xs.append(out)
            return torch.cat(xs, dim=-1)

        for layer, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = self.gns[layer](x)
            xs.append(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        xs.append(self.convs[-1](x, edge_index, edge_weight))
        return torch.cat(xs, dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout=0.0, activation=nn.ELU(inplace=True)):
        super().__init__()
        assert num_layers >= 1
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation)
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GNN(nn.Module):
    """
    Proper batched graph classification:
    - node embeddings -> global pooling by batch vector -> graph embedding -> predictor
    """
    def __init__(self, conv: nn.Module, pred: nn.Module, aggr="sum"):
        super().__init__()
        self.conv = conv
        self.pred = pred
        self.aggr = aggr

    def forward(self, x, edge_index, edge_weight, batch_vec):
        emb = self.conv(x, edge_index, edge_weight)  # [N, D]
        if self.aggr == "sum":
            g = global_add_pool(emb, batch_vec)  # [B, D]
        elif self.aggr == "mean":
            g = global_mean_pool(emb, batch_vec)
        else:
            raise ValueError(f"unknown aggr: {self.aggr}")
        out = self.pred(g)  # [B, C] or [B, 1]
        return out


def read_jsonl(path: str):
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def edges_to_data(edge_list, label, use_degree=True, use_one=False, undirected=True):
    if len(edge_list) == 0:
        x = torch.ones((1, 1), dtype=torch.float)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0,), dtype=torch.float)
        y = torch.tensor([int(label)], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    nodes = sorted({int(u) for e in edge_list for u in e})
    node2idx = {nid: i for i, nid in enumerate(nodes)}

    src, dst = [], []
    for u, v in edge_list:
        u = node2idx[int(u)]
        v = node2idx[int(v)]
        src.append(u)
        dst.append(v)
        if undirected:
            src.append(v)
            dst.append(u)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    n = len(nodes)

    feats = []
    if use_degree:
        deg = torch.zeros(n, dtype=torch.float)
        for s in src:
            deg[s] += 1.0
        feats.append(deg.view(-1, 1))

    if use_one or (len(feats) == 0):
        feats.append(torch.ones((n, 1), dtype=torch.float))

    x = torch.cat(feats, dim=-1)
    edge_attr = torch.ones(edge_index.size(1), dtype=torch.float)  # used as edge_weight
    y = torch.tensor([int(label)], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def load_edge_subgraph_dataset(dataset_name: str, root="edge_dataset", use_degree=True, use_one=False, undirected=True):
    path = os.path.join(root, dataset_name, "edge_subgraph.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing: {path}")

    recs = read_jsonl(path)

    trn, val, tst = [], [], []
    max_label = -1
    for r in recs:
        label = int(r["label"])
        split = r["split"]
        edge_list = r["graph"]
        max_label = max(max_label, label)

        d = edges_to_data(edge_list, label, use_degree=use_degree, use_one=use_one, undirected=undirected)

        if split == "train":
            trn.append(d)
        elif split == "valid":
            val.append(d)
        elif split == "test":
            tst.append(d)
        else:
            raise ValueError(f"unknown split: {split}")

    num_classes = max_label + 1
    return trn, val, tst, num_classes


def buildModel(input_channels, output_channels, hidden_dim, conv_layer, dropout, aggr="sum"):
    tmp2 = hidden_dim * conv_layer
    conv_impl = functools.partial(GCNConv, add_self_loops=False)

    conv = GConv(
        input_channels=input_channels,
        hidden_channels=hidden_dim,
        output_channels=hidden_dim,
        num_layers=conv_layer,
        conv=conv_impl,
        activation=nn.ELU(inplace=True),
        dropout=dropout,
    )

    mlp = MLP(
        tmp2,
        hidden_dim,
        output_channels,
        2,
        dropout=dropout,
        activation=nn.ELU(inplace=True),
    )

    gnn = GNN(conv, mlp, aggr=aggr).to(config.device)
    return gnn


def train_epoch(optimizer, model, loader, loss_fn):
    model.train()
    total_loss, steps = 0.0, 0

    for batch in loader:
        batch = batch.to(config.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        steps += 1

    return total_loss / max(1, steps)


def _to_numpy_logits_and_label(out_t: torch.Tensor, y_t: torch.Tensor):
    out = out_t.detach().cpu().numpy()
    y = y_t.view(-1).detach().cpu().numpy()
    return out, y


@torch.no_grad()
def test_epoch(model, loader, score_fn_np, loss_fn=None):
    """
    score_fn_np: function(pred_np, label_np) -> float (from impl.metrics)
    """
    model.eval()
    outs, ys = [], []
    total_loss, steps = 0.0, 0

    for batch in loader:
        batch = batch.to(config.device, non_blocking=True)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        outs.append(out)
        ys.append(batch.y)

        if loss_fn is not None:
            total_loss += float(loss_fn(out, batch.y).item())
            steps += 1

    out_t = torch.cat(outs, dim=0) if outs else torch.empty((0,), device=config.device)
    y_t = torch.cat(ys, dim=0) if ys else torch.empty((0,), device=config.device)

    if y_t.numel() > 0:
        out_np, y_np = _to_numpy_logits_and_label(out_t, y_t)
        try:
            score = float(score_fn_np(out_np, y_np))
        except ValueError as e:
            # AUROC can be undefined if only one class exists in y_true, etc.
            print(f"[metric-error] {score_fn_np.__name__}: {e}", flush=True)
            score = float("nan")
    else:
        score = 0.0

    loss = (total_loss / max(1, steps)) if loss_fn is not None else None
    return score, loss


def run_once(
    trn_dataset,
    val_dataset,
    tst_dataset,
    input_channels,
    output_channels,
    loss_fn,
    score_fn_np,
    hidden_dim,
    conv_layer,
    dropout,
    lr,
    batch_size,
    max_epochs=500,
    num_workers=0,
    aggr="sum",
):
    pin = (config.device.type == "cuda")

    trn_loader = pygDataloader(
        trn_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=pin
    )
    val_loader = pygDataloader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin
    )
    tst_loader = pygDataloader(
        tst_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=pin
    )

    gnn = buildModel(input_channels, output_channels, hidden_dim, conv_layer, dropout, aggr=aggr)
    optimizer = Adam(gnn.parameters(), lr=lr)
    scd = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, min_lr=5e-5)

    val_best = 0.0
    early_stop = 0.0

    # keep best-val checkpoint so AUROC/test are aligned with selection
    best_state = None

    for i in range(max_epochs):
        loss = train_epoch(optimizer, gnn, trn_loader, loss_fn)
        scd.step(loss)

        if i % 5 == 0:
            score, _ = test_epoch(gnn, val_loader, score_fn_np, loss_fn=loss_fn)
            early_stop += 1.0

            if score > val_best:
                val_best = score
                best_state = copy.deepcopy(gnn.state_dict())

                score_t, _ = test_epoch(gnn, tst_loader, score_fn_np, loss_fn=loss_fn)
                print(f"iter {i} loss {loss:.4f} val {val_best:.4f} tst {score_t:.4f}", flush=True)
                early_stop /= 2.0
            elif score >= val_best - 1e-5:
                score_t, _ = test_epoch(gnn, tst_loader, score_fn_np, loss_fn=loss_fn)
                print(f"iter {i} loss {loss:.4f} val {val_best:.4f} tst {score_t:.4f}", flush=True)
                early_stop /= 2.0
            else:
                score_t, _ = test_epoch(gnn, tst_loader, score_fn_np, loss_fn=loss_fn)
                print(f"iter {i} loss {loss:.4f} val {score:.4f} tst {score_t:.4f}", flush=True)

            if early_stop > 10:
                break

    # load best-val checkpoint, then compute final test score + auroc once
    if best_state is not None:
        gnn.load_state_dict(best_state)

    tst_best, _ = test_epoch(gnn, tst_loader, score_fn_np, loss_fn=loss_fn)

    # AUROC: always computed via impl.metrics.auroc (binary/multiclass/multilabel)
    tst_auroc, _ = test_epoch(gnn, tst_loader, metrics.auroc, loss_fn=loss_fn)

    return val_best, tst_best, tst_auroc


def set_seed(seed: int):
    print("seed =", seed, flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def objective(
    trial,
    trn_dataset,
    val_dataset,
    tst_dataset,
    input_channels,
    output_channels,
    loss_fn,
    score_fn_np,
    batch_size,
    num_workers,
    aggr,
):
    hidden_dim = trial.suggest_categorical("hidden_dim", [4, 8, 16, 32, 64, 128])
    conv_layer = trial.suggest_int("conv_layer", 1, 10)
    dropout = trial.suggest_float("dropout", 0.0, 0.6)
    lr = trial.suggest_float("lr", 5e-4, 5e-3, log=True)

    set_seed(0)
    val_best, _, _ = run_once(
        trn_dataset,
        val_dataset,
        tst_dataset,
        input_channels,
        output_channels,
        loss_fn,
        score_fn_np,
        hidden_dim,
        conv_layer,
        dropout,
        lr,
        batch_size,
        max_epochs=500,
        num_workers=num_workers,
        aggr=aggr,
    )
    return float(val_best)


def _cfg_dir():
    return os.path.join("config", "GNNseg")


def _cfg_path(dataset_name: str):
    return os.path.join(_cfg_dir(), f"{dataset_name}.yml")


def load_config_or_none(dataset_name: str):
    path = _cfg_path(dataset_name)
    if not os.path.exists(path):
        return None, path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f), path


def save_config(dataset_name: str, params: dict):
    os.makedirs(_cfg_dir(), exist_ok=True)
    path = _cfg_path(dataset_name)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(params, f)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNNseg baseline (edge_subgraph.jsonl) + Optuna (FAST global pooling)")
    parser.add_argument("--dataset", type=str, default="DocRED")  # DocRED / VisualGenome / Connectome
    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--optruns", type=int, default=100)

    parser.add_argument("--batch_size", type=int, default=160)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--use_degree", action="store_true")
    parser.add_argument("--use_one", action="store_true")
    parser.add_argument("--directed", action="store_true")

    # pooling
    parser.add_argument("--aggr", type=str, default="sum", choices=["sum", "mean"])

    args = parser.parse_args()

    if (not args.use_degree) and (not args.use_one):
        args.use_degree = True

    config.set_device(args.device)
    undirected = (not args.directed)

    print(args, flush=True)

    trn_list, val_list, tst_list, num_classes = load_edge_subgraph_dataset(
        dataset_name=args.dataset,
        root="edge_dataset",
        use_degree=args.use_degree,
        use_one=args.use_one,
        undirected=undirected,
    )

    trn_dataset = GsDataset(trn_list)
    val_dataset = GsDataset(val_list)
    tst_dataset = GsDataset(tst_list)

    input_channels = trn_list[0].x.size(-1) if len(trn_list) > 0 else 1

    # score_fn is numpy-based from impl.metrics
    if num_classes == 2:
        def loss_fn(x, y):
            return BCEWithLogitsLoss()(x.flatten(), y.flatten().to(torch.float))

        output_channels = 1
        score_fn_np = metrics.binaryf1
    else:
        loss_ce = CrossEntropyLoss()

        def loss_fn(x, y):
            return loss_ce(x, y.view(-1))

        output_channels = num_classes
        score_fn_np = metrics.microf1

    params, cfg_path = load_config_or_none(args.dataset)

    if params is None:
        print("=" * 40, flush=True)
        print(f"Config not found. Starting Optuna Optimization ({args.optruns} trials)...", flush=True)
        print("=" * 40, flush=True)

        print(f"[GNNseg] Starting Optuna ({args.optruns} trials)...", flush=True)
        study = optuna.create_study(direction="maximize")
        t0 = time.time()

        study.optimize(
            lambda trial: objective(
                trial,
                trn_dataset,
                val_dataset,
                tst_dataset,
                input_channels,
                output_channels,
                loss_fn,
                score_fn_np,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                aggr=args.aggr,
            ),
            n_trials=args.optruns,
        )

        print(f"[GNNseg] Optuna done. elapsed={time.time() - t0:.1f}s", flush=True)
        print("[GNNseg] best_value", study.best_value, flush=True)
        print("[GNNseg] best_params", study.best_params, flush=True)

        best = study.best_params
        params = {
            "hidden_dim": int(best["hidden_dim"]),
            "conv_layer": int(best["conv_layer"]),
            "dropout": float(best["dropout"]),
            "lr": float(best["lr"]),
        }

        cfg_path = save_config(args.dataset, params)
        print(f"Saved best config to {cfg_path}", flush=True)
    else:
        print(f"Loading config from {cfg_path}", flush=True)

    print(f"\nRunning Final Evaluation on Test Set ({args.repeat} repeats)...", flush=True)
    print("Params:", params, flush=True)

    outs = []
    aurocs = []

    for r in range(args.repeat):
        print(f"\n--- Repeat {r}/{args.repeat} ---", flush=True)
        set_seed((1 << r) - 1)

        val_best, tst_best, tst_auroc = run_once(
            trn_dataset,
            val_dataset,
            tst_dataset,
            input_channels,
            output_channels,
            loss_fn,
            score_fn_np,
            hidden_dim=int(params["hidden_dim"]),
            conv_layer=int(params["conv_layer"]),
            dropout=float(params["dropout"]),
            lr=float(params["lr"]),
            batch_size=args.batch_size,
            max_epochs=500,
            num_workers=args.num_workers,
            aggr=args.aggr,
        )

        print(f"end: val {val_best:.4f} tst {tst_best:.4f} auroc {tst_auroc:.4f}", flush=True)
        outs.append(tst_best)
        aurocs.append(tst_auroc)

    outs = np.array(outs, dtype=np.float64)
    aurocs = np.array(aurocs, dtype=np.float64)

    print("\n" + "=" * 40, flush=True)
    print(f"Final Results over {args.repeat} runs:", flush=True)
    print(
        f"Average Test Score: {float(outs.mean()):.3f} error {float(outs.std() / np.sqrt(max(1, len(outs)))):.3f}",
        flush=True,
    )

    # AUROC summary (nan can still appear if AUROC is undefined for that split; we don't crash)
    if np.isfinite(aurocs).all():
        print(
            f"Average Test AUROC: {float(aurocs.mean()):.3f} error {float(aurocs.std() / np.sqrt(max(1, len(aurocs)))):.3f}",
            flush=True,
        )
    else:
        au = aurocs[np.isfinite(aurocs)]
        print(
            f"Average Test AUROC: {float(au.mean()):.3f} error {float(au.std() / np.sqrt(max(1, len(au)))):.3f} (nan excluded)",
            flush=True,
        )
    print("=" * 40, flush=True)
