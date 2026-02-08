import argparse
import functools
import itertools
import json
import os
import os.path
import random
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
import optuna
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam, lr_scheduler
from torch_geometric.nn import MLP, GCNConv
from torch_sparse import SparseTensor

from impl import models_hybrid, train_hybrid, metrics, utils, config
import warnings
from edge_dataset import SubGDataset_hybrid, node_datasets as datasets

from impl.models_hybrid import COMGraphConv

warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', RuntimeWarning)

trn_dataset, val_dataset, tst_dataset = None, None, None
max_deg, output_channels = 0, 1
score_fn = None
loss_fn = None
row, col = None, None

# =========================
# GLOBAL CACHES (NO RE-READ)
# =========================
baseG = None
_SPLITS = None          # {"train": tuple, "valid": tuple, "test": tuple}
_NODE_EMB = None        # torch.Tensor on CPU (num_nodes x 64)
_TORCH_CLUSTER_SHIM_SET = False


def set_seed(seed: int):
    # PRINT FORMAT ONLY (GLASS style)
    print("seed ", seed, flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi gpu


def random_walk_fallback(rowptr, colidx, start, walk_length):
    start = start.to(torch.long)
    device = start.device
    num_walks = start.numel()

    walks = torch.empty((num_walks, walk_length + 1), device=device, dtype=torch.long)
    walks[:, 0] = start

    cur = start
    for t in range(1, walk_length + 1):
        deg = rowptr[cur + 1] - rowptr[cur]
        nxt = cur.clone()

        mask = deg > 0
        if mask.any():
            cur_m = cur[mask]
            deg_m = deg[mask]

            max_deg_local = int(deg_m.max().item())
            off = torch.randint(
                low=0,
                high=max_deg_local,
                size=(cur_m.numel(),),
                device=device,
                dtype=torch.long
            )
            off = off % deg_m
            edge_idx = rowptr[cur_m] + off
            nxt[mask] = colidx[edge_idx]

        walks[:, t] = nxt
        cur = nxt

    return walks


def _emb_file(args):
    if args.dataset == "DocRED":
        return os.path.join(args.path, "docred_64.pt")
    if args.dataset == "VisualGenome":
        return os.path.join(args.path, "vg_64.pt")
    if args.dataset == "Connectome":
        return os.path.join(args.path, "connectome_64.pt")
    raise RuntimeError(f"Use exactly: DocRED / VisualGenome / Connectome")


def _fix_empty_comp_inplace(ds):
    if not hasattr(ds, "comp"):
        return
    comp = getattr(ds, "comp")
    if torch.is_tensor(comp) and comp.ndim == 2 and comp.numel() > 0:
        mask = (comp == -1).all(dim=1)
        if mask.any():
            comp[mask, 0] = 0
            setattr(ds, "comp", comp)


def init_cache(args):
    """
    딱 1번만:
    - dataset load
    - node feature 세팅
    - CSR(row/col) 생성 + GPU로 이동
    - split tuple 캐싱
    - embedding pt 파일 1번만 load (use_nodeid일 때)
    - torch_cluster.random_walk shim 1번만 세팅
    """
    global baseG, _SPLITS, _NODE_EMB, _TORCH_CLUSTER_SHIM_SET
    global max_deg, row, col

    baseG = datasets.load_dataset(args.dataset)

    # label dtype (binary vs multi-class)
    if baseG.y.unique().shape[0] == 2:
        baseG.y = baseG.y.to(torch.float)
    else:
        baseG.y = baseG.y.to(torch.int64)

    # initialize node features (딱 1번만)
    if args.use_deg:
        baseG.setDegreeFeature()
    elif args.use_one:
        baseG.setOneFeature()
    elif args.use_nodeid:
        baseG.setNodeIdFeature()
    else:
        raise NotImplementedError

    max_deg = torch.max(baseG.x)

    # CSR 딱 1번만
    N = baseG.x.shape[0]
    E = baseG.edge_index.size()[-1]
    sparse_adj = SparseTensor(
        row=baseG.edge_index[0], col=baseG.edge_index[1],
        value=torch.arange(E, device="cpu"),
        sparse_sizes=(N, N)
    )
    row, col, _ = sparse_adj.csr()

    # move CSR to GPU once
    row = row.to(config.device)
    col = col.to(config.device)

    baseG.to(config.device)

    # cache split tuples once
    _SPLITS = {
        "train": baseG.get_split("train"),
        "valid": baseG.get_split("valid"),
        "test": baseG.get_split("test"),
    }

    # load embedding ONCE
    if args.use_nodeid:
        emb_path = _emb_file(args)
        print("load ", emb_path, flush=True)
        emb = torch.load(emb_path, map_location=torch.device('cpu')).detach()
        if (not torch.is_tensor(emb)) or emb.ndim != 2:
            raise RuntimeError(f"[SSNP] Bad embedding tensor at {emb_path}, got type={type(emb)}")
        if int(emb.shape[1]) != 64:
            raise RuntimeError(f"[SSNP] Emb dim must be 64. {emb_path} has shape {tuple(emb.shape)}")
        _NODE_EMB = emb.contiguous()  # keep on CPU

    # set torch_cluster shim ONCE
    if not _TORCH_CLUSTER_SHIM_SET:
        class _TorchClusterShim:
            @staticmethod
            def random_walk(rowptr_, colidx_, start_, walk_length_, p_, q_):
                walks = random_walk_fallback(rowptr_, colidx_, start_.to(config.device), int(walk_length_))
                return (walks,)

        torch.ops.torch_cluster = _TorchClusterShim()
        _TORCH_CLUSTER_SHIM_SET = True


def split(args):
    """
    NO dataset reload.
    NO embedding reload.
    split tuple에서 매번 GDataset만 새로 만들고,
    pos/comp 샘플링만 매번 수행 (m/M/nv 바뀌니까 이건 어쩔 수 없음).
    """
    global trn_dataset1, val_dataset, tst_dataset
    global loader_fn, tloader_fn

    trn_dataset1 = SubGDataset_hybrid.GDataset(*_SPLITS["train"])
    val_dataset = SubGDataset_hybrid.GDataset(*_SPLITS["valid"])
    tst_dataset = SubGDataset_hybrid.GDataset(*_SPLITS["test"])

    trn_dataset1.sample_pos_comp_train(
        m=args.m, M=args.M, nv=args.nv,
        device=config.device, row=row, col=col, dataset=args.dataset
    )
    val_dataset.sample_pos_comp_test(
        m=args.m, M=args.M, device=config.device,
        row=row, col=col, dataset=args.dataset
    )
    tst_dataset.sample_pos_comp_test(
        m=args.m, M=args.M, device=config.device,
        row=row, col=col, dataset=args.dataset
    )

    trn_dataset1 = trn_dataset1.to(config.device)
    val_dataset = val_dataset.to(config.device)
    tst_dataset = tst_dataset.to(config.device)

    _fix_empty_comp_inplace(trn_dataset1)
    _fix_empty_comp_inplace(val_dataset)
    _fix_empty_comp_inplace(tst_dataset)

    if args.use_maxzeroone:
        def tfunc(ds, bs, shuffle=True, drop_last=True):
            return SubGDataset_hybrid.ZGDataloader(
                ds, bs, z_fn=utils.MaxZOZ, shuffle=shuffle, drop_last=drop_last
            )

        def loader_fn(ds, bs, seed):
            return tfunc(ds, bs)

        def tloader_fn(ds, bs, seed):
            return tfunc(ds, bs, True, False)
    else:
        def loader_fn(ds, bs, seed):
            return SubGDataset_hybrid.GDataloader(ds, bs, seed=seed)

        def tloader_fn(ds, bs, seed):
            return SubGDataset_hybrid.GDataloader(ds, bs, shuffle=True, seed=seed)


def _calc_in_channels(hidden_dim, conv_layer, jk, model, diffusion):
    num_rep = 2 if (model == 2 and (not diffusion)) else 1

    if jk:
        if model == 0:
            return hidden_dim * conv_layer * num_rep
        if model == 2 and (not diffusion):
            return hidden_dim * conv_layer * num_rep
        return hidden_dim * 1 * num_rep
    else:
        if model == 2 and (not diffusion):
            return hidden_dim * num_rep
        return hidden_dim


def buildModel(hidden_dim, conv_layer, dropout, jk, pool1, pool2, z_ratio, aggr, args=None):
    global _NODE_EMB

    if args.use_nodeid:
        hidden_dim = 64

    conv = functools.partial(COMGraphConv, aggr=aggr, dropout=dropout)
    if args.use_gcn_conv:
        conv = functools.partial(GCNConv, add_self_loops=False)

    conv = models_hybrid.COMGraphLayerNet(
        hidden_dim,
        hidden_dim,
        conv_layer,
        max_deg=max_deg,
        activation=nn.ELU(inplace=True),
        jk=jk,
        dropout=dropout,
        conv=conv,
        gn=True
    )

    # NO file read here. use cached tensor.
    if args.use_nodeid:
        if _NODE_EMB is None:
            raise RuntimeError("[SSNP] _NODE_EMB is None. init_cache(args) must be called before buildModel().")
        conv.input_emb = nn.Embedding.from_pretrained(_NODE_EMB.clone(), freeze=False)

    in_channels = _calc_in_channels(
        hidden_dim=int(hidden_dim),
        conv_layer=int(conv_layer),
        jk=int(jk),
        model=int(args.model),
        diffusion=bool(args.diffusion),
    )

    mlp = MLP(channel_list=[in_channels, output_channels], dropout=[0], norm=None, act=None)

    pool_fn_fn = {
        "mean": models_hybrid.MeanPool,
        "max": models_hybrid.MaxPool,
        "sum": models_hybrid.AddPool,
        "size": models_hybrid.SizePool
    }
    if pool1 in pool_fn_fn and pool2 in pool_fn_fn:
        pool_fn1 = pool_fn_fn[pool1]()
        pool_fn2 = pool_fn_fn[pool2]()
        if args.model == 1 or args.model == 2:
            pooling_layers = torch.nn.ModuleList([pool_fn1, pool_fn2])
        else:
            pooling_layers = torch.nn.ModuleList([pool_fn1])
    else:
        raise NotImplementedError

    if args.use_mlp:
        in_channels = _calc_in_channels(
            hidden_dim=int(hidden_dim),
            conv_layer=int(conv_layer),
            jk=int(jk),
            model=int(args.model),
            diffusion=bool(args.diffusion),
        )
        mlp = MLP(channel_list=[in_channels, output_channels], dropout=[0], norm=None, act=None)
        gnn = models_hybrid.COMGraphMLPMasterNet(
            preds=torch.nn.ModuleList([mlp]),
            pools=pooling_layers,
            model_type=args.model,
            hidden_dim=hidden_dim,
            max_deg=max_deg,
            diffusion=args.diffusion
        ).to(config.device)
    else:
        gnn = models_hybrid.COMGraphMasterNet(
            conv,
            torch.nn.ModuleList([mlp]),
            pooling_layers,
            args.model,
            hidden_dim,
            conv_layer,
            args.diffusion
        ).to(config.device)

    return gnn


def _stack_labels_list(y_list, ref_y):
    if len(y_list) == 0:
        if torch.is_tensor(ref_y):
            return torch.empty((0,), dtype=ref_y.dtype)
        return torch.empty((0,), dtype=torch.float)

    y0 = y_list[0]
    if torch.is_tensor(y0):
        return torch.stack([yy if torch.is_tensor(yy) else torch.tensor(yy) for yy in y_list], dim=0)

    if torch.is_tensor(ref_y):
        return torch.tensor(y_list, dtype=ref_y.dtype)
    return torch.tensor(y_list)


def sample_views(args, nve, trn_dataset1, batch_size, repeat):
    trn_dataset = trn_dataset1
    selected_views = random.sample(range(0, args.nv), nve)
    selected_pos = [trn_dataset1.pos_temp[i] for i in selected_views]
    selected_comp = [trn_dataset1.comp_temp[i] for i in selected_views]
    selected_y = [trn_dataset1.y_temp[i] for i in selected_views]

    trn_dataset.pos = torch.stack(list(itertools.chain.from_iterable(selected_pos)), dim=0)
    trn_dataset.comp = pad_sequence(
        list(itertools.chain.from_iterable(selected_comp)),
        batch_first=True,
        padding_value=-1
    ).to(torch.int64)

    flat_y = list(itertools.chain.from_iterable(selected_y))
    trn_dataset.y = _stack_labels_list(flat_y, trn_dataset1.y)

    _fix_empty_comp_inplace(trn_dataset)

    trn_dataset = trn_dataset.to(config.device)
    return SubGDataset_hybrid.GDataloader(trn_dataset, batch_size, seed=repeat)


def setup_task(baseG_):
    global output_channels, score_fn, loss_fn

    if baseG_.y.unique().shape[0] == 2:
        def loss_fn(x, y):
            return BCEWithLogitsLoss()(x.flatten(), y.flatten())

        if baseG_.y.ndim > 1:
            output_channels = baseG_.y.shape[1]
        else:
            output_channels = 1
        score_fn = metrics.binaryf1
    else:
        loss_fn = CrossEntropyLoss()
        output_channels = baseG_.y.unique().shape[0]
        score_fn = metrics.microf1


def train_and_eval(args, params, repeat_seed, max_epoch):
    set_seed(repeat_seed)

    # fixed experiment knobs
    args.M = 1
    args.nv = 20
    args.nve = 5
    args.model = 2

    args.m = int(params["m"])

    split(args)

    batch_size = int(params["batch_size"])
    lr = float(params["lr"])
    resi = float(params["resi"])

    gnn = buildModel(
        hidden_dim=int(params["hidden_dim"]),
        conv_layer=int(params["conv_layer"]),
        dropout=float(params["dropout"]),
        jk=int(params["jk"]),
        pool1=str(params["pool1"]),
        pool2=str(params["pool2"]),
        z_ratio=float(params["z_ratio"]),
        aggr=str(params["aggr"]),
        args=args
    )

    optimizer = Adam(gnn.parameters(), lr=lr)
    scd = lr_scheduler.ReduceLROnPlateau(optimizer, factor=resi, min_lr=5e-5)

    nve = args.nve
    trn_loader1 = sample_views(args, nve, trn_dataset1, batch_size, repeat_seed + 1)
    trn_loader2 = sample_views(args, nve, trn_dataset1, batch_size, repeat_seed + 2)
    trn_loader3 = sample_views(args, nve, trn_dataset1, batch_size, repeat_seed + 3)
    trn_loader4 = sample_views(args, nve, trn_dataset1, batch_size, repeat_seed + 4)

    val_loader = SubGDataset_hybrid.GDataloader(val_dataset, batch_size, shuffle=True, seed=repeat_seed + 11)
    tst_loader = SubGDataset_hybrid.GDataloader(tst_dataset, batch_size, shuffle=True, seed=repeat_seed + 21)

    warmup = 50
    if args.use_mlp:
        warmup = 0

    val_score = 0.0
    tst_score = 0.0
    early_stop = 0
    trn_time = []

    num_div = tst_dataset.y.shape[0] / batch_size
    if args.dataset in ["density", "component", "cut_ratio", "coreness"]:
        num_div /= 5

    patience = 100 / num_div

    for i in range(max_epoch):
        trn_loader = random.choice([trn_loader1, trn_loader2, trn_loader3, trn_loader4])

        # time accounting (LOG ONLY)
        t1 = time.time()
        trn_score, loss = train_hybrid.train(
            optimizer, gnn, trn_loader, score_fn, loss_fn,
            device=config.device, row=row, col=col, run=repeat_seed, epoch=i
        )
        trn_time.append(time.time() - t1)

        scd.step(loss)

        if i >= warmup:
            score, _ = train_hybrid.test(
                gnn, val_loader, score_fn, loss_fn=loss_fn,
                device=config.device, row=row, col=col, run=repeat_seed, epoch=i
            )

            if score > val_score:
                early_stop = 0
                val_score = score
                score_tst, _ = train_hybrid.test(
                    gnn, tst_loader, score_fn, loss_fn=loss_fn,
                    device=config.device, row=row, col=col, run=repeat_seed, epoch=i
                )
                tst_score = score_tst
                # PRINT FORMAT ONLY (remove "train ...")
                print(f"iter {i} loss {loss:.4f} val {val_score:.4f} tst {tst_score:.4f}", flush=True)

            elif score >= val_score - 1e-5:
                score_tst, _ = train_hybrid.test(
                    gnn, tst_loader, score_fn, loss_fn=loss_fn,
                    device=config.device, row=row, col=col, run=repeat_seed, epoch=i
                )
                tst_score = max(score_tst, tst_score)
                # GLASS style prints this case too
                print(f"iter {i} loss {loss:.4f} val {val_score:.4f} tst {score_tst:.4f}", flush=True)

            else:
                early_stop += 1
                if i % 10 == 0:
                    tst_curr, _ = train_hybrid.test(
                        gnn, tst_loader, score_fn, loss_fn=loss_fn,
                        device=config.device, row=row, col=col, run=repeat_seed, epoch=i
                    )
                    print(f"iter {i} loss {loss:.4f} val {score:.4f} tst {tst_curr:.4f}", flush=True)

        if val_score >= 1 - 1e-5:
            early_stop += 1

        if (not args.use_mlp) and (early_stop > patience):
            break

    # AUROC는 학습 도중 절대 출력/계산하지 않고, 여기서 딱 1번만 계산
    tst_auroc, _ = train_hybrid.test(
        gnn, tst_loader, metrics.auroc, loss_fn=loss_fn,
        device=config.device, row=row, col=col, run=repeat_seed, epoch=i
    )

    # GLASS style end line
    print(f"end: epoch {i+1}, train time {sum(trn_time):.2f} s, val {val_score:.3f}, tst {tst_score:.3f}, auroc {tst_auroc:.3f}", flush=True)
    return float(val_score), float(tst_score), float(tst_auroc)


def objective(trial, args):
    params = {
        "m": trial.suggest_categorical("m", [1, 5]),
        "conv_layer": trial.suggest_int("conv_layer", 1, 8),

        "hidden_dim": 64,

        "pool1": trial.suggest_categorical("pool1", ["size", "mean", "max", "sum"]),
        "pool2": trial.suggest_categorical("pool2", ["size", "mean", "max", "sum"]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.6),
        "aggr": trial.suggest_categorical("aggr", ["mean", "sum"]),
        "jk": trial.suggest_categorical("jk", [0, 1]),
        "z_ratio": trial.suggest_float("z_ratio", 0.6, 0.95),
        "batch_size": trial.suggest_int("batch_size", 8, 210),
        "resi": trial.suggest_float("resi", 0.3, 0.9),
    }

    # trial time (LOG ONLY)
    t0 = time.time()
    try:
        val_score, _, _ = train_and_eval(args, params, repeat_seed=0, max_epoch=args.search_epochs)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()
        raise e
    finally:
        print(f"Trial time {time.time() - t0:.2f} s", flush=True)

    return val_score


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='DocRED')

    parser.add_argument('--path', type=str, default="./Emb/node/")

    parser.add_argument('--use_deg', action='store_true')
    parser.add_argument('--use_one', action='store_true')
    parser.add_argument('--use_nodeid', action='store_true')

    parser.add_argument('--model', type=int, default=2)
    parser.add_argument('--M', type=int, default=1)
    parser.add_argument('--nv', type=int, default=20)
    parser.add_argument('--nve', type=int, default=5)
    parser.add_argument('--m', type=int, default=1)

    parser.add_argument('--diffusion', action='store_true')
    parser.add_argument('--use_maxzeroone', action='store_true')

    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--search_epochs', type=int, default=120)

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--use_seed', action='store_true')

    parser.add_argument('--use_gcn_conv', action='store_true')
    parser.add_argument('--use_mlp', action='store_true')

    parser.add_argument('--optruns', type=int, default=100)

    args = parser.parse_args()

    config.set_device(args.device)

    if args.use_seed:
        set_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

    # =========================
    # ONE-TIME LOADS HERE
    # =========================
    init_cache(args)
    setup_task(baseG)

    config_dir = "config/node/SSNP"
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    config_path = f"{config_dir}/{args.dataset}.yml"

    if not os.path.exists(config_path):
        print("=" * 40, flush=True)
        print(f"[SSNP] Config not found. Starting Optuna ({args.optruns} trials)...", flush=True)
        print("=" * 40, flush=True)

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, args), n_trials=args.optruns)

        best_params = dict(study.best_trial.params)
        best_params["hidden_dim"] = 64

        with open(config_path, 'w') as f:
            yaml.dump(best_params, f)

        print(f"Best trial: {best_params}", flush=True)
        print(f"Best Val Score: {study.best_value}", flush=True)
        print(f"Saved best config to {config_path}", flush=True)

        params = best_params
    else:
        print(f"[SSNP] Loading config from {config_path}", flush=True)
        with open(config_path) as f:
            params = yaml.safe_load(f)

        params["hidden_dim"] = 64

    print("-" * 64, flush=True)
    print("[SSNP] Final Evaluation", flush=True)
    print("Params:", params, flush=True)
    print("-" * 64, flush=True)

    # Final Evaluation: keep val per-run internally, but final report is TEST only (as requested)
    tst_scores = []
    tst_aurocs = []

    for r in range(args.repeat):
        seed = (1 << r) - 1
        print(f"\n--- Repeat {r}/{args.repeat} ---", flush=True)
        print(f"seed  {seed}", flush=True)
        val_s, tst_s, tst_a = train_and_eval(args, params, repeat_seed=seed, max_epoch=args.epochs)
        tst_scores.append(tst_s)
        tst_aurocs.append(tst_a)

    print("\n" + "=" * 40, flush=True)
    print(f"Final Results over {args.repeat} runs:", flush=True)
    print(f"Average Test Score: {np.mean(tst_scores):.3f} error {np.std(tst_scores) / np.sqrt(len(tst_scores)):.3f}", flush=True)
    print(f"Average Test AUROC: {np.mean(tst_aurocs):.3f} error {np.std(tst_aurocs) / np.sqrt(len(tst_aurocs)):.3f}", flush=True)
    print("=" * 40, flush=True)

    # keep json writing logic identical except: results should reflect TEST only per your request
    exp_results = {
        f"{args.dataset}_SSNP_model{args.model}": {
            "config_path": config_path,
            "params": params,
            "results": {
                "Test mean": f"{np.mean(tst_scores):.3f}",
                "Test err": f"{np.std(tst_scores) / np.sqrt(len(tst_scores)):.3f}",
                "AUROC mean": f"{np.mean(tst_aurocs):.3f}",
                "AUROC err": f"{np.std(tst_aurocs) / np.sqrt(len(tst_aurocs)):.3f}",
            }
        }
    }
    out_json = f"{args.dataset}_SSNP_results.json"
    with open(out_json, "w") as f:
        json.dump(exp_results, f, indent=2)


if __name__ == "__main__":
    main()
