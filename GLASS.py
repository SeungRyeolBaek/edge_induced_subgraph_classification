import argparse
import torch
import torch.nn as nn
import functools
import numpy as np
import time
import random
import yaml
import os

import optuna
from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

# --- [Custom Imports] ---
from impl import models, train, metrics, utils, config
from edge_dataset import node_SubGDataset as SubGDataset 
from edge_dataset import node_datasets 

parser = argparse.ArgumentParser(description='')
# Dataset settings
parser.add_argument('--dataset', type=str, default='DocRED') 
# Node feature settings. 
parser.add_argument('--use_deg', action='store_true')
parser.add_argument('--use_one', action='store_true')
parser.add_argument('--use_nodeid', action='store_true')
# node label settings
parser.add_argument('--use_maxzeroone', action='store_true')

# Optimization settings
parser.add_argument('--optruns', type=int, default=100, help="Number of trials for Optuna")
parser.add_argument('--repeat', type=int, default=10, help="Repeats for final evaluation")
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--use_seed', action='store_true')
parser.add_argument('--path', type=str, default="./Emb/node/") 

args = parser.parse_args()
config.set_device(args.device)

def set_seed(seed: int):
    print("seed ", seed, flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

if args.use_seed:
    set_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- [Load Dataset] ---
print(f"Loading dataset {args.dataset}...", flush=True)
baseG = node_datasets.load_dataset(args.dataset)

trn_dataset, val_dataset, tst_dataset = None, None, None
max_deg, output_channels = 0, 1
score_fn = None

# Classification Logic
unique_labels = baseG.y.unique()
if len(unique_labels) == 2 and baseG.y.ndim == 1:
    def loss_fn(x, y):
        return BCEWithLogitsLoss()(x.flatten(), y.flatten())
    baseG.y = baseG.y.to(torch.float)
    output_channels = 1
    score_fn = metrics.binaryf1
elif baseG.y.ndim > 1:
    def loss_fn(x, y):
        return BCEWithLogitsLoss()(x, y)
    output_channels = baseG.y.shape[1]
    score_fn = metrics.binaryf1 
else:
    baseG.y = baseG.y.to(torch.int64)
    loss_fn = CrossEntropyLoss()
    output_channels = len(unique_labels)
    score_fn = metrics.microf1

loader_fn = SubGDataset.GDataloader
tloader_fn = SubGDataset.GDataloader

def split():
    global trn_dataset, val_dataset, tst_dataset, baseG
    global max_deg, output_channels, loader_fn, tloader_fn
    
    # Dataset Re-initialization for correct splitting logic
    baseG = node_datasets.load_dataset(args.dataset) # Reload to ensure clean state
    if len(unique_labels) == 2 and baseG.y.ndim == 1:
        baseG.y = baseG.y.to(torch.float)
    elif baseG.y.ndim > 1:
        pass
    else:
        baseG.y = baseG.y.to(torch.int64)

    if args.use_deg:
        baseG.setDegreeFeature()
    elif args.use_one:
        baseG.setOneFeature()
    elif args.use_nodeid:
        baseG.setNodeIdFeature()
    else:
        baseG.setNodeIdFeature()

    max_deg = torch.max(baseG.x)
    baseG.to(config.device)
    
    trn_dataset = SubGDataset.GDataset(*baseG.get_split("train"))
    val_dataset = SubGDataset.GDataset(*baseG.get_split("valid"))
    tst_dataset = SubGDataset.GDataset(*baseG.get_split("test"))
    
    if args.use_maxzeroone:
        def tfunc(ds, bs, shuffle=True, drop_last=True):
            return SubGDataset.ZGDataloader(ds, bs, z_fn=utils.MaxZOZ, shuffle=shuffle, drop_last=drop_last)
        def loader_fn(ds, bs): return tfunc(ds, bs)
        def tloader_fn(ds, bs): return tfunc(ds, bs, True, False)
    else:
        def loader_fn(ds, bs): return SubGDataset.GDataloader(ds, bs)
        def tloader_fn(ds, bs): return SubGDataset.GDataloader(ds, bs, shuffle=True)

def buildModel(hidden_dim, conv_layer, dropout, jk, pool, z_ratio, aggr):
    conv = models.EmbZGConv(hidden_dim, hidden_dim, conv_layer, max_deg=max_deg,
                            activation=nn.ELU(inplace=True), jk=jk, dropout=dropout,
                            conv=functools.partial(models.GLASSConv, aggr=aggr, z_ratio=z_ratio, dropout=dropout),
                            gn=True)

    if args.use_nodeid:
        emb_file = f"{args.path}{args.dataset}_{hidden_dim}.pt"
        if os.path.exists(emb_file):
            print(f"load {emb_file}", flush=True)
            emb = torch.load(emb_file, map_location=torch.device('cpu')).detach()
            conv.input_emb = nn.Embedding.from_pretrained(emb, freeze=False)

    mlp = nn.Linear(hidden_dim * (conv_layer) if jk else hidden_dim, output_channels)
    pool_fn_fn = { "mean": models.MeanPool, "max": models.MaxPool, "sum": models.AddPool, "size": models.SizePool }
    pool_fn1 = pool_fn_fn[pool]()
    gnn = models.GLASS(conv, torch.nn.ModuleList([mlp]), torch.nn.ModuleList([pool_fn1])).to(config.device)
    return gnn

def train_and_eval(params, repeat_seed, max_epoch=300):
    """
    기존 코드의 test 함수 로직을 그대로 사용
    """
    set_seed(repeat_seed)
    split()
    
    gnn = buildModel(params['hidden_dim'], params['conv_layer'], params['dropout'], 
                     params['jk'], params['pool'], params['z_ratio'], params['aggr'])
    
    batch_size = params['batch_size']
    lr = params['lr']
    resi = params['resi']

    trn_loader = loader_fn(trn_dataset, batch_size)
    val_loader = tloader_fn(val_dataset, batch_size)
    tst_loader = tloader_fn(tst_dataset, batch_size)

    optimizer = Adam(gnn.parameters(), lr=lr)
    scd = lr_scheduler.ReduceLROnPlateau(optimizer, factor=resi, min_lr=5e-5)
    
    val_score = 0
    tst_score = 0
    early_stop = 0
    trn_time = []
    
    num_div = tst_dataset.y.shape[0] / batch_size
    num_div = max(1.0, float(num_div))
    if args.dataset in ["density", "component", "cut_ratio", "coreness"]: num_div /= 5

    for i in range(max_epoch):
        t1 = time.time()
        loss = train.train(optimizer, gnn, trn_loader, loss_fn)
        trn_time.append(time.time() - t1)
        scd.step(loss)
        if i >= 100 / num_div:
            score, _ = train.test(gnn, val_loader, score_fn, loss_fn=loss_fn)

            if score > val_score:
                early_stop = 0
                val_score = score
                score_tst, _ = train.test(gnn, tst_loader, score_fn, loss_fn=loss_fn)
                tst_score = max(score_tst, tst_score)
                print(f"iter {i} loss {loss:.4f} val {val_score:.4f} tst {tst_score:.4f}", flush=True)
            
            elif score >= val_score - 1e-5:
                score_tst, _ = train.test(gnn, tst_loader, score_fn, loss_fn=loss_fn)
                tst_score = max(score_tst, tst_score)
                print(f"iter {i} loss {loss:.4f} val {val_score:.4f} tst {score_tst:.4f}", flush=True)
            
            else:
                early_stop += 1
                if i % 10 == 0:
                     tst_curr, _ = train.test(gnn, tst_loader, score_fn, loss_fn=loss_fn)
                     print(f"iter {i} loss {loss:.4f} val {score:.4f} tst {tst_curr:.4f}", flush=True)

        if val_score >= 1 - 1e-5:
            early_stop += 1
        if early_stop > 100 / num_div:
            break

    # AUROC는 학습 도중 절대 출력/계산하지 않고, 여기서 딱 1번만 계산
    tst_auroc, _ = train.test(gnn, tst_loader, metrics.auroc, loss_fn=loss_fn)

    print(f"end: epoch {i+1}, train time {sum(trn_time):.2f} s, val {val_score:.3f}, tst {tst_score:.3f}, auroc {tst_auroc:.3f}", flush=True)
    return val_score, tst_score, tst_auroc

def objective(trial):
    """
    Optuna Objective Function
    """
    params = {
        "conv_layer": trial.suggest_int("conv_layer", 1, 8),
        "hidden_dim": 64, 
        "pool": trial.suggest_categorical("pool", ["size", "mean", "max", "sum"]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.6),
        "aggr": trial.suggest_categorical("aggr", ["mean", "sum"]),
        "jk": trial.suggest_categorical("jk", [0, 1]),
        "z_ratio": trial.suggest_float("z_ratio", 0.6, 0.95),
        "batch_size": trial.suggest_int("batch_size", 24, 210),
        "resi": trial.suggest_float("resi", 0.3, 0.9)
    }

    # Search 단계에서는 1회만 (repeat=0 seed 사용)
    # 기존 코드와 동일한 로그를 출력함
    try:
        val_score, _, _ = train_and_eval(params, 0, max_epoch=200) # Search는 Epoch 약간 줄이거나 300 유지 가능
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()
        else:
            raise e
    
    return val_score

def main():
    config_dir = "config/node/GLASS"
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
        
    config_path = f"{config_dir}/{args.dataset}.yml"

    if not os.path.exists(config_path):
        print("="*40, flush=True)
        print(f"Config not found. Starting Optuna Optimization ({args.optruns} trials)...", flush=True)
        print("="*40, flush=True)
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.optruns)
        
        print(f"Best trial: {study.best_trial.params}", flush=True)
        print(f"Best Val Score: {study.best_value}", flush=True)
        
        best_params = study.best_trial.params
        best_params['hidden_dim'] = 64 
        
        with open(config_path, 'w') as f:
            yaml.dump(best_params, f)
        print(f"Saved best config to {config_path}", flush=True)
        
        params = best_params
    else:
        print(f"Loading config from {config_path}", flush=True)
        with open(config_path) as f:
            params = yaml.safe_load(f)

    # Final Evaluation
    print(f"\nRunning Final Evaluation on Test Set ({args.repeat} repeats)...", flush=True)
    print("Params:", params, flush=True)
    
    tst_scores = []
    tst_aurocs = []
    
    for r in range(args.repeat):
        print(f"\n--- Repeat {r}/{args.repeat} ---", flush=True)
        # Final Eval은 seed를 변경하며 수행
        val_s, tst_s, tst_a = train_and_eval(params, (1 << r) - 1, max_epoch=300)
        tst_scores.append(tst_s)
        tst_aurocs.append(tst_a)
    
    print("\n" + "="*40, flush=True)
    print(f"Final Results over {args.repeat} runs:", flush=True)
    print(f"Average Test Score: {np.mean(tst_scores):.3f} error {np.std(tst_scores) / np.sqrt(len(tst_scores)):.3f}", flush=True)
    print(f"Average Test AUROC: {np.mean(tst_aurocs):.3f} error {np.std(tst_aurocs) / np.sqrt(len(tst_aurocs)):.3f}", flush=True)
    print("="*40, flush=True)

if __name__ == "__main__":
    main()
