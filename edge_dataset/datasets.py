# datasets.py
import os
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import one_hot
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected, to_undirected, negative_sampling
import networkx as nx

# -----------------------------------------------------------------------------
# 1. BaseGraph Class (기존 구조 유지 + subG_edge 추가)
# -----------------------------------------------------------------------------
class BaseGraph(Data):
    def __init__(self, x, edge_index, edge_weight, subG_node, subG_edge, subG_label, mask):
        '''
        A general format for datasets. (UNDIRECTED ONLY)
        Args:
            x: node feature. For our used datasets, x is empty vector.
            edge_index: UNDIRECTED edge list (2, E)
            edge_weight: (E,)
            subG_node: padded node set matrix (num_subg, max_nodes), -1 padding
            subG_edge: padded edge list tensor (num_subg, max_edges, 2), -1 padding
            subG_label: the target of subgraphs.
            mask: shape (num_subg), mask[i]=0/1/2 for train/valid/test
        '''
        super(BaseGraph, self).__init__(x=x,
                                        edge_index=edge_index,
                                        edge_attr=edge_weight,
                                        pos=subG_node,
                                        y=subG_label)
        self.mask = mask
        self.subG_edge = subG_edge  # ★ 핵심: subgraph edge list 자체를 저장
        self.to_undirected()

    def addDegreeFeature(self):
        adj = torch.sparse_coo_tensor(self.edge_index, self.edge_attr,
                                      (self.x.shape[0], self.x.shape[0]))
        degree = torch.sparse.sum(adj, dim=1).to_dense().to(torch.int64)
        self.x = torch.cat((self.x, one_hot(degree).to(torch.float).reshape(
            self.x.shape[0], 1, -1)),
            dim=-1)

    def addOneFeature(self):
        self.x = torch.cat(
            (self.x, torch.ones(self.x.shape[0], self.x.shape[1], 1)),
            dim=-1)

    def setDegreeFeature(self, mod=1):
        adj = torch.sparse_coo_tensor(self.edge_index, self.edge_attr,
                                      (self.x.shape[0], self.x.shape[0]))
        degree = torch.sparse.sum(adj, dim=1).to_dense().to(torch.int64)
        degree = torch.div(degree, mod, rounding_mode='floor')
        degree = torch.unique(degree, return_inverse=True)[1]
        self.x = degree.reshape(self.x.shape[0], 1, -1)

    def setOneFeature(self):
        self.x = torch.ones((self.x.shape[0], 1, 1), dtype=torch.int64)

    def setNodeIdFeature(self):
        self.x = torch.arange(self.x.shape[0], dtype=torch.int64).reshape(
            self.x.shape[0], 1, -1)

    def get_split(self, split: str):
        tar_mask = {"train": 0, "valid": 1, "val": 1, "test": 2}[split]
        sel = (self.mask == tar_mask)
        return self.x, self.edge_index, self.edge_attr, self.pos[sel], self.subG_edge[sel], self.y[sel]

    def to_undirected(self):
        if not is_undirected(self.edge_index):
            self.edge_index, self.edge_attr = to_undirected(
                self.edge_index, self.edge_attr)

    def get_LPdataset(self, use_loop=False):
        neg_edge = negative_sampling(self.edge_index)
        x = self.x
        ei = self.edge_index
        ea = self.edge_attr
        pos = torch.cat((self.edge_index, neg_edge), dim=1).t()
        y = torch.cat((torch.ones(ei.shape[1]),
                       torch.zeros(neg_edge.shape[1]))).to(ei.device)
        if use_loop:
            mask = (ei[0] == ei[1])
            pos_loops = ei[0][mask]
            all_loops = torch.arange(x.shape[0],
                                     device=x.device).reshape(-1, 1)[:, [0, 0]]
            y_loop = torch.zeros(x.shape[0], device=y.device)
            y_loop[pos_loops] = 1
            pos = torch.cat((pos, all_loops), dim=0)
            y = torch.cat((y, y_loop), dim=0)
        return x, ei, ea, pos, y

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        self.pos = self.pos.to(device)
        self.subG_edge = self.subG_edge.to(device)
        self.y = self.y.to(device)
        self.mask = self.mask.to(device)
        return self


# -----------------------------------------------------------------------------
# 2. load_dataset (edge_list.txt + edge_subgraph.jsonl) (UNDIRECTED ONLY)
# -----------------------------------------------------------------------------
def load_dataset(name: str):
    # if name not in ["DocRED", "VisualGenome", "Connectome"]:
    #     raise NotImplementedError("Only DocRED / VisualGenome / Connectome are supported.")

    multilabel = False
    base_path = f"./edge_dataset/{name}"

    # ---------------------------------------------------------
    # (A) Base graph: undirected edge list 그대로 로드
    # ---------------------------------------------------------
    rawedge = nx.read_edgelist(f"{base_path}/edge_list.txt", nodetype=int).edges
    edge_index = torch.tensor([[int(i[0]), int(i[1])] for i in rawedge]).t()  # (2, E)
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float)

    # ---------------------------------------------------------
    # (B) Subgraphs: edge_subgraph.jsonl 로드
    #     - pos: subgraph node list (edge endpoint union)
    #     - subG_edge: subgraph edge list 자체 (각 row가 [u,v])
    # ---------------------------------------------------------
    cache_files = [
        f"{base_path}/train_pos.pt",
        f"{base_path}/train_edges.pt",
        f"{base_path}/train_y.pt",
        f"{base_path}/valid_pos.pt",
        f"{base_path}/valid_edges.pt",
        f"{base_path}/valid_y.pt",
        f"{base_path}/test_pos.pt",
        f"{base_path}/test_edges.pt",
        f"{base_path}/test_y.pt",
        f"{base_path}/multilabel_flag.pt",
    ]

    if all([os.path.exists(p) for p in cache_files]):
        train_pos = torch.load(f"{base_path}/train_pos.pt")
        train_edges = torch.load(f"{base_path}/train_edges.pt")
        train_y = torch.load(f"{base_path}/train_y.pt")

        valid_pos = torch.load(f"{base_path}/valid_pos.pt")
        valid_edges = torch.load(f"{base_path}/valid_edges.pt")
        valid_y = torch.load(f"{base_path}/valid_y.pt")

        test_pos = torch.load(f"{base_path}/test_pos.pt")
        test_edges = torch.load(f"{base_path}/test_edges.pt")
        test_y = torch.load(f"{base_path}/test_y.pt")

        multilabel = bool(torch.load(f"{base_path}/multilabel_flag.pt").item())
    else:
        labels = {}
        label_idx = 0

        train_nodes, valid_nodes, test_nodes = [], [], []
        train_edges_list, valid_edges_list, test_edges_list = [], [], []
        train_labels, valid_labels, test_labels = [], [], []

        jsonl_path = f"{base_path}/edge_subgraph.jsonl"
        with open(jsonl_path, "r") as fin:
            for line in fin:
                if line.strip() == "":
                    continue
                obj = json.loads(line)

                split_str = obj["split"].strip()
                if split_str == "val":
                    split_str = "valid"

                lab = obj["label"]
                if isinstance(lab, str) and ("-" in lab):
                    multilabel = True
                    lab_list = lab.split("-")
                else:
                    lab_list = [str(lab)]

                for l in lab_list:
                    if l not in labels:
                        labels[l] = label_idx
                        label_idx += 1

                # subgraph edge list (UNDIRECTED) 그대로 저장
                edges = [[int(u), int(v)] for (u, v) in obj["graph"]]
                edge_tensor = torch.tensor(edges, dtype=torch.long).reshape(-1, 2)  # (E_s, 2) even if empty

                # node list = endpoints union
                node_set = set()
                for u, v in edges:
                    node_set.add(u); node_set.add(v)
                node_list = sorted(list(node_set))
                node_tensor = torch.tensor(node_list, dtype=torch.long)

                y_item = [labels[l] for l in lab_list]

                if split_str == "train":
                    train_nodes.append(node_tensor)
                    train_edges_list.append(edge_tensor)
                    train_labels.append(y_item)
                elif split_str == "valid":
                    valid_nodes.append(node_tensor)
                    valid_edges_list.append(edge_tensor)
                    valid_labels.append(y_item)
                elif split_str == "test":
                    test_nodes.append(node_tensor)
                    test_edges_list.append(edge_tensor)
                    test_labels.append(y_item)

        # labels -> tensor
        if multilabel:
            all_ll = train_labels + valid_labels + test_labels
            max_label = max([max(i) for i in all_ll]) if len(all_ll) > 0 else -1

            def multilabel_tensor(ll_list):
                Y = torch.zeros(len(ll_list), max_label + 1, dtype=torch.float)
                for i, labs in enumerate(ll_list):
                    Y[i][torch.LongTensor(labs)] = 1.0
                return Y

            train_y = multilabel_tensor(train_labels)
            valid_y = multilabel_tensor(valid_labels)
            test_y = multilabel_tensor(test_labels)
        else:
            train_y = torch.tensor([i[0] for i in train_labels], dtype=torch.long)
            valid_y = torch.tensor([i[0] for i in valid_labels], dtype=torch.long)
            test_y = torch.tensor([i[0] for i in test_labels], dtype=torch.long)

        # pos padding: (num_subg, max_nodes)
        train_pos = pad_sequence(train_nodes, batch_first=True, padding_value=-1) if len(train_nodes) > 0 else torch.empty((0, 0), dtype=torch.long)
        valid_pos = pad_sequence(valid_nodes, batch_first=True, padding_value=-1) if len(valid_nodes) > 0 else torch.empty((0, 0), dtype=torch.long)
        test_pos  = pad_sequence(test_nodes,  batch_first=True, padding_value=-1) if len(test_nodes) > 0 else torch.empty((0, 0), dtype=torch.long)

        # subgraph edge padding: (num_subg, max_edges, 2)
        def pad_edges(edge_list):
            if len(edge_list) == 0:
                return torch.empty((0, 0, 2), dtype=torch.long)
            max_e = max([e.shape[0] for e in edge_list])
            out = torch.full((len(edge_list), max_e, 2), -1, dtype=torch.long)
            for i, e in enumerate(edge_list):
                out[i, :e.shape[0], :] = e
            return out

        train_edges = pad_edges(train_edges_list)
        valid_edges = pad_edges(valid_edges_list)
        test_edges  = pad_edges(test_edges_list)

        # cache
        torch.save(train_pos, f"{base_path}/train_pos.pt")
        torch.save(train_edges, f"{base_path}/train_edges.pt")
        torch.save(train_y, f"{base_path}/train_y.pt")

        torch.save(valid_pos, f"{base_path}/valid_pos.pt")
        torch.save(valid_edges, f"{base_path}/valid_edges.pt")
        torch.save(valid_y, f"{base_path}/valid_y.pt")

        torch.save(test_pos, f"{base_path}/test_pos.pt")
        torch.save(test_edges, f"{base_path}/test_edges.pt")
        torch.save(test_y, f"{base_path}/test_y.pt")

        torch.save(torch.tensor(int(multilabel)), f"{base_path}/multilabel_flag.pt")

    # ---------------------------------------------------------
    # (B-2) Harmonize padding across splits BEFORE torch.cat
    #       - train/valid/test가 각각 따로 pad되어 있으면 길이가 달라 cat에서 터짐
    # ---------------------------------------------------------
    def _pad_2d_to(x, target_len, pad_value=-1):
        if not isinstance(x, torch.Tensor):
            return x
        if x.numel() == 0:
            return x
        cur = x.shape[1]
        if cur == target_len:
            return x
        out = torch.full((x.shape[0], target_len), pad_value, dtype=x.dtype)
        out[:, :cur] = x
        return out

    def _pad_3d_to(x, target_len, pad_value=-1):
        if not isinstance(x, torch.Tensor):
            return x
        if x.numel() == 0:
            return x
        cur = x.shape[1]
        if cur == target_len:
            return x
        out = torch.full((x.shape[0], target_len, x.shape[2]), pad_value, dtype=x.dtype)
        out[:, :cur, :] = x
        return out

    max_nodes = 0
    if train_pos.numel() > 0:
        max_nodes = max(max_nodes, train_pos.shape[1])
    if valid_pos.numel() > 0:
        max_nodes = max(max_nodes, valid_pos.shape[1])
    if test_pos.numel() > 0:
        max_nodes = max(max_nodes, test_pos.shape[1])

    if max_nodes > 0:
        train_pos = _pad_2d_to(train_pos, max_nodes, -1)
        valid_pos = _pad_2d_to(valid_pos, max_nodes, -1)
        test_pos  = _pad_2d_to(test_pos,  max_nodes, -1)

    max_edges = 0
    if train_edges.numel() > 0:
        max_edges = max(max_edges, train_edges.shape[1])
    if valid_edges.numel() > 0:
        max_edges = max(max_edges, valid_edges.shape[1])
    if test_edges.numel() > 0:
        max_edges = max(max_edges, test_edges.shape[1])

    if max_edges > 0:
        train_edges = _pad_3d_to(train_edges, max_edges, -1)
        valid_edges = _pad_3d_to(valid_edges, max_edges, -1)
        test_edges  = _pad_3d_to(test_edges,  max_edges, -1)

    # ---------------------------------------------------------
    # (C) Merge splits into one BaseGraph (mask로 구분)
    # ---------------------------------------------------------
    mask = torch.cat(
        (torch.zeros(train_y.shape[0], dtype=torch.int64),
         torch.ones(valid_y.shape[0], dtype=torch.int64),
         2 * torch.ones(test_y.shape[0], dtype=torch.int64)),
        dim=0
    )

    pos = torch.cat((train_pos, valid_pos, test_pos), dim=0) if (train_pos.numel() + valid_pos.numel() + test_pos.numel()) > 0 else torch.empty((0, 0), dtype=torch.long)
    subG_edge = torch.cat((train_edges, valid_edges, test_edges), dim=0) if (train_edges.numel() + valid_edges.numel() + test_edges.numel()) > 0 else torch.empty((0, 0, 2), dtype=torch.long)

    if multilabel:
        label = torch.cat((train_y, valid_y, test_y), dim=0).to(torch.float)
    else:
        label = torch.cat((train_y, valid_y, test_y), dim=0).to(torch.float)

    num_node = int(torch.max(edge_index).item()) + 1 if edge_index.numel() > 0 else 0
    if pos.numel() > 0:
        num_node = max(num_node, int(torch.max(pos[pos >= 0]).item()) + 1)
    x = torch.empty((num_node, 1, 0))

    return BaseGraph(x, edge_index, edge_weight, pos, subG_edge, label, mask)
