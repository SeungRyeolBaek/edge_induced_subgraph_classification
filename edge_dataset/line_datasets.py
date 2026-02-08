import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import one_hot
from torch_geometric.utils import is_undirected, to_undirected, negative_sampling
from torch_geometric.data import Data
import networkx as nx
import os

# -----------------------------------------------------------------------------
# 1. BaseGraph Class (구조 유지)
# -----------------------------------------------------------------------------
class BaseGraph(Data):
    def __init__(self, x, edge_index, edge_weight, subG_node, subG_label, mask):
        super(BaseGraph, self).__init__(x=x,
                                        edge_index=edge_index,
                                        edge_attr=edge_weight,
                                        pos=subG_node,
                                        y=subG_label)
        self.mask = mask
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
        tar_mask = {"train": 0, "valid": 1, "test": 2}[split]
        return self.x, self.edge_index, self.edge_attr, self.pos[
            self.mask == tar_mask], self.y[self.mask == tar_mask]

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
        self.y = self.y.to(device)
        self.mask = self.mask.to(device)
        return self


# -----------------------------------------------------------------------------
# 2. load_line_dataset (Line Graph 전용)
# -----------------------------------------------------------------------------
def load_line_dataset(name: str):
    """
    DocRED, VisualGenome 등의 Line Graph 데이터를 로드.
    파일명 충돌 방지를 위해 캐시 파일명 앞에 'line_' 접두어를 사용함.
    """
    
    if name in ["DocRED", "VisualGenome","Connectome"]:
        multilabel = False
        
        # 기본 경로 설정
        base_path = f"./edge_dataset/{name}"
        
        # ---------------------------------------------------------
        # Inner function: read_subgraphs
        # ---------------------------------------------------------
        def read_subgraphs(sub_f):
            label_idx = 0
            labels = {}
            train_sub_G, val_sub_G, test_sub_G = [], [], []
            train_sub_G_label, val_sub_G_label, test_sub_G_label = [], [], []
            train_mask, val_mask, test_mask = [], [], []
            nonlocal multilabel
            
            # line_subgraph.pth (Line Graph Node IDs) 읽기
            with open(sub_f, 'r') as fin:
                subgraph_idx = 0
                for line in fin:
                    parts = line.strip().split("\t")
                    if len(parts) < 3: continue 

                    nodes_str = parts[0]
                    label_str = parts[1]
                    split_str = parts[2]

                    nodes = [
                        int(n) for n in nodes_str.split("-")
                        if n != ""
                    ]
                    
                    if len(nodes) != 0:
                        l = label_str.split("-")
                        if len(l) > 1:
                            multilabel = True
                        
                        for lab in l:
                            if lab not in labels.keys():
                                labels[lab] = label_idx
                                label_idx += 1
                        
                        if split_str.strip() == "train":
                            train_sub_G.append(nodes)
                            train_sub_G_label.append([labels[lab] for lab in l])
                            train_mask.append(subgraph_idx)
                        elif split_str.strip() == "val":
                            val_sub_G.append(nodes)
                            val_sub_G_label.append([labels[lab] for lab in l])
                            val_mask.append(subgraph_idx)
                        elif split_str.strip() == "test":
                            test_sub_G.append(nodes)
                            test_sub_G_label.append([labels[lab] for lab in l])
                            test_mask.append(subgraph_idx)
                        subgraph_idx += 1
            
            if not multilabel:
                train_sub_G_label = torch.tensor(train_sub_G_label).squeeze()
                val_sub_G_label = torch.tensor(val_sub_G_label).squeeze()
                test_sub_G_label = torch.tensor(test_sub_G_label).squeeze()

            if len(val_mask) < len(test_mask):
                return train_sub_G, train_sub_G_label, test_sub_G, test_sub_G_label, val_sub_G, val_sub_G_label

            return train_sub_G, train_sub_G_label, val_sub_G, val_sub_G_label, test_sub_G, test_sub_G_label

        # ---------------------------------------------------------
        # Caching Logic (파일명 앞에 'line_' 붙임)
        # ---------------------------------------------------------
        # 캐시 파일들이 존재하는지 확인 (line_ 접두어 사용)
        if os.path.exists(f"{base_path}/line_train_sub_G.pt"):
            print(f"Loading cached line graph data from {base_path}...")
            train_sub_G = torch.load(f"{base_path}/line_train_sub_G.pt")
            train_sub_G_label = torch.load(f"{base_path}/line_train_sub_G_label.pt")
            val_sub_G = torch.load(f"{base_path}/line_val_sub_G.pt")
            val_sub_G_label = torch.load(f"{base_path}/line_val_sub_G_label.pt")
            test_sub_G = torch.load(f"{base_path}/line_test_sub_G.pt")
            test_sub_G_label = torch.load(f"{base_path}/line_test_sub_G_label.pt")
        else:
            print(f"Processing {base_path}/line_subgraph.pth ...")
            # line_subgraph.pth를 읽음
            train_sub_G, train_sub_G_label, val_sub_G, val_sub_G_label, test_sub_G, test_sub_G_label = read_subgraphs(
                f"{base_path}/line_subgraph.pth")
            
            # Cache 저장 (line_ 접두어 사용)
            torch.save(train_sub_G, f"{base_path}/line_train_sub_G.pt")
            torch.save(train_sub_G_label, f"{base_path}/line_train_sub_G_label.pt")
            torch.save(val_sub_G, f"{base_path}/line_val_sub_G.pt")
            torch.save(val_sub_G_label, f"{base_path}/line_val_sub_G_label.pt")
            torch.save(test_sub_G, f"{base_path}/line_test_sub_G.pt")
            torch.save(test_sub_G_label, f"{base_path}/line_test_sub_G_label.pt")

        # ---------------------------------------------------------
        # Construct Data Objects
        # ---------------------------------------------------------
        mask = torch.cat(
            (torch.zeros(len(train_sub_G_label), dtype=torch.int64),
             torch.ones(len(val_sub_G_label), dtype=torch.int64),
             2 * torch.ones(len(test_sub_G_label))),
            dim=0)
        
        if multilabel:
            tlist = train_sub_G_label + val_sub_G_label + test_sub_G_label
            max_label = max([max(i) for i in tlist])
            label = torch.zeros(len(tlist), max_label + 1)
            for idx, ll in enumerate(tlist):
                label[idx][torch.LongTensor(ll)] = 1
        else:
            label = torch.cat(
                (train_sub_G_label, val_sub_G_label, test_sub_G_label))
        
        # pad_sequence
        pos = pad_sequence(
            [torch.tensor(i) for i in train_sub_G + val_sub_G + test_sub_G],
            batch_first=True,
            padding_value=-1)
        
        # Edge List 로드 (line_edge_list.txt)
        # nodetype=int 필수
        rawedge = nx.read_edgelist(f"{base_path}/line_edge_list.txt", nodetype=int).edges
        edge_index = torch.tensor([[int(i[0]), int(i[1])]
                                   for i in rawedge]).t()
        
        num_node = max([torch.max(pos), torch.max(edge_index)]) + 1
        x = torch.empty((num_node, 1, 0))

        return BaseGraph(x, edge_index, torch.ones(edge_index.shape[1]), pos,
                         label.to(torch.float), mask)
    else:
        raise NotImplementedError("Only DocRED and VisualGenome are supported.")