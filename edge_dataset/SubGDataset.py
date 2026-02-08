# SubGDataset.py
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

class GDataset:
    '''
    UNDIRECTED ONLY.
    Returns per subgraph:
        pos: padded node list (max_nodes,)
        subG_edge: padded edge list (max_edges, 2) with -1 padding
        y: label
    '''
    def __init__(self, x, edge_index, edge_attr, pos, subG_edge, y):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y

        # pos padding (if list)
        if isinstance(pos, list):
            pos_tensors = [torch.tensor(p, dtype=torch.long) for p in pos]
            self.pos = pad_sequence(pos_tensors, batch_first=True, padding_value=-1)
        else:
            self.pos = pos

        # subG_edge는 datasets.py에서 이미 (N, max_edges, 2)로 padding된 텐서로 들어오는 걸 가정
        self.subG_edge = subG_edge

        self.num_nodes = x.shape[0]

    def __len__(self):
        return self.pos.shape[0]

    def __getitem__(self, idx):
        return self.pos[idx], self.subG_edge[idx], self.y[idx]

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        if self.edge_attr is not None:
            self.edge_attr = self.edge_attr.to(device)
        self.pos = self.pos.to(device)
        self.subG_edge = self.subG_edge.to(device)
        self.y = self.y.to(device)
        return self


class GDataloader(DataLoader):
    '''
    Dataloader for GDataset (UNDIRECTED ONLY)
    '''
    def __init__(self, Gdataset, batch_size=64, shuffle=True, drop_last=False):
        super(GDataloader, self).__init__(
            torch.arange(len(Gdataset)).to(Gdataset.x.device),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last
        )
        self.Gdataset = Gdataset

    def get_x(self):
        return self.Gdataset.x

    def get_ei(self):
        return self.Gdataset.edge_index

    def get_ea(self):
        return self.Gdataset.edge_attr

    def get_pos(self):
        return self.Gdataset.pos

    def get_subG_edge(self):
        return self.Gdataset.subG_edge

    def get_y(self):
        return self.Gdataset.y

    def __iter__(self):
        self.iter = super(GDataloader, self).__iter__()
        return self

    def __next__(self):
        perm = next(self.iter)
        return self.get_x(), self.get_ei(), self.get_ea(), \
               self.get_pos()[perm], self.get_subG_edge()[perm], self.get_y()[perm]


class ZGDataloader(GDataloader):
    '''
    Dataloader for GDataset with Distance Labeling.
    z_fn은 node-set(pos)로만 계산하면 됨 (기존 GLASS 방식 그대로).
    '''
    def __init__(self, Gdataset, batch_size=64, shuffle=True, drop_last=False,
                 z_fn=lambda x, pos: torch.zeros((x.shape[0], x.shape[1]), dtype=torch.int64)):
        super(ZGDataloader, self).__init__(Gdataset, batch_size, shuffle, drop_last)
        self.z_fn = z_fn

    def __next__(self):
        perm = next(self.iter)
        tpos = self.get_pos()[perm]
        return self.get_x(), self.get_ei(), self.get_ea(), \
               tpos, self.get_subG_edge()[perm], self.z_fn(self.get_x(), tpos), self.get_y()[perm]
