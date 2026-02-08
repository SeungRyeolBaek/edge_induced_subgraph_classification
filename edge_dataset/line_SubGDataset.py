import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

class LineGDataset:
    '''
    A class to contain splitted data for Custom Line-Graph Subgraphs.
    Structure is IDENTICAL to GDataset.
    '''
    def __init__(self, x, edge_index, edge_attr, pos, y):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y

        # [핵심] 가변 길이 pos 리스트를 패딩된 텐서로 변환 (Node 코드와 동일)
        if isinstance(pos, list):
            pos_tensors = [torch.tensor(p, dtype=torch.long) for p in pos]
            # batch_first=True -> (N_subgraphs, Max_Nodes)
            self.pos = pad_sequence(pos_tensors, batch_first=True, padding_value=-1)
        else:
            self.pos = pos

        self.num_nodes = x.shape[0]

    def __len__(self):
        return self.pos.shape[0]

    def __getitem__(self, idx):
        return self.pos[idx], self.y[idx]

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        if self.edge_attr is not None:
            self.edge_attr = self.edge_attr.to(device)
        self.pos = self.pos.to(device)
        self.y = self.y.to(device)
        return self


class LineGDataloader(DataLoader):
    '''
    Dataloader for LineGDataset
    '''
    def __init__(self, Gdataset, batch_size=64, shuffle=True, drop_last=False):
        super(LineGDataloader, self).__init__(
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

    def get_y(self):
        return self.Gdataset.y

    def __iter__(self):
        self.iter = super(LineGDataloader, self).__iter__()
        return self

    def __next__(self):
        perm = next(self.iter)
        return self.get_x(), self.get_ei(), self.get_ea(), \
               self.get_pos()[perm], self.get_y()[perm]


class LineZGDataloader(LineGDataloader):
    '''
    Dataloader for LineGDataset with Distance Labeling.
    '''
    def __init__(self, Gdataset, batch_size=64, shuffle=True, drop_last=False,
                 z_fn=lambda x, y: torch.zeros((x.shape[0], x.shape[1]), dtype=torch.int64)):
        super(LineZGDataloader, self).__init__(Gdataset, batch_size, shuffle, drop_last)
        self.z_fn = z_fn

    def __next__(self):
        perm = next(self.iter)
        tpos = self.get_pos()[perm]
        return self.get_x(), self.get_ei(), self.get_ea(), \
               tpos, self.z_fn(self.get_x(), tpos), self.get_y()[perm]