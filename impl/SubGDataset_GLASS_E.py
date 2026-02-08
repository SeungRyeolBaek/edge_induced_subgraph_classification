# impl/SubGDataset_GLASS_E.py
import torch
from torch.utils.data import DataLoader


class GDataset:
    """
    GLASS_E 전용 (기존 impl/SubGDataset.py와 호환 유지)
    pos: (S, maxN) padded with -1
    """
    def __init__(self, x, edge_index, edge_attr, pos, y):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.pos = pos
        self.num_nodes = x.shape[0]

    def __len__(self):
        return self.pos.shape[0]

    def __getitem__(self, idx):
        return self.pos[idx], self.y[idx]

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        self.pos = self.pos.to(device)
        self.y = self.y.to(device)
        return self


class GDataloader(DataLoader):
    """
    기존 GDataloader와 동일 동작:
      - iterator는 subgraph row index를 뽑고
      - (x, ei, ea, pos_batch, y_batch)를 반환
    """
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

    def get_y(self):
        return self.Gdataset.y

    def __iter__(self):
        self.iter = super(GDataloader, self).__iter__()
        return self

    def __next__(self):
        perm = next(self.iter)
        return self.get_x(), self.get_ei(), self.get_ea(), self.get_pos()[perm], self.get_y()[perm]


class ZGDataloader(GDataloader):
    """
    GLASS_E 전용 ZGDataloader:
      - 핵심: perm(=subgraph row indices)을 z_fn에 함께 넘겨줌
      - 기존 impl/SubGDataset.py는 수정하지 않음
    """
    def __init__(self,
                 Gdataset,
                 batch_size=64,
                 shuffle=True,
                 drop_last=False,
                 z_fn=lambda x, pos, perm: torch.zeros((x.shape[0], x.shape[1]), dtype=torch.int64)):
        super(ZGDataloader, self).__init__(Gdataset, batch_size, shuffle, drop_last)
        self.z_fn = z_fn

    def __next__(self):
        perm = next(self.iter)
        tpos = self.get_pos()[perm]
        z = self.z_fn(self.get_x(), tpos, perm)
        return self.get_x(), self.get_ei(), self.get_ea(), tpos, z, self.get_y()[perm]
