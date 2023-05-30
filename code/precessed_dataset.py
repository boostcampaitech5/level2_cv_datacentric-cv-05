import os
import pickle

import torch
from torch.utils.data import Dataset


class PreDataset(Dataset):
    def __init__(self, datadir, to_tensor=True):
        self.datadir = datadir
        datalist = os.listdir(datadir)
        self.datalist = [d for d in datalist if d.endswith(".pkl")]
        self.datalist.sort()
        self.to_tensor = to_tensor

    def __getitem__(self, idx):
        with open(file=self.datadir + f"/{self.datalist[idx]}", mode="rb") as f:
            tup = pickle.load(f)
        image, score_map, geo_map, roi_mask = tup
        if self.to_tensor:
            image = torch.Tensor(image)
            score_map = torch.Tensor(score_map)
            geo_map = torch.Tensor(geo_map)
            roi_mask = torch.Tensor(roi_mask)

        return image, score_map, geo_map, roi_mask

    def __len__(self):
        return len(self.datalist)
