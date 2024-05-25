import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class TorchFacialFeaturesDataset(Dataset):
    def __init__(self):
        Xy = pd.read_csv("../Data/EditedData.csv.gz", compression="gzip")
        self.n_samples = Xy.shape[0]
        X = Xy.iloc[:, -1].apply(lambda x: np.fromstring(x, sep=" ", dtype=np.float64)).values/255
        self.x = torch.tensor(np.stack(X))
        self.y = torch.tensor(Xy.iloc[:, :-1].values)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        return self.x[item], self.y[item]