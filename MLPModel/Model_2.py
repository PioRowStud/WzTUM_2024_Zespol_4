import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(input_dim,1036)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(1036,1036)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(1036,output_dim)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)
        return out

