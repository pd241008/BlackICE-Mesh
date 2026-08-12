import torch
import torch.nn as nn
import torch.nn.functional as F


class TabularMLP(nn.Module):
    def __init__(self, input_dim=18, num_classes=2):
        super(TabularMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
