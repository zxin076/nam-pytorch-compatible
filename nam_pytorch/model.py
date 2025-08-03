# nam_pytorch/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class NAMBlock(nn.Module):
    """单特征对应一个 MLP 子网络"""
    def __init__(self, input_dim=1, hidden_dims=[64, 64]):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))  # 输出单一贡献值
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class NAM(nn.Module):
    """NAM 模型主结构"""
    def __init__(self, input_dim, hidden_dims=[64, 64]):
        super().__init__()
        self.blocks = nn.ModuleList([
            NAMBlock(input_dim=1, hidden_dims=hidden_dims) for _ in range(input_dim)
        ])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 每列单独送入对应的 MLP block
        contributions = [block(x[:, i:i+1]) for i, block in enumerate(self.blocks)]
        output = torch.stack(contributions, dim=0).sum(dim=0) + self.bias
        return output.squeeze(1)