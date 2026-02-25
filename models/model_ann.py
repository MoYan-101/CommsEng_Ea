"""
models/model_ann.py

Implementation of an ANN for multi-output regression with optional Dropout.
"""
import torch
import torch.nn as nn

class ANNRegression(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=[32,64,32],
                 dropout=0.0,
                 activation="ReLU",
                 random_seed=None):
        """
        :param input_dim:  输入特征维度
        :param output_dim: 输出维度
        :param hidden_dims: 隐藏层大小列表
        :param dropout: Dropout 概率, e.g. 0.2
        :param activation: 激活函数名称, e.g. "ReLU" / "Tanh" / "Sigmoid"
        :param random_seed: 若指定, 则使用该种子来初始化网络权重
        """
        super().__init__()

        if random_seed is not None:
            torch.manual_seed(random_seed)

        act_fn = None
        if activation.lower() == "relu":
            act_fn = nn.ReLU
        elif activation.lower() == "tanh":
            act_fn = nn.Tanh
        elif activation.lower() == "sigmoid":
            act_fn = nn.Sigmoid
        elif activation.lower() == "leakyrelu":
            act_fn = lambda: nn.LeakyReLU(negative_slope=0.005)
        else:
            act_fn = nn.ReLU

        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(act_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
