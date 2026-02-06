import xlrd
import numpy as np
import torch.optim as optim
from efficient_kan import KAN
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.,
                 channel_first=False):
        """
        :param channel_first: if True, during forward the tensor shape is [B, C, T, J] and fc layers are performed with
                              1x1 convolutions.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.batch512=nn.BatchNorm1d(512)
        self.batch128 = nn.BatchNorm1d(128)
        self.batch64 = nn.BatchNorm1d(64)
        self.batch2 = nn.BatchNorm1d(2)
        self.sigmoid2=nn.Sigmoid()
        self.fc1 = nn.Linear(in_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x=self.batch512(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.act(x)
        x = self.batch128(x)
        x = self.drop(x)

        x = self.fc3(x)
        x = self.act(x)
        x = self.batch64(x)
        x = self.drop(x)

        x = self.fc4(x)
        x = self.batch2(x)
        x = self.drop(x)
        return self.sigmoid2(x)

import torch
import torch.nn as nn


class CombinedMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,use_kanmlp=True, act_layer=nn.ReLU, drop=0.,
                 grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0,
                 scale_spline=1.0, channel_first=False):
        """
        :param channel_first: if True, use convolutions; else, use linear layers.
        """
        super().__init__()
        self.use_kanmlp=use_kanmlp
        self.mlp = MLP(in_features, hidden_features, out_features, act_layer, drop, channel_first)
        self.kanmlp=KAN([in_features, 512,256,128, 64, 32,16, out_features])

    def forward(self, x):

        if self.use_kanmlp:
            return  self.kanmlp(x)  #(self.kanmlp(x)+self.mlp(x))*0.5
        else:
            return self.mlp(x)






