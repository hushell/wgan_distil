import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels,
        kernel_size=3, padding=1,
        bias=False, stride=stride)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels,
        kernel_size=1, padding=0,
        bias=False, stride=stride)

class Conv1x1Regressor(nn.Module):
    def __init__(self, planes):
        super(Conv1x1Regressor, self).__init__()
        depth = len(planes)-1
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.extend([
                nn.BatchNorm2d(planes[i]),
                nn.ReLU(inplace=True)])
            self.layers.append(conv1x1(planes[i], planes[i+1]))

    def forward(self, input):
        out = input
        for layer in self.layers:
            out = layer(out)

        return out

class LinearRegressor(nn.Module):
    def __init__(self, sizes):
        super(LinearRegressor, self).__init__()
        depth = len(sizes)-1

        self.layers = nn.ModuleList([nn.Linear(sizes[0], sizes[1])])
        for i in range(1, depth):
            self.layers.extend([
                nn.BatchNorm1d(sizes[i]),
                nn.ReLU(inplace=True),
                nn.Linear(sizes[i], sizes[i+1])])

    def forward(self, input):
        out = input.view(input.size(0), -1)
        for layer in self.layers:
            out = layer(out)

        return out
