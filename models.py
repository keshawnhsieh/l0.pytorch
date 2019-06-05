import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import tensorboardX

from trainer import Trainer
from l0module import L0Linear, L0Conv2d
from prunemodule import PruneL0Conv2d, PruneL0Linear
from prunemodule import PruneLinear, PruneConv2d
from l0module import L0Sequential

class L0Net(nn.Module):
    """
    based on Caffe LeNet (https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt)
    """

    def __init__(self, mean, temp):
        super(L0Net, self).__init__()
        # self.conv1 = L0Conv2d(1, 20, kernel_size=5, stride=1, loc_mean=mean)
        # self.conv2 = L0Conv2d(20, 50, kernel_size=5, stride=1, loc_mean=mean)
        # self.dense1 = L0Linear(800, 500, loc_mean=mean)
        # self.dense2 = L0Linear(500, 10, loc_mean=mean)

        self.conv1 = PruneL0Conv2d(1, 20, kernel_size=5, stride=1, loc_mean=mean, beta=temp)
        self.conv2 = PruneL0Conv2d(20, 50, kernel_size=5, stride=1, loc_mean=mean, beta=temp)
        self.dense1 = PruneL0Linear(800, 500, loc_mean=mean, beta=temp)
        self.dense2 = PruneL0Linear(500, 10, loc_mean=mean, beta=temp)

    def forward(self, x):
        x, z1 = self.conv1(x)
        x = F.max_pool2d(x, 2, stride=2)
        x, z2 = self.conv2(x)
        x = F.max_pool2d(x, 2, stride=2)
        x = x.view(x.shape[0], -1)
        x, z3 = self.dense1(x)
        x = F.relu(x)
        x, z4 = self.dense2(x)
        penalty = z1 + z2 + z3 + z4
        return x, penalty


class LeNet(nn.Module):
    """
    based on Caffe LeNet (https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt)
    """

    def __init__(self):
        super(LeNet, self).__init__()
        # self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1)
        # self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1)
        # self.dense1 = nn.Linear(800, 500)
        # self.dense2 = nn.Linear(500, 10)
        self.conv1 = PruneConv2d(1, 20, kernel_size=5, stride=1)
        self.conv2 = PruneConv2d(20, 50, kernel_size=5, stride=1)
        self.dense1 = PruneLinear(800, 500)
        self.dense2 = PruneLinear(500, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, stride=2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, stride=2)
        x = x.view(x.shape[0], -1)
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        return x

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
class VGG(nn.Module):
    def __init__(self, network):
        super(VGG, self).__init__()
        layers = []
        in_channels = 3
        for x in cfg.get(network):
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [PruneConv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                        nn.ReLU(inplace=True)]
                in_channels = x
        self.features = nn.Sequential(*layers)
        self.dense1 = PruneLinear(512, 128)
        self.dense2 = PruneLinear(128, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        return x

class L0VGG(nn.Module):
    def __init__(self, network, loc, temp):
        super(L0VGG, self).__init__()
        layers = []
        in_channels = 3
        for x in cfg.get(network):
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [PruneL0Conv2d(in_channels, x, kernel_size=3, padding=1, loc_mean=loc, beta=temp),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        self.features = L0Sequential(*layers)
        self.dense1 = PruneL0Linear(512, 128, loc_mean=loc, beta=temp)
        self.dense2 = PruneL0Linear(128, 10, loc_mean=loc, beta=temp)

    def forward(self, x):
        x, z0 = self.features(x)
        x = x.view(x.shape[0], -1)
        x, z1 = self.dense1(x)
        x = F.relu(x)
        x, z2 = self.dense2(x)
        return x, z0 + z1 + z2
