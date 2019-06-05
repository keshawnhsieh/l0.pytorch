import torch
from torch import nn
from torch.nn import functional as F

from l0module import L0Linear, L0Conv2d
from l0module import hard_sigmoid


class PruneL0Linear(L0Linear):
    def __init__(self, in_features, out_features, bias=True, prune=False, **kwargs):
        super(PruneL0Linear, self).__init__(in_features, out_features, bias, **kwargs)
        self.prune = prune

        self.register_buffer("mask", torch.zeros(self._size))

    def forward(self, input):
        if not self.prune:
            mask, penalty = self._get_mask()
            self.mask = mask
            return F.linear(input, self._origin.weight * mask, self._origin.bias), penalty
        else:
            return F.linear(input, self._origin.weight * self.mask, self._origin.bias), .0

class PruneL0Conv2d(L0Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, prune=False,
                **kwargs):
        super(PruneL0Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                            dilation=dilation, groups=groups, bias=bias, **kwargs)
        self.prune = prune

        self.register_buffer("mask", torch.zeros(self._size))

    def forward(self, input):
        if not self.prune:
            mask, penalty = self._get_mask()
            self.mask = mask
            conv = F.conv2d(input, self._origin.weight * mask, self._origin.bias, stride=self._origin.stride,
                            padding=self._origin.padding, dilation=self._origin.dilation, groups=self._origin.groups)
            return conv, penalty
        else:
            conv = F.conv2d(input, self._origin.weight * self.mask, self._origin.bias, stride=self._origin.stride,
                            padding=self._origin.padding, dilation=self._origin.dilation, groups=self._origin.groups)
            return conv, .0

class PruneConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, prune=False):
        super(PruneConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                          dilation=dilation, groups=groups, bias=bias)
        self.prune = prune
        self.register_buffer("mask", torch.zeros(self.weight.size()))

    def forward(self, input):
        if self.prune:
            return F.conv2d(input, self.weight * self.mask, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

class PruneLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, prune=False):
        super(PruneLinear, self).__init__(in_features, out_features, bias=bias)
        self.prune = prune
        self.register_buffer("mask", torch.zeros(self.weight.size()))

    def forward(self, input):
        if self.prune:
            return F.linear(input, self.weight * self.mask, self.bias)
        else:
            return F.linear(input, self.weight, self.bias)
