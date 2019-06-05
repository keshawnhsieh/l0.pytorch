import torch

import argparse

from models import LeNet, L0Net
from config import *

parser = argparse.ArgumentParser()
parser.add_argument("--pt", type=str, default="pt/l0net.pt")
parser.add_argument("--weight", action="store_true")

args = parser.parse_args()
pt = args.pt
weight = args.weight

model = pt2stru(pt)
model.load_state_dict(torch.load(pt))

total_parameter = 0
total_nonzeros = 0
for name, module in model.named_modules():
    if hasattr(module, "mask"):
        if weight:
            total_parameter += module.weight.numel()
            total_nonzeros += module.weight.nonzero().size(0)
        else:
            total_parameter += module.mask.numel()
            total_nonzeros += module.mask.nonzero().size(0)
        print("layer: {layer}, parameter: {parameter}, kept percentage: {percentage:.4f}%".format(
            layer=name,
            parameter=module.mask.numel() if not weight else module.weight.numel(),
            percentage=float(module.mask.nonzero().size(0)) / module.mask.numel() * 100.0 if not weight else float(module.weight.nonzero().size(0) / module.weight.numel() * 100.0)
        ))
print("total, parameter: {parameter}, kept percentage: {percentage:.4f}%".format(
    parameter=total_parameter,
    percentage=float(total_nonzeros) / total_parameter * 100.0
))
