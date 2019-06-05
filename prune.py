import argparse
import numpy as np
import os
import torch
import shutil
from torchvision import datasets, transforms
from torch.nn import functional as F
import torch.optim as optim

from trainer import Trainer
from models import L0Net, LeNet

import tensorboardX

from data import get_loader
from config import *
import string, random

def tag(length=6):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))

parser = argparse.ArgumentParser()
parser.add_argument("--rate", type=int, default=90, help="how much to prune? give number range 0 to 100")
parser.add_argument("--pt", type=str, default="pt/VGG16.pt")
parser.add_argument("--rand", action="store_true", default=False)

args = parser.parse_args()
rate = args.rate
pt = args.pt
rand = args.rand
tag = tag()

model = pt2stru(pt)
model.load_state_dict(torch.load(pt))

all_weights = []
for m in model.modules():
    if hasattr(m, "mask"):
        if "l0" in pt:
            all_weights += list(m.mask.cpu().abs().numpy().flatten())
        else:
            all_weights += list(m.weight.cpu().data.abs().numpy().flatten())
threshold = np.percentile(np.array(all_weights), rate)

for m in model.modules():
    if hasattr(m, "mask"):
        if "l0" in pt:
            keep = m.mask.abs() > threshold
            m.mask *= keep.to(torch.float32)
        else:
            keep = m.weight.data.abs() > threshold
            if rand:
                idx = torch.randperm(keep.nelement())
                keep = keep.view(-1)[idx].view(keep.size())
            m.mask = keep.to(torch.float32)
        m.prune = True

_, test_loader = get_loader(name=pt2data(pt), batch_size=128)
logdir = pruning_logger_dir(pt, rate)
if rand:
    logdir += "_random<{tag}>".format(tag=tag)
if os.path.exists(logdir):
    shutil.rmtree(logdir)
logger = tensorboardX.SummaryWriter(logdir)

l0loss = lambda output, target: F.cross_entropy(output[0], target)
loss_f = F.cross_entropy if "l0" not in pt else l0loss
optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
trainer = Trainer(model, optimizer, loss_f, logger)
if "l0" in pt:
    correct = lambda output, target: (output[0].max(dim=1)[1] == target).sum().item()
    trainer.correct = correct

trainer.test(test_loader)
for name, module in model.named_modules():
    if hasattr(module, "mask"): logger.add_histogram("test/" + name + ".mask", module.mask, 0, bins="sqrt")
logger.close()

save_name = pruning_checkpoint(pt, rate)
if rand:
    save_name = save_name.split(".")[0] + "_random<{tag}>.pt".format(tag=tag)
torch.save(model.state_dict(), save_name)


