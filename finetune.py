import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import shutil
import tensorboardX

from trainer import Trainer
from l0module import L0Linear, L0Conv2d
from prunemodule import PruneL0Conv2d, PruneL0Linear
from prunemodule import PruneLinear, PruneConv2d

from data import get_loader
from models import L0Net, LeNet

from config import *

def weight_init(m):
    if isinstance(m, PruneLinear):
        m.reset_parameters()
    if isinstance(m, PruneConv2d):
        m.reset_parameters()

# def main(epochs, batch_size, coef, mean, baseline):
def main(epochs, batch_size, pt, reinitialize):
    train_loader, test_loader = get_loader(name=pt2data(pt), batch_size=batch_size)
    logdir = finetune_logger_dir(pt)
    if reinitialize:
        logdir += "_reinitialize"
    if "random" in pt:
        logdir += "_random" + pt.split("_random")[-1].split(".")[0]
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    logger = tensorboardX.SummaryWriter(logdir)

    model = pt2stru(pt)
    model.load_state_dict(torch.load(pt))
    if reinitialize:
        model.apply(weight_init)
    model.cuda()

    for m in model.modules():
        if hasattr(m, "mask"): m.prune = True

    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
    # l0_loss = lambda output, target: F.cross_entropy(output[0], target) + coef / len(train_loader.dataset) * output[1]
    # loss_f = F.cross_entropy if baseline else l0_loss
    l0_loss = lambda output, target: F.cross_entropy(output[0], target)
    loss_f = F.cross_entropy if "l0" not in pt else l0_loss
    trainer = Trainer(model, optimizer, loss_f, logger)
    if "l0" in pt:
        correct = lambda output, target: (output[0].max(dim=1)[1] == target).sum().item()
        trainer.correct = correct

    trainer.start(epochs, train_loader, test_loader)

    name = finetune_checkpoint(pt)
    torch.save(model.state_dict(), name)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=g_epochs)
    # parser.add_argument("--coef", type=float, default=1e-1)
    # parser.add_argument("--mean", type=float, default=1)
    # parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--pt", type=str, default="pt/lenet_p90.pt")
    parser.add_argument("--reinitialize", action="store_true", default=False)

    main(**vars(parser.parse_args()))
