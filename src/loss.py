from typing import Iterable, Callable

from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.model import Mnist_CNN

loss_func = F.cross_entropy


def loss_batch(
    model: Mnist_CNN = Mnist_CNN(), loss_func: Callable = loss_func, xb: Iterable, yb: Iterable, opt: optim = None
):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

