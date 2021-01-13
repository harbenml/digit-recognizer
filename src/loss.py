from typing import Any, Callable, Tuple

from torch import nn
from torch import optim
from torch import Tensor

# from torch.utils.data import DataLoader
# import torch.nn.functional as F

from model import Mnist_CNN


def loss_batch(
    xb: Tensor, yb: Tensor, model: Callable, loss_func: Callable, opt: Any = None,
) -> Tuple[Any, int]:
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

