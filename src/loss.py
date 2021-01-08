from typing import Any, Callable, Tuple

from torch import nn
from torch import optim
from torch import Tensor

from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.model import Mnist_CNN

loss_func = F.cross_entropy
model = Mnist_CNN()


def loss_batch(
    xb: Tensor,
    yb: Tensor,
    opt: Any = None,
    model: Callable = model,
    loss_func: Callable = loss_func,
) -> Tuple[Any, int]:
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

