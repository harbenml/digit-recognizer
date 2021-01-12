from typing import Any
from typing import Callable

from src.data_io import get_data
from src.dataloader import get_dataloaders
from src.loss import loss_batch
from src.model import Mnist_CNN

import numpy as np  # type: ignore

import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F


def fit(
    epochs: int,
    model: Mnist_CNN,
    loss_func: Callable,
    opt: Any,
    train_dl: DataLoader,
    valid_dl: DataLoader,
) -> Mnist_CNN:
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(xb, yb, model, loss_func, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(xb, yb, model, loss_func) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)

    return model


if __name__ == "__main__":

    bs = 32
    lr = 0.1
    momentum = 0.9
    epochs = 2
    model = Mnist_CNN()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss_func = F.cross_entropy

    train_ds, valid_ds = get_data()
    train_dl, valid_dl = get_dataloaders(train_ds, valid_ds, bs=bs)

    print("Start training")
    model = fit(
        epochs=epochs,
        model=model,
        loss_func=loss_func,
        opt=opt,
        train_dl=train_dl,
        valid_dl=valid_dl,
    )
