from typing import Callable

from data_io import get_data
from dataloader import get_dataloaders
from loss import loss_batch
from model import Mnist_CNN


import numpy as np

import torch
from torch import optim
import torch.nn.functional as F


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
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
    epochs = 10
    model = Mnist_CNN()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss_func = F.cross_entropy

    train_ds, valid_ds = get_data()
    train_dl, valid_dl = get_dataloaders(train_ds, valid_ds, bs=bs)

    model = fit(
        epochs=epochs,
        model=model,
        loss_func=loss_func,
        opt=opt,
        train_dl=train_dl,
        valid_dl=valid_dl,
    )
