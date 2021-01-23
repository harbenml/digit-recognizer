from data_io import get_data
from dataloader import get_dataloaders
from dataloader import preprocess
from dataloader import WrappedDataLoader
from loss import loss_batch
from model import Mnist_CNN
from typing import Any
from typing import Callable

import numpy as np  # type: ignore
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader


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

    model = Mnist_CNN()

    if torch.cuda.is_available():
        print("Training uses GPU")
        device = torch.device("cuda")
        model.to(device)

    bs = 32
    lr = 0.1
    momentum = 0.9
    epochs = 10

    opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss_func = F.cross_entropy

    train_ds, valid_ds = get_data()
    train_dl = WrappedDataLoader(train_ds, preprocess)
    valid_dl = WrappedDataLoader(valid_ds, preprocess)

    print("Start training")
    model = fit(
        epochs=epochs,
        model=model,
        loss_func=loss_func,
        opt=opt,
        train_dl=train_dl,
        valid_dl=valid_dl,
    )
