from typing import Callable
from typing import Generator
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


def get_dataloaders(
    train_ds: TensorDataset, valid_ds: TensorDataset, bs: int
) -> Tuple[DataLoader, DataLoader]:
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


def preprocess(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return x.view(-1, 1, 28, 28).to(device), y.to(device)


class WrappedDataLoader:
    def __init__(self, dl: DataLoader, func: Callable) -> None:
        self.dl = dl
        self.func = func

    def __len__(self) -> int:
        return len(self.dl)

    def __iter__(self) -> Generator:
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))
