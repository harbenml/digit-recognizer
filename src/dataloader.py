from typing import Tuple

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


def get_dataloaders(
    train_ds: TensorDataset, valid_ds: TensorDataset, bs: int
) -> Tuple[DataLoader, DataLoader]:
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


def preprocess(x, y):
    return x.view(-1, 1, 28, 28), y


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))
