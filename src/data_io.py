from typing import cast
from typing import IO
from typing import Tuple

from config import DATA_PATH
from config import FILENAME
from config import URL

import gzip
import numpy as np  # type: ignore
import pickle
import requests

import torch
from torch.utils.data import TensorDataset


def get_data() -> Tuple[TensorDataset, TensorDataset]:
    """
    Downloads the MNIST dataset from github, if not already done, and 
    unzips the file. The function returns the training set and the 
    validation set.
    """
    if not (DATA_PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (DATA_PATH / FILENAME).open("wb").write(content)

    with gzip.open((DATA_PATH / FILENAME).as_posix(), "rb") as unzipped_file:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(
            cast(IO[bytes], unzipped_file), encoding="latin-1"
        )

    x_train, y_train, x_valid, y_valid = map(
        torch.tensor, (x_train, y_train, x_valid, y_valid)
    )
    train_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_valid, y_valid)

    return train_ds, valid_ds

