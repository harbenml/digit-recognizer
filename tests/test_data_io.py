from src.data_io import get_data
from torch.utils.data import TensorDataset

import numpy as np


def test_get_data():
    train_ds, valid_ds = get_data()
    assert isinstance(train_ds, TensorDataset)
    assert isinstance(valid_ds, TensorDataset)
    assert len(train_ds) == 50_000
    assert len(valid_ds) == 10_000

