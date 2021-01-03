from src.data import get_data

import numpy as np


def test_get_data():
    x_train, y_train, x_valid, y_valid = get_data()
    assert np.shape(x_train) == (50_000, 784)
    assert np.shape(y_train) == (50_000,)
    assert np.shape(x_valid) == (10_000, 784)
    assert np.shape(y_valid) == (10_000,)

