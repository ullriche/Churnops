import pytest
import pandas as pd
import numpy as np

from churnops.data.preprocess import preprocess_data


def test_preprocess_data():
    random_state = 42
    df = pd.DataFrame(
        {
            "customerID": [str(i) for i in range(20)],  # Will be removed
            "numericalFeature": list(range(20)),
            "categoricalFeature": ["Yes"] * 10 + ["No"] * 6 + ["Maybe"] * 4,
            "boolFeature": [True] * 10
            + [False] * 10,  # Will be dropped, since bools are not implemented
            "Churn": ["Yes"] * 10 + ["No"] * 10,  # Will be split of
        }
    )

    train, val, test = None, None, None
    with pytest.warns(
        UserWarning, match="Data contains data types that are not supported!"
    ):
        train, val, test = preprocess_data(
            df, [0.8, 0.1, 0.1], random_state=random_state
        )

    X_train, y_train = train
    X_val, y_val = val
    X_test, y_test = test

    assert tuple(X_train.shape) == (16, 3)
    assert tuple(y_train.shape) == (16,)
    assert tuple(X_val.shape) == (2, 3)
    assert tuple(y_val.shape) == (2,)
    assert tuple(X_test.shape) == (2, 3)
    assert tuple(y_test.shape) == (2,)

    # Numerical Feature Standard Scaled
    assert np.isclose(np.mean(X_train[:, 0]), 0)
    assert np.isclose(np.std(X_train[:, 0]), 1)

    # One-Hot Encoded
    assert list(np.unique(X_train[:, 1:2])) == [0, 1]
    assert list(np.unique(y_test)) == [0, 1]


def test_preprocess_data_invalid_input():
    random_state = 42
    df = pd.DataFrame(
        {
            "customerID": [str(i) for i in range(20)],  # Will be removed
            "numericalFeature": list(range(20)),
            "categoricalFeature": ["Yes"] * 10 + ["No"] * 6 + ["Maybe"] * 4,
            "Churn": ["Yes"] * 10 + ["No"] * 10,  # Will be split of
        }
    )

    with pytest.raises(ValueError):
        preprocess_data(df, [0.9, 0.1, 0.1], random_state=random_state)
