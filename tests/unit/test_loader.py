import pytest
import numpy as np
from omegaconf import DictConfig

from churnops.data.loader import load_dataset


def test_load_dataset(tmp_path):
    X_train = np.ones((10, 3))
    y_train = np.ones((10))

    X_val_test = np.zeros((4, 3))
    y_val_test = np.zeros((4))

    tmp_dir = tmp_path / "data_dir"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(str(tmp_dir / "train.npz"), X=X_train, y=y_train)
    np.savez_compressed(str(tmp_dir / "val.npz"), X=X_val_test, y=y_val_test)
    np.savez_compressed(str(tmp_dir / "test.npz"), X=X_val_test, y=y_val_test)

    cfg = DictConfig(
        {
            "processed_file_type": "npz",
            "preprocessed_dir": str(tmp_dir),
            "train_file": "train.npz",
            "val_file": "val.npz",
            "test_file": "test.npz",
        }
    )

    l_X_train, l_y_train, l_X_val, l_y_val, l_X_test, l_y_test = load_dataset(cfg)

    assert np.all(l_X_train == X_train)
    assert np.all(l_y_train == y_train)
    assert np.all(l_X_val == X_val_test)
    assert np.all(l_y_val == y_val_test)
    assert np.all(l_X_test == X_val_test)
    assert np.all(l_y_test == y_val_test)

    cfg["processed_file_type"] = "jpg"

    with pytest.raises(AttributeError, match="Unknown `processed_file_type`"):
        load_dataset(cfg)
