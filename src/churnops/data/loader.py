from omegaconf import DictConfig
import os
import numpy as np


def load_dataset(data_config: DictConfig):
    def load_npz(filepath: str) -> tuple[np.ndarray, np.ndarray]:
        with open(filepath, "rb") as f:
            loaded = np.load(f)
            return loaded["X"], loaded["y"]

    if data_config.processed_file_type == "npz":
        X_train, y_train = load_npz(
            os.path.join(data_config.preprocessed_dir, data_config.train_file)
        )
        X_val, y_val = load_npz(
            os.path.join(data_config.preprocessed_dir, data_config.val_file)
        )
        X_test, y_test = load_npz(
            os.path.join(data_config.preprocessed_dir, data_config.test_file)
        )

        return X_train, y_train, X_val, y_val, X_test, y_test
    else:
        raise AttributeError(
            f"Unknown `processed_file_type` ({data_config.processed_file_type}) in data config!"
        )
