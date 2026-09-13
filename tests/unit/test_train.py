import pytest
import numpy as np
from matplotlib.figure import Figure

from churnops.training.train import flatten_dict, calc_and_log_metrics


def test_flatten_dict():
    d = {
        "value1": {"value2": 2, "value3": {"value4": 4}},
        "value5": 5,
    }

    flattened_dict = flatten_dict(d)

    assert flattened_dict == {
        "value1.value2": 2,
        "value1.value3.value4": 4,
        "value5": 5,
    }

    d = {
        "value1": {"value2": 5},  # Will be mapped to key `value1.value2`
        "value1.value2": 10,  # Will be mapped to key `value1.value2`
    }

    with pytest.warns(UserWarning, match="Key Collision!"):
        flatten_dict(d)


def test_calc_and_log_metrics(mocker):
    mock_log_metrics = mocker.patch("churnops.training.train.mlflow.log_metrics")
    mock_log_figure = mocker.patch("churnops.training.train.mlflow.log_figure")

    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])

    calc_and_log_metrics(y_true, y_prob, threshold=0.5)

    mock_log_metrics.assert_called_once()
    logged_metrics = mock_log_metrics.call_args[0][0]

    assert logged_metrics["accuracy"] == 1.0
    assert logged_metrics["f1"] == 1
    assert "roc_auc" in logged_metrics

    mock_log_figure.assert_called_once()
    logged_fig, logged_artifact_path = mock_log_figure.call_args[0]

    assert isinstance(logged_fig, Figure)
    assert logged_artifact_path == "plots/confusion_matrix.png"

    # Test `zero_division=0`
    y_true = np.array([0, 0, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.2])

    _ = mocker.patch("churnops.training.train.calc_and_log_confusion_matrix")

    calc_and_log_metrics(y_true, y_prob)

    logged_metrics = mock_log_metrics.call_args[0][0]
    assert logged_metrics["precision"] == 0
    assert logged_metrics["recall"] == 0
