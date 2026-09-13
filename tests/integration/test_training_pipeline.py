import hydra
import numpy as np

from churnops.training.train import run_training
from churnops.training import train


def test_run_training(tmp_path, mocker):
    # Data
    X_train = np.ones((20, 3))
    y_train = np.array([0] * 10 + [1] * 10)

    X_val = np.zeros((4, 3))
    y_val = np.array([0] * 2 + [1] * 2)
    X_test = np.zeros((4, 3))
    y_test = np.array([0] * 2 + [1] * 2)

    data_dir = tmp_path / "data_dir"
    data_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(str(data_dir / "train.npz"), X=X_train, y=y_train)
    np.savez_compressed(str(data_dir / "val.npz"), X=X_val, y=y_val)
    np.savez_compressed(str(data_dir / "test.npz"), X=X_test, y=y_test)

    # Mocks
    spy_predict_proba = mocker.spy(train.XGBClassifier, "predict_proba")
    spy_log_metrics = mocker.spy(train.mlflow, "log_metrics")
    spy_log_figure = mocker.spy(train.mlflow, "log_figure")

    mock_hydra_cfg = mocker.MagicMock()
    mock_hydra_cfg.mode.name = "RUN"
    mock_hydra_cfg.sweep.dir = "test-sweep"
    mocker.patch("churnops.training.train.HydraConfig.get", return_value=mock_hydra_cfg)

    # Config
    with hydra.initialize(config_path="../../configs", version_base=None):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                "training.epochs=2",
                f"data.preprocessed_dir={data_dir}",
                f"mlflow.tracking_uri=sqlite:///{tmp_path}/mlflow.db",
                "mlflow.artifact_location=null",
            ],
        )
        # HydraConfig.instance().set_config(cfg)
        run_training(cfg)

    spy_predict_proba.assert_called_once()

    preds = spy_predict_proba.spy_return
    assert len(preds) == len(y_val)
    assert np.all((preds >= 0) & (preds <= 1))

    spy_log_metrics.assert_called_once()
    logged_metrics = spy_log_metrics.call_args.args[0]

    metric_names = ["roc_auc", "precision", "recall", "f1", "accuracy"]
    assert all([metric in logged_metrics for metric in metric_names])
    assert all([logged_metrics[metric] is not None for metric in metric_names])

    spy_log_figure.assert_called_once()
    fig, artifact_path = spy_log_figure.call_args.args
    assert fig is not None
    assert artifact_path == "plots/confusion_matrix.png"
