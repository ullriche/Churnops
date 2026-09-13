import hydra
from hydra.core.hydra_config import HydraConfig, OmegaConf
from hydra.types import RunMode
from collections.abc import MutableMapping
from xgboost import XGBClassifier
from omegaconf import DictConfig
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    ConfusionMatrixDisplay,
)
import numpy as np
import warnings
from collections import defaultdict
from typing import Any

from churnops.data.loader import load_dataset


def setup_mlflow(cfg: DictConfig) -> tuple[str, str, str]:
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(cfg.mlflow.experiment_name)
    if not exp:
        client.create_experiment(
            name=cfg.mlflow.experiment_name,
            artifact_location=cfg.mlflow.artifact_location,
        )

    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # Determine Run Name and tags
    hydra_cfg = HydraConfig.get()
    is_sweep = hydra_cfg.mode == RunMode.MULTIRUN
    excecution_mode = "multirun" if is_sweep else "single_run"
    sweep_id = hydra_cfg.sweep.dir if is_sweep else "single-run"

    if is_sweep:
        job_num = hydra_cfg.job.num
        run_name = f"{cfg.model.name}-trial_{job_num}"
    else:
        run_name = f"{cfg.model.name}-baseline"

    should_log_models = (
        not is_sweep if cfg.mlflow.log_models == "auto" else cfg.mlflow.log_models
    )
    should_log_dataset = (
        not is_sweep if cfg.mlflow.log_datasets == "auto" else cfg.mlflow.log_datasets
    )

    mlflow.xgboost.autolog(
        log_models=should_log_models,
        log_input_examples=cfg.mlflow.log_input_examples,
        log_model_signatures=cfg.mlflow.log_model_signatures,
        log_datasets=should_log_dataset,
    )

    return run_name, excecution_mode, sweep_id


def flatten_dict(
    cfg: MutableMapping[Any, Any], parent_key: str = "", sep: str = "."
) -> dict[str, Any]:
    items: list[tuple[str, Any]] = []

    for k, v in cfg.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(cfg=v, parent_key=new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    # Check for Key Collisions (multiple values get mapped to the same key)
    read_keys = defaultdict(list)  # Dict with default element being an empty list
    for item in items:
        read_keys[item[0]].append(item[1])

    for key in read_keys:
        if len(read_keys[key]) > 1:
            warnings.warn(
                f"Key Collision! Multiple entries are mapped to the same key. Key: {key}, Values: {read_keys[key]}",
                UserWarning,
            )

    return dict(items)


def calc_and_log_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> None:
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    mlflow.log_metrics(metrics)

    calc_and_log_confusion_matrix(y_true, y_pred)


def calc_and_log_confusion_matrix(y_true, y_pred) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        display_labels=["No Churn", "Churn"],
        cmap="Blues",
        normalize="true",
        ax=ax,
    )
    ax.set_title("Validation Confusion Matrix")
    plt.tight_layout()

    mlflow.log_figure(fig, "plots/confusion_matrix.png")

    plt.close()


@hydra.main(version_base=None, config_path="../../../configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


def run_training(cfg: DictConfig) -> None:
    run_name, excecution_mode, sweep_id = setup_mlflow(cfg)

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("execution_mode", excecution_mode)
        mlflow.set_tag("sweep_group", sweep_id)

        # # Convert config to dict
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(config_dict, MutableMapping):
            raise TypeError(f"Expected config to be a mapping, got {type(config_dict)}")

        mlflow.log_params(flatten_dict(config_dict))

        model = XGBClassifier(
            n_estimators=cfg.training.epochs,
            learning_rate=cfg.training.learning_rate,
            max_depth=cfg.model.max_depth,
            subsample=cfg.model.subsample,
            colsample_bytree=cfg.model.colsample_bytree,
            reg_lambda=cfg.model.reg_lambda,
            early_stopping_rounds=cfg.training.early_stopping_rounds,  # Halts if eval metric doesn't improve
            eval_metric=cfg.training.eval_metric,
            tree_method=cfg.model.tree_method,  # Fast histogram-based split finding
            device=cfg.general.device,  # Use "cuda" for GPU, "cpu" for CPU
            random_state=cfg.general.random_state,
        )

        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(cfg.data)

        print(
            f"Starting training of '{cfg.model.name}' with dataset '{cfg.data.name}'..."
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

        preds = model.predict_proba(X_val)[:, 1]

        calc_and_log_metrics(y_val, preds)


if __name__ == "__main__":
    main()
