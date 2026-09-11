import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from xgboost import XGBClassifier
from omegaconf import DictConfig
import mlflow
from sklearn.metrics import roc_auc_score

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
    excecution_mode = "mutlirun" if is_sweep else "single_run"
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


@hydra.main(version_base=None, config_path="../../../configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


def run_training(cfg: DictConfig) -> float:
    run_name, excecution_mode, sweep_id = setup_mlflow(cfg)

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("execution_mode", excecution_mode)
        mlflow.set_tag("sweep_group", sweep_id)

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
        roc_auc = roc_auc_score(y_val, preds)
        print(f"ROC-AUC: {roc_auc:.4f}")

        return roc_auc


if __name__ == "__main__":
    main()
