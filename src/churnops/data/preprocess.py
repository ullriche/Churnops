import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def read_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess data for churn analysis.")

    parser.add_argument("--input", type=Path, help="Path to the (raw) input data file.")
    parser.add_argument(
        "--output", type=Path, help="Directory to save the preprocessed data to."
    )
    parser.add_argument(
        "--split",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        help="Train/validation/test split ratios. Default is [0.8, 0.1, 0.1].",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility. Default is 42.",
    )

    return parser.parse_args()


def read_data(input_path: Path) -> pd.DataFrame:
    if not input_path.is_file():
        raise AttributeError(f"The input path {input_path} must be a file")

    if not input_path.exists():
        raise FileNotFoundError(f"The input file {input_path} does not exist.")

    df = pd.read_parquet(input_path)
    if df.empty:
        raise ValueError("The input data is empty. Please check the input file.")

    return df


def preprocess_data(df: pd.DataFrame, split: list, random_state: int) -> tuple:
    df = df.drop(columns=["customerID"])
    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    categorical_cols = X.select_dtypes(include=["object", "str"]).columns.to_list()
    numerical_cols = X.select_dtypes(include=["number"]).columns.to_list()

    if len(split) != 3 or not np.isclose(sum(split), 1.0):
        raise ValueError(
            "Split ratios must be a list of three numbers that sum to 1.0."
        )

    train_size, val_size, test_size = split

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=val_size + test_size, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_size / (val_size + test_size),
        random_state=random_state,
        stratify=y_temp,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            (
                "cat",
                OneHotEncoder(
                    drop="first", handle_unknown="ignore", sparse_output=False
                ),
                categorical_cols,
            ),
        ]
    )

    X_train_np = preprocessor.fit_transform(X_train)
    y_train_np = y_train.to_numpy()

    X_val_np = preprocessor.transform(X_val)
    y_val_np = y_val.to_numpy()

    X_test_np = preprocessor.transform(X_test)
    y_test_np = y_test.to_numpy()

    return (X_train_np, y_train_np), (X_val_np, y_val_np), (X_test_np, y_test_np)


def save_data(
    train: tuple[np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
    output_path: Path,
) -> None:
    if not output_path.is_dir():
        raise AttributeError(f"The output path {output_path} must be a directory")

    output_path.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(output_path / "train.npz", X=train[0], y=train[1])
    np.savez_compressed(output_path / "val.npz", X=val[0], y=val[1])
    np.savez_compressed(output_path / "test.npz", X=test[0], y=test[1])

    print(f"Preprocessed data saved to {output_path}.")


if __name__ == "__main__":
    args = read_cli_args()
    df = read_data(args.input)
    train, val, test = preprocess_data(df, args.split, args.random_state)
    save_data(train, val, test, args.output)
