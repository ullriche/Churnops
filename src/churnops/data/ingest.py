import argparse
from pathlib import Path
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from churnops.data.schemas import RawChurnSchema
import os


def ingest_data(source_url: str, is_kaggle: bool) -> pd.DataFrame:
    print(f"Fetching data from {source_url}...")
    if is_kaggle:
        handle = os.path.dirname(source_url)
        path = os.path.basename(source_url)
        df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, handle, path)
    else:
        df = pd.read_csv(source_url)

    if df.empty:
        raise ValueError("The fetched data is empty. Please check the source URL.")

    # Turn spaces (" ") in "TotalCharges" into NaN, then convert column to float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)  # Fill NaN values with 0

    return df


def save_data(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )  # Ensure the output directory exists

    df.to_parquet(
        output_path, index=False
    )  # Apache Parquet: A binary columnar storage format optimized for use with data frames.
    print(f"Data successfully ingested and saved to {output_path} [{len(df)} rows].")


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest data for churn analysis.")
    parser.add_argument(
        "--source", type=str, required=True, help="Path to the input data file."
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Path to save the ingested data."
    )
    parser.add_argument(
        "--kaggle",
        action="store_true",
        help="Flag to indicate if the source is a Kaggle dataset. If set, the source should be in the format 'username/dataset-name'.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()
    df = ingest_data(args.source, args.kaggle)
    validated_df = RawChurnSchema.validate(
        df
    )  # Validate the DataFrame against the schema
    save_data(validated_df, args.output)
