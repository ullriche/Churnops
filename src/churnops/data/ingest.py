from pathlib import Path
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from churnops.data.schemas import RawChurnSchema
import os
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../../configs/", config_name="config")
def main(cfg: DictConfig):
    df = ingest_data(cfg.data.source, cfg.data.from_kaggle)
    validated_df = RawChurnSchema.validate(df)
    save_data(validated_df, Path(cfg.data.raw_path))


def ingest_data(source_url: str, is_kaggle: bool) -> pd.DataFrame:
    print(f"Fetching data from {source_url}...")
    if is_kaggle:
        handle = os.path.dirname(source_url)
        path = os.path.basename(source_url)
        df = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, handle, path)
    else:
        df = pd.read_csv(source_url)

    if df.empty:
        raise ValueError("The fetched data is empty. Please check the source URL.")

    # Turn spaces (" ") in "TotalCharges" into NaN, then convert column to float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)  # Fill NaN values with 0

    return df


def save_data(df: pd.DataFrame, output_path: Path) -> None:
    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Apache Parquet: A binary columnar storage format optimized for use with data frames.
    df.to_parquet(output_path, index=False)
    print(f"Data successfully ingested and saved to {output_path} [{len(df)} rows].")


if __name__ == "__main__":
    main()
