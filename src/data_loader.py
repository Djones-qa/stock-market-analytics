"""
data_loader.py — Load stock price data, validate schema, save outputs.
"""

import os
import pandas as pd
import yaml


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(file_name, config=None, file_type="csv"):
    if config is None:
        config = load_config()
    file_path = os.path.join(config["data"]["raw_dir"], file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    if file_type == "csv":
        df = pd.read_csv(file_path, parse_dates=[config["data"]["date_column"]])
    elif file_type == "parquet":
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported: {file_type}")
    print(f"Loaded {len(df):,} rows x {len(df.columns)} columns from {file_path}")
    return df


def validate_schema(df, config=None):
    if config is None:
        config = load_config()
    required = [config["data"]["price_column"], config["data"]["date_column"]]
    all_expected = required + config["price_columns"]
    expected_set = set(all_expected)
    actual = set(df.columns)
    missing = expected_set - actual
    result = {"valid": len(missing) == 0, "missing": sorted(missing), "columns": sorted(actual)}
    if missing:
        print(f"WARNING: Missing {len(missing)} columns: {result['missing']}")
    else:
        print("Schema validation passed.")
    return result


def save_processed(df, file_name, config=None):
    if config is None:
        config = load_config()
    out_dir = config["data"]["processed_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, file_name)
    if file_name.endswith(".parquet"):
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows to {out_path}")


def load_multiple_tickers(file_names, config=None):
    """Load and concatenate multiple ticker CSVs."""
    frames = []
    for fn in file_names:
        try:
            df = load_raw_data(fn, config)
            frames.append(df)
        except FileNotFoundError:
            print(f"  Skipping: {fn}")
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        print(f"Combined: {len(combined):,} rows from {len(frames)} files")
        return combined
    return pd.DataFrame()
