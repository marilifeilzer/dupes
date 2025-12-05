"""
Loading/cleaning helpers.

Uses the latest raw dump in source_data and writes a cleaned 
version next to it. Keeping paths relative to the repo root.
"""

from pathlib import Path
import pandas as pd

from dupes.data.clean_data import load_raw_data, clean_data

RAW_FILENAME = "products_data__0412.csv" #Edit later with the set source
CLEAN_FILENAME = "products_data_cleaned.csv" #Workable version


def project_root() -> Path:
    """Return the repository root (two levels up from this file)."""
    return Path(__file__).resolve().parents[2]


def source_data_dir() -> Path:
    return project_root() / "source_data"


def raw_data_path() -> Path:
    return source_data_dir() / RAW_FILENAME


def cleaned_data_path() -> Path:
    return source_data_dir() / CLEAN_FILENAME


def build_clean_dataset(refresh: bool = False) -> pd.DataFrame:
    """
    Load the raw CSV, clean it, and persist the cleaned copy.
    If a cleaned file already exists and refresh is False, reuse it.
    """
    dst = cleaned_data_path()
    if dst.exists() and not refresh:
        return pd.read_csv(dst)

    src = raw_data_path()
    if not src.exists():
        raise FileNotFoundError(f"Raw data file not found: {src}")

    raw_df = load_raw_data(str(src))
    cleaned_df = clean_data(raw_df)

    dst.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(dst, index=False)
    return cleaned_df


def load_clean_dataset() -> pd.DataFrame:
    """Public entrypoint to get the cleaned dataframe (builds it on first call)."""
    return build_clean_dataset(refresh=False)
