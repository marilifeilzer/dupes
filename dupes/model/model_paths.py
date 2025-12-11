"""
Simple centralized model paths and utilities.
Fixes naming inconsistencies and path issues across the codebase.
"""
import os
from pathlib import Path

# Base cache directory
CACHE_ROOT = Path(os.getenv("MODELS_CACHE_DIR", "models_cache"))

# Model directories
PRICE_DIR = CACHE_ROOT / "price"
INGREDIENTS_DIR = CACHE_ROOT / "ingredients"
DESCRIPTIONS_DIR = CACHE_ROOT / "descriptions"

# Price model paths
PRICE_MODEL_BASE = PRICE_DIR / "xgb_best.pkl"
PRICE_MODEL_MANU = PRICE_DIR / "xgb_best_manu.pkl"
PRICE_META_BASE = PRICE_DIR / "xgb_meta.pkl"  # Move from root to price dir
PRICE_META_MANU = PRICE_DIR / "xgb_meta_manu.pkl"  # Move from root to price dir

# MLB ingredients path (renamed from model.pkl)
MLB_INGREDIENTS_PATH = INGREDIENTS_DIR / "mlb_ingredients.pkl"

# ChromaDB paths (these already work correctly)
CHROMA_INGREDIENTS_DIR = INGREDIENTS_DIR / "chroma"
CHROMA_INGREDIENTS_ARCHIVE = INGREDIENTS_DIR / "chroma.tar.gz"
CHROMA_DESCRIPTIONS_DIR = DESCRIPTIONS_DIR / "chroma"
CHROMA_DESCRIPTIONS_ARCHIVE = DESCRIPTIONS_DIR / "chroma.tar.gz"

# GCS blob names
GCS_PRICE_BLOB = os.getenv("PRICE_MODEL_BLOB", "price/xgb_best.pkl")
GCS_PRICE_MANU_BLOB = os.getenv("PRICE_MODEL_MANU_BLOB", "price/xgb_best_manu.pkl")
GCS_PRICE_META_BLOB = os.getenv("PRICE_META_BLOB", "price/xgb_meta.pkl")
GCS_PRICE_META_MANU_BLOB = os.getenv("PRICE_META_MANU_BLOB", "price/xgb_meta_manu.pkl")
GCS_MLB_BLOB = os.getenv("INGREDIENTS_MLB_BLOB", "ingredients/mlb_ingredients.pkl")
GCS_CHROMA_INGREDIENTS_BLOB = os.getenv("INGREDIENTS_CHROMA_BLOB", "ingredients/chroma.tar.gz")
GCS_CHROMA_DESCRIPTIONS_BLOB = os.getenv("DESCRIPTION_CHROMA_BLOB", "descriptions/chroma.tar.gz")


def ensure_model_dirs():
    """Ensure all model directories exist."""
    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    INGREDIENTS_DIR.mkdir(parents=True, exist_ok=True)
    DESCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)


def get_price_model_path(manufacturer: bool = False) -> Path:
    """Get path for price model."""
    return PRICE_MODEL_MANU if manufacturer else PRICE_MODEL_BASE


def get_price_meta_path(manufacturer: bool = False) -> Path:
    """Get path for price meta model."""
    return PRICE_META_MANU if manufacturer else PRICE_META_BASE


def get_price_gcs_blob(manufacturer: bool = False) -> str:
    """Get GCS blob name for price model."""
    return GCS_PRICE_MANU_BLOB if manufacturer else GCS_PRICE_BLOB


def get_price_meta_gcs_blob(manufacturer: bool = False) -> str:
    """Get GCS blob name for price meta model."""
    return GCS_PRICE_META_MANU_BLOB if manufacturer else GCS_PRICE_META_BLOB