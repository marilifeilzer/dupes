"""
Orchestrates a full training run:
- description embeddings → upload Chroma archive
- ingredients embeddings/MLB → upload artifacts
- price model → upload pickle

Skips training if artifacts already exist unless FORCE_RETRAIN=true.
Set TRAIN_AT_START=false to bypass training entirely (but still ensure downloads).
"""

import os
from pathlib import Path

from dupes.data.gc_client import load_table_to_df
import dupes.model.descriptions_chromadb as desc
import dupes.model.model_chromadb as ingr
from dupes.model.price_prediction import (
    ensure_price_model,
    preprocess_data,
    save_price_model,
    train_model,
)
from dupes.model.model_paths import PRICE_MODEL_BASE


def _artifacts_present() -> bool:
    ingredients_ready = ingr.MLB_PATH.is_file() and (ingr.CHROMA_DIR / "chroma.sqlite3").is_file()
    descriptions_ready = (desc.CHROMA_DIR / "chroma.sqlite3").is_file()
    price_ready = PRICE_MODEL_BASE.is_file()
    return ingredients_ready and descriptions_ready and price_ready


def main() -> None:
    train_enabled = os.getenv("TRAIN_AT_START", "true").lower() not in {"0", "false", "no"}
    force_retrain = os.getenv("FORCE_RETRAIN", "false").lower() in {"1", "true", "yes"}

    if not train_enabled:
        print("TRAIN_AT_START disabled; ensuring downloads only.")
        desc.ensure_description_artifacts()
        ingr.ensure_ingredients_artifacts()
        ensure_price_model()
        return

    if not force_retrain and _artifacts_present():
        print("Artifacts found; skipping training. Set FORCE_RETRAIN=true to rebuild.")
        return

    print("Loading data from BigQuery...")
    df = load_table_to_df()

    print("Training description embeddings...")
    desc.embedding_description_get_recommendation()

    print("Training ingredients embeddings...")
    ingr.create_ingr_db(df)

    print("Training price model...")
    preprocess = preprocess_data(df)
    model = train_model(preprocess)
    save_price_model(model)

    print("Training and uploads complete.")


if __name__ == "__main__":
    main()
