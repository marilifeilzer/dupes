#!/usr/bin/env python3

from dupes.data.gc_client import load_table_to_df
from dupes.model.model_chromadb import create_ingr_db

def recreate_ingredients_collection():
    """Recreate the ingredients ChromaDB collection from scratch"""
    print("Recreating ingredients ChromaDB collection...")

    df = load_table_to_df()
    print(f"Loaded {len(df)} products")

    # This will recreate the collection from scratch
    create_ingr_db(df)
    print("✅ Successfully recreated ingredients ChromaDB collection")

if __name__ == "__main__":
    recreate_ingredients_collection()