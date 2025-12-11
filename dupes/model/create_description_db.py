import chromadb
from dupes.model.descriptions_chromadb import CHROMA_DIR, create_description_db
from dupes.data.gc_client import load_table_to_df


def check_and_create_description_collection():
    # Check if ChromaDB exists
    if not (CHROMA_DIR / "chroma.sqlite3").exists():
        print("Description ChromaDB not found, creating from scratch...")
        df = load_table_to_df()
        create_description_db(df)
        print("✅ Created description ChromaDB")
        return

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # List all collections
    collections = client.list_collections()
    print(f"Found {len(collections)} collections:")
    for collection in collections:
        print(f"  - {collection.name}")

    # Check if description_embed exists
    collection_names = [c.name for c in collections]

    if "description_embed" not in collection_names:
        print("\nCollection 'description_embed' not found. Creating it...")

        from dupes.model.descriptions_chromadb import (
            embedding_description_get_data,
            embedding_description_embed,
            embedding_description_populate_chromadb,
        )

        df = load_table_to_df()
        dropped_desc = embedding_description_get_data(df)
        print(f"Loaded {len(dropped_desc)} products with descriptions")

        embeddings_desc = embedding_description_embed(dropped_desc)

        collection = embedding_description_populate_chromadb(
            dropped_desc, embeddings_desc, client=client
        )
        print(f"✅ Created collection with {collection.count()} embeddings")
    else:
        print(f"\n✅ Collection 'description_embed' already exists")
        collection = client.get_collection("description_embed")
        print(f"Collection has {collection.count()} embeddings")


if __name__ == "__main__":
    check_and_create_description_collection()
