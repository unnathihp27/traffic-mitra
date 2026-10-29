"""Build the local FAISS index from data/."""

import argparse

from config import DATA_DIR, EMBEDDING_MODEL_NAME, INDEX_PATH, METADATA_PATH
from rag_pipeline import load_or_build_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild even if unchanged")
    args = parser.parse_args()
    bundle = load_or_build_index(
        DATA_DIR, INDEX_PATH, METADATA_PATH, model_name=EMBEDDING_MODEL_NAME,
        force_rebuild=args.force
    )
    print(f"Knowledge base ready: {len(bundle.chunks)} chunks indexed.")
    print(f"Index: {INDEX_PATH}")


if __name__ == "__main__":
    main()

