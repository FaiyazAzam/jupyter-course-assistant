"""
build_course_memory.py

Instructor helper script to (re)build the course knowledge index from PDFs.

Usage:
    python build_course_memory.py

This will:
  - Read all PDFs under ./course_materials/ (recursively)
  - Build a VectorStoreIndex using OpenAI embeddings
  - Persist it to ./course_index/
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

load_dotenv()

COURSE_DIR = Path("course_materials")
PERSIST_DIR = Path("course_index")


def build_course_memory():
    if not COURSE_DIR.exists():
        raise FileNotFoundError(
            f"Course materials directory not found: {COURSE_DIR}. "
            f"Create it and add PDFs before running this script."
        )

    print(f"[Build] Loading documents from: {COURSE_DIR}")
    docs = SimpleDirectoryReader(str(COURSE_DIR), recursive=True).load_data()
    print(f"[Build] Loaded {len(docs)} documents.")

    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    Settings.embed_model = embed_model

    # Build index + persist
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
    )

    PERSIST_DIR.mkdir(exist_ok=True)
    storage_context.persist(persist_dir=str(PERSIST_DIR))

    print(f"[Build] Course index built and persisted to: {PERSIST_DIR.resolve()}")


if __name__ == "__main__":
    build_course_memory()
