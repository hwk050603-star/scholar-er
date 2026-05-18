__all__ = [
    "SparseRetriever",
    "SearchEngine",
    "set_base_path",
]

import os
from pathlib import Path

from .sparse_retriever.sparse_retriever import SparseRetriever
from .sparse_retriever.sparse_retriever import SparseRetriever as SearchEngine

# Set environment variables ----------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["RETRIV_BASE_PATH"] = str(Path.home() / ".retriv")


def set_base_path(path: str):
    os.environ["RETRIV_BASE_PATH"] = path
