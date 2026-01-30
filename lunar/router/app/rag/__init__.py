"""RAG (Retrieval-Augmented Generation) module."""

from .service import RAGService
from .schemas import (
    Document,
    Chunk,
    IngestRequest,
    IngestResponse,
    RetrieveRequest,
    RetrieveResponse,
    SearchResult,
    RAGStats,
    DeleteResponse,
)

__all__ = [
    "RAGService",
    "Document",
    "Chunk",
    "IngestRequest",
    "IngestResponse",
    "RetrieveRequest",
    "RetrieveResponse",
    "SearchResult",
    "RAGStats",
    "DeleteResponse",
]
