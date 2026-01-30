"""RAG schemas for API requests and responses."""

from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field


class ChunkingStrategy(str, Enum):
    """Chunking strategies for document processing."""
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"


class Document(BaseModel):
    """A document in the RAG system."""
    id: str
    filename: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str
    chunk_count: int = 0


class Chunk(BaseModel):
    """A chunk of text from a document."""
    id: str
    document_id: str
    content: str
    position: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    """Request to ingest content into RAG."""
    content: str = Field(..., description="Text content to ingest")
    document_id: Optional[str] = Field(default=None, description="Optional document ID")
    filename: Optional[str] = Field(default=None, description="Optional filename")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    chunking_strategy: ChunkingStrategy = Field(default=ChunkingStrategy.SENTENCE)
    chunk_size: int = Field(default=512, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=50, description="Overlap between chunks")


class IngestResponse(BaseModel):
    """Response from a RAG ingestion operation."""
    document_id: str
    chunks_created: int
    status: str = "success"
    message: Optional[str] = None


class RetrieveRequest(BaseModel):
    """Request to retrieve documents from RAG."""
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=5, description="Number of results to return")
    min_score: float = Field(default=0.0, description="Minimum similarity score")
    metadata_filter: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")
    rerank: bool = Field(default=False, description="Whether to rerank results")
    rerank_top_k: int = Field(default=3, description="Number of results after reranking")


class SearchResult(BaseModel):
    """A single search result from RAG retrieval."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrieveResponse(BaseModel):
    """Response from a RAG retrieval query."""
    query: str
    results: List[SearchResult] = Field(default_factory=list)
    latency_ms: float = 0.0
    total_results: int = 0


class RAGStats(BaseModel):
    """Statistics about the RAG store."""
    total_documents: int = 0
    total_chunks: int = 0
    storage_bytes: Optional[int] = None


class DeleteResponse(BaseModel):
    """Response from a delete operation."""
    success: bool
    deleted_chunks: int = 0
    message: Optional[str] = None
