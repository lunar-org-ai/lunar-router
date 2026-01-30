"""RAG service with in-memory vector store."""

import re
import time
import uuid
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .schemas import (
    Chunk,
    ChunkingStrategy,
    DeleteResponse,
    Document,
    IngestRequest,
    IngestResponse,
    RAGStats,
    RetrieveResponse,
    SearchResult,
)

logger = logging.getLogger(__name__)


class RAGService:
    """RAG service with in-memory vector store using sentence-transformers."""

    _instance: Optional["RAGService"] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if RAGService._initialized:
            return

        logger.info("Initializing RAG service...")

        # In-memory storage (per tenant)
        self._documents: Dict[str, Dict[str, Document]] = {}  # tenant_id -> doc_id -> Document
        self._chunks: Dict[str, Dict[str, Chunk]] = {}  # tenant_id -> chunk_id -> Chunk
        self._embeddings: Dict[str, Dict[str, np.ndarray]] = {}  # tenant_id -> chunk_id -> embedding

        # Lazy load embedding model
        self._model = None
        self._model_name = "all-MiniLM-L6-v2"  # Fast, good quality

        RAGService._initialized = True
        logger.info("RAG service initialized")

    @property
    def model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self._model_name}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            logger.info("Embedding model loaded")
        return self._model

    def _ensure_tenant(self, tenant_id: str) -> None:
        """Ensure tenant storage exists."""
        if tenant_id not in self._documents:
            self._documents[tenant_id] = {}
            self._chunks[tenant_id] = {}
            self._embeddings[tenant_id] = {}

    def _chunk_text(
        self,
        text: str,
        strategy: ChunkingStrategy,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> List[str]:
        """Split text into chunks based on strategy."""
        if strategy == ChunkingStrategy.SENTENCE:
            return self._chunk_by_sentence(text, chunk_size, chunk_overlap)
        elif strategy == ChunkingStrategy.PARAGRAPH:
            return self._chunk_by_paragraph(text, chunk_size)
        else:  # FIXED_SIZE
            return self._chunk_fixed_size(text, chunk_size, chunk_overlap)

    def _chunk_by_sentence(
        self, text: str, max_chunk_size: int, overlap: int
    ) -> List[str]:
        """Split text by sentences, grouping into chunks."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_size = len(sentence)

            if current_size + sentence_size > max_chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Keep last sentence for overlap
                if overlap > 0 and current_chunk:
                    current_chunk = [current_chunk[-1]]
                    current_size = len(current_chunk[0])
                else:
                    current_chunk = []
                    current_size = 0

            current_chunk.append(sentence)
            current_size += sentence_size + 1  # +1 for space

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks if chunks else [text]

    def _chunk_by_paragraph(self, text: str, max_chunk_size: int) -> List[str]:
        """Split text by paragraphs."""
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(para) <= max_chunk_size:
                chunks.append(para)
            else:
                # Split large paragraphs by sentences
                chunks.extend(self._chunk_by_sentence(para, max_chunk_size, 0))

        return chunks if chunks else [text]

    def _chunk_fixed_size(
        self, text: str, chunk_size: int, overlap: int
    ) -> List[str]:
        """Split text into fixed-size chunks with overlap."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # Try to break at word boundary
            if end < len(text):
                last_space = chunk.rfind(' ')
                if last_space > chunk_size * 0.8:
                    chunk = chunk[:last_space]
                    end = start + last_space

            chunks.append(chunk.strip())
            start = end - overlap

        return chunks if chunks else [text]

    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for text."""
        return self.model.encode(text, convert_to_numpy=True)

    def _compute_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Compute embeddings for multiple texts."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [embeddings[i] for i in range(len(texts))]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    async def ingest(
        self,
        tenant_id: str,
        request: IngestRequest,
    ) -> IngestResponse:
        """Ingest content into the RAG store."""
        self._ensure_tenant(tenant_id)

        # Generate document ID
        doc_id = request.document_id or f"doc_{uuid.uuid4().hex[:12]}"

        # Chunk the content
        chunk_texts = self._chunk_text(
            request.content,
            request.chunking_strategy,
            request.chunk_size,
            request.chunk_overlap,
        )

        # Compute embeddings for all chunks
        embeddings = self._compute_embeddings_batch(chunk_texts)

        # Create chunks
        chunks_created = 0
        for i, (text, embedding) in enumerate(zip(chunk_texts, embeddings)):
            chunk_id = f"chunk_{uuid.uuid4().hex[:12]}"
            chunk = Chunk(
                id=chunk_id,
                document_id=doc_id,
                content=text,
                position=i,
                metadata=request.metadata.copy(),
            )
            self._chunks[tenant_id][chunk_id] = chunk
            self._embeddings[tenant_id][chunk_id] = embedding
            chunks_created += 1

        # Create document record
        doc = Document(
            id=doc_id,
            filename=request.filename,
            metadata=request.metadata,
            created_at=datetime.utcnow().isoformat(),
            chunk_count=chunks_created,
        )
        self._documents[tenant_id][doc_id] = doc

        logger.info(f"Ingested document {doc_id} with {chunks_created} chunks for tenant {tenant_id}")

        return IngestResponse(
            document_id=doc_id,
            chunks_created=chunks_created,
            status="success",
        )

    async def retrieve(
        self,
        tenant_id: str,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        metadata_filter: Optional[Dict[str, Any]] = None,
        rerank: bool = False,
        rerank_top_k: int = 3,
    ) -> RetrieveResponse:
        """Retrieve relevant chunks for a query."""
        start_time = time.time()
        self._ensure_tenant(tenant_id)

        # Compute query embedding
        query_embedding = self._compute_embedding(query)

        # Search all chunks for this tenant
        results: List[Tuple[str, float]] = []

        for chunk_id, chunk in self._chunks[tenant_id].items():
            # Apply metadata filter if provided
            if metadata_filter:
                match = True
                for key, value in metadata_filter.items():
                    if chunk.metadata.get(key) != value:
                        match = False
                        break
                if not match:
                    continue

            # Compute similarity
            chunk_embedding = self._embeddings[tenant_id][chunk_id]
            score = self._cosine_similarity(query_embedding, chunk_embedding)

            if score >= min_score:
                results.append((chunk_id, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        # Limit to top_k (or more for reranking)
        k = top_k * 3 if rerank else top_k
        results = results[:k]

        # Build search results
        search_results = []
        for chunk_id, score in results:
            chunk = self._chunks[tenant_id][chunk_id]
            search_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    score=score,
                    metadata=chunk.metadata,
                )
            )

        # Simple reranking (could use cross-encoder in production)
        if rerank and len(search_results) > rerank_top_k:
            search_results = search_results[:rerank_top_k]

        latency_ms = (time.time() - start_time) * 1000

        return RetrieveResponse(
            query=query,
            results=search_results[:top_k],
            latency_ms=latency_ms,
            total_results=len(search_results),
        )

    async def get_document(self, tenant_id: str, document_id: str) -> Optional[Document]:
        """Get document by ID."""
        self._ensure_tenant(tenant_id)
        return self._documents[tenant_id].get(document_id)

    async def list_documents(
        self,
        tenant_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Document]:
        """List all documents for a tenant."""
        self._ensure_tenant(tenant_id)
        docs = list(self._documents[tenant_id].values())
        return docs[offset:offset + limit]

    async def delete_document(self, tenant_id: str, document_id: str) -> DeleteResponse:
        """Delete a document and all its chunks."""
        self._ensure_tenant(tenant_id)

        if document_id not in self._documents[tenant_id]:
            return DeleteResponse(success=False, message="Document not found")

        # Delete chunks
        deleted_chunks = 0
        chunk_ids_to_delete = [
            chunk_id
            for chunk_id, chunk in self._chunks[tenant_id].items()
            if chunk.document_id == document_id
        ]

        for chunk_id in chunk_ids_to_delete:
            del self._chunks[tenant_id][chunk_id]
            del self._embeddings[tenant_id][chunk_id]
            deleted_chunks += 1

        # Delete document
        del self._documents[tenant_id][document_id]

        return DeleteResponse(success=True, deleted_chunks=deleted_chunks)

    async def stats(self, tenant_id: str) -> RAGStats:
        """Get statistics for a tenant's RAG store."""
        self._ensure_tenant(tenant_id)

        return RAGStats(
            total_documents=len(self._documents[tenant_id]),
            total_chunks=len(self._chunks[tenant_id]),
        )

    async def clear(self, tenant_id: str) -> DeleteResponse:
        """Clear all documents and chunks for a tenant."""
        self._ensure_tenant(tenant_id)

        deleted_chunks = len(self._chunks[tenant_id])

        self._documents[tenant_id] = {}
        self._chunks[tenant_id] = {}
        self._embeddings[tenant_id] = {}

        return DeleteResponse(success=True, deleted_chunks=deleted_chunks)


# Global singleton instance
rag_service = RAGService()
