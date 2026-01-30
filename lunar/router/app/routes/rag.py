"""RAG API routes for document retrieval and ingestion."""

import json
import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile

from ..rag import (
    RAGService,
    IngestRequest,
    IngestResponse,
    RetrieveResponse,
    SearchResult,
    RAGStats,
    DeleteResponse,
    Document,
)
from ..rag.schemas import ChunkingStrategy, RetrieveRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/rag", tags=["rag"])

# Get the singleton RAG service
rag_service = RAGService()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_content(
    request: Request,
    body: IngestRequest,
):
    """
    Ingest text content into the RAG store.

    The content will be chunked and embedded for later retrieval.
    """
    tenant_id = getattr(request.state, 'tenant_id', 'default')

    try:
        result = await rag_service.ingest(tenant_id, body)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file using PyMuPDF."""
    import fitz  # PyMuPDF

    text_parts = []
    with fitz.open(stream=file_content, filetype="pdf") as doc:
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                text_parts.append(f"[Page {page_num + 1}]\n{text}")

    return "\n\n".join(text_parts)


def extract_text_from_file(filename: str, content: bytes) -> str:
    """Extract text from file based on extension."""
    ext = filename.lower().split(".")[-1] if "." in filename else ""

    if ext == "pdf":
        return extract_text_from_pdf(content)
    elif ext in ("txt", "md", "csv", "json", "xml", "html"):
        # Plain text files - try UTF-8 first, then latin-1
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return content.decode("latin-1")
    else:
        raise ValueError(f"Unsupported file type: {ext}")


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    request: Request,
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
    chunking_strategy: str = Form("sentence"),
    chunk_size: int = Form(512),
    chunk_overlap: int = Form(50),
):
    """
    Ingest a file (PDF, TXT, etc.) into the RAG store.

    Supported file types:
    - PDF (.pdf)
    - Text files (.txt, .md, .csv, .json, .xml, .html)

    The file content will be extracted, chunked, and embedded for later retrieval.
    """
    tenant_id = getattr(request.state, 'tenant_id', 'default')

    # Read file content
    try:
        file_content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    # Extract text from file
    try:
        text_content = extract_text_from_file(file.filename or "unknown.txt", file_content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to extract text from {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {str(e)}")

    if not text_content.strip():
        raise HTTPException(status_code=400, detail="No text content extracted from file")

    # Parse metadata JSON if provided
    parsed_metadata: Dict[str, Any] = {}
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid metadata JSON")

    # Parse chunking strategy
    try:
        strategy = ChunkingStrategy(chunking_strategy)
    except ValueError:
        strategy = ChunkingStrategy.SENTENCE

    # Create ingest request
    ingest_request = IngestRequest(
        content=text_content,
        document_id=document_id,
        filename=file.filename,
        metadata=parsed_metadata,
        chunking_strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Ingest the content
    try:
        result = await rag_service.ingest(tenant_id, ingest_request)
        logger.info(f"Ingested file {file.filename} as {result.document_id} with {result.chunks_created} chunks")
        return result
    except Exception as e:
        logger.error(f"Failed to ingest file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_content(
    request: Request,
    body: RetrieveRequest,
):
    """
    Retrieve relevant content for a query.

    Returns the most similar chunks to the query based on semantic search.
    """
    tenant_id = getattr(request.state, 'tenant_id', 'default')

    try:
        result = await rag_service.retrieve(
            tenant_id=tenant_id,
            query=body.query,
            top_k=body.top_k,
            min_score=body.min_score,
            metadata_filter=body.metadata_filter,
            rerank=body.rerank,
            rerank_top_k=body.rerank_top_k,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")


@router.get("/stats", response_model=RAGStats)
async def get_stats(request: Request):
    """
    Get statistics about the RAG store.

    Returns the total number of documents and chunks.
    """
    tenant_id = getattr(request.state, 'tenant_id', 'default')

    try:
        return await rag_service.stats(tenant_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.get("/documents", response_model=List[Document])
async def list_documents(
    request: Request,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    """
    List all documents in the RAG store.
    """
    tenant_id = getattr(request.state, 'tenant_id', 'default')

    try:
        return await rag_service.list_documents(tenant_id, limit, offset)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.get("/documents/{document_id}", response_model=Document)
async def get_document(
    request: Request,
    document_id: str,
):
    """
    Get a document by ID.
    """
    tenant_id = getattr(request.state, 'tenant_id', 'default')

    try:
        doc = await rag_service.get_document(tenant_id, document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return doc
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(
    request: Request,
    document_id: str,
):
    """
    Delete a document and all its chunks.
    """
    tenant_id = getattr(request.state, 'tenant_id', 'default')

    try:
        result = await rag_service.delete_document(tenant_id, document_id)
        if not result.success:
            raise HTTPException(status_code=404, detail=result.message)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@router.delete("/clear", response_model=DeleteResponse)
async def clear_store(request: Request):
    """
    Clear all documents and chunks from the RAG store.

    Warning: This permanently deletes all data!
    """
    tenant_id = getattr(request.state, 'tenant_id', 'default')

    try:
        return await rag_service.clear(tenant_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear store: {str(e)}")
