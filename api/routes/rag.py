"""
api/routes/rag.py — Phase 11: RAG Knowledge Retrieval Endpoint

POST /api/v1/rag/query
    Retrieve relevant passages from the bearing knowledge base.
    Uses KnowledgeRetriever (Phase 9) with nomic-embed-text via Ollama.

    Returns HTTP 503 specifically when Ollama is unavailable.
    Returns HTTP 400 if the query is empty (caught by Pydantic min_length=1).
    Returns HTTP 500 for unexpected errors.

IMPORTANT:
  This endpoint performs RETRIEVAL ONLY.
  No LLM generation occurs.
  No text is fabricated.
  All results come from the existing knowledge_base/ Markdown documents.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from api.dependencies import get_pipeline_service
from api.schemas import RAGQueryRequest, RAGQueryResponse, RAGResult

router = APIRouter(prefix="/api/v1/rag", tags=["RAG Knowledge Base"])

_DISCLAIMER = (
    "Retrieval only. No LLM generation. "
    "Source: knowledge_base/ Markdown documents. "
    "Results are retrieved passages, not generated answers."
)
_EMBEDDING_MODEL = "nomic-embed-text (Ollama)"


@router.post(
    "/query",
    response_model=RAGQueryResponse,
    summary="Query the bearing knowledge base",
    description=(
        "Retrieve relevant passages from the Phase 9 ChromaDB knowledge base "
        "using vector similarity search. "
        "Embedding is performed by the nomic-embed-text Ollama model. "
        "Returns HTTP 503 if Ollama is not running."
    ),
    responses={
        400: {"description": "Empty or invalid query"},
        422: {"description": "Pydantic validation error (query too long, top_k out of range)"},
        500: {"description": "Retrieval error (ChromaDB or other internal error)"},
        503: {"description": "Ollama embedding service unavailable"},
    },
)
async def rag_query(
    request: RAGQueryRequest,
    service=Depends(get_pipeline_service),
) -> RAGQueryResponse:
    result = service.retrieve_knowledge(request.query, top_k=request.top_k)

    if "error" in result:
        if result.get("is_ollama_error"):
            raise HTTPException(
                status_code=503,
                detail={
                    "error":  result["error"],
                    "detail": "Start Ollama and ensure nomic-embed-text is available.",
                    "hint":   result.get("details", []),
                },
            )
        raise HTTPException(
            status_code=500,
            detail=result["error"],
        )

    rag_results = [
        RAGResult(
            content       = r["content"],
            source        = r["source"],
            title         = r["title"],
            section       = r["section"],
            chunk_index   = r["chunk_index"],
            distance      = r["distance"],
            relevance_pct = round(max(0.0, (1.0 - r["distance"] / 2.0)) * 100, 1),
        )
        for r in result.get("results", [])
    ]

    return RAGQueryResponse(
        query           = request.query,
        count           = len(rag_results),
        results         = rag_results,
        disclaimer      = _DISCLAIMER,
        embedding_model = _EMBEDDING_MODEL,
    )
