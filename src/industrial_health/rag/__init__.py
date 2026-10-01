# industrial_health.rag — Phase 9: RAG Knowledge Base
#
# Public API:
#   from industrial_health.rag import KnowledgeRetriever
#   from industrial_health.rag.documents import load_documents, make_chunk_id
#   from industrial_health.rag.embeddings import EmbeddingConfig, get_embedding_function
#   from industrial_health.rag.vector_store import KnowledgeVectorStore
#   from industrial_health.rag.retriever import KnowledgeRetriever, RetrievalResult

from industrial_health.rag.documents import load_documents, DocumentChunk, make_chunk_id
from industrial_health.rag.embeddings import EmbeddingConfig, get_embedding_function
from industrial_health.rag.vector_store import KnowledgeVectorStore
from industrial_health.rag.retriever import KnowledgeRetriever, RetrievalResult

__all__ = [
    "load_documents",
    "DocumentChunk",
    "make_chunk_id",
    "EmbeddingConfig",
    "get_embedding_function",
    "KnowledgeVectorStore",
    "KnowledgeRetriever",
    "RetrievalResult",
]
