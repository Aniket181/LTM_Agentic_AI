"""
retriever.py — Phase 9: RAG Knowledge Retriever

RESPONSIBILITIES:
  1. Accept a natural-language query.
  2. Embed the query using the local sentence-transformers model.
  3. Query the ChromaDB collection for the top-k most similar chunks.
  4. Return structured results with content, metadata, and distance.

DESIGN:
  - KnowledgeRetriever is the public interface. It wraps KnowledgeVectorStore
    and exposes a simple retrieve() method.
  - No LLM is used. Retrieval is purely vector-similarity search.
  - Results are ordered by distance (ascending — closest first), which is
    deterministic for identical queries.
  - retrieve() validates top_k > 0 and reports a clear error if the
    collection has not been built.

NO LLM:
  This module performs retrieval only. Answer generation (using a language
  model) is a Phase 10+ concern.

SCORE SEMANTICS:
  ChromaDB with cosine space returns L2-normalized cosine distance:
    distance = 0.0 → identical vectors (most similar)
    distance = 2.0 → maximally dissimilar

  Results are returned in ascending distance order (most relevant first).

BACKWARD COMPATIBILITY:
  This module replaces the monolithic stub from Phase 1. The class name
  KnowledgeRetriever is preserved for any existing imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from industrial_health.rag.documents import load_documents
from industrial_health.rag.embeddings import EmbeddingConfig
from industrial_health.rag.vector_store import KnowledgeVectorStore, COLLECTION_NAME


# ── RetrievalResult ────────────────────────────────────────────────────────────

class RetrievalResult:
    """
    A single retrieved chunk with all associated metadata.

    Attributes:
        content:     Raw text of the chunk.
        source:      Source Markdown filename.
        title:       Document H1 title.
        section:     H2 section heading the chunk belongs to.
        chunk_index: 0-based index of this chunk within its document.
        distance:    Cosine distance (0=most similar, 2=most dissimilar).
        metadata:    Full metadata dict from ChromaDB.
    """

    def __init__(
        self,
        content:     str,
        source:      str,
        title:       str,
        section:     str,
        chunk_index: int,
        distance:    float,
        metadata:    dict[str, Any],
    ):
        self.content     = content
        self.source      = source
        self.title       = title
        self.section     = section
        self.chunk_index = chunk_index
        self.distance    = distance
        self.metadata    = metadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "content":     self.content,
            "source":      self.source,
            "title":       self.title,
            "section":     self.section,
            "chunk_index": self.chunk_index,
            "distance":    self.distance,
        }

    def preview(self, max_chars: int = 120) -> str:
        """Return a truncated content preview for display."""
        text = self.content.replace("\n", " ").strip()
        return text[:max_chars] + ("..." if len(text) > max_chars else "")


# ── KnowledgeRetriever ─────────────────────────────────────────────────────────

class KnowledgeRetriever:
    """
    Retrieves relevant knowledge chunks from the ChromaDB vector store.

    Args:
        knowledge_base_path: Path to the Markdown knowledge-base directory.
        chroma_path:         Path to the ChromaDB persistence directory.
        embedding_config:    EmbeddingConfig for model selection.

    Usage:
        retriever = KnowledgeRetriever(knowledge_base_path=Path("knowledge_base"))
        retriever.load_or_build()
        results = retriever.retrieve("outer race bearing fault symptoms", top_k=5)
    """

    def __init__(
        self,
        knowledge_base_path: Path | str = "knowledge_base",
        chroma_path: Path | str = "chroma_db",
        embedding_config: EmbeddingConfig | None = None,
    ):
        self.knowledge_base_path = Path(knowledge_base_path)
        self._store = KnowledgeVectorStore(
            chroma_path=chroma_path,
            embedding_config=embedding_config,
            collection_name=COLLECTION_NAME,
        )
        self._is_ready = False

    # ── Build / Load ───────────────────────────────────────────────────────────

    def load_or_build(self, rebuild: bool = False) -> int:
        """
        Load an existing collection, or build it from the knowledge-base files.

        Args:
            rebuild: If True, explicitly clear and rebuild the collection.
                     Without rebuild, existing chunks are upserted idempotently.

        Returns:
            Total number of chunks in the collection after the operation.
        """
        if rebuild:
            self._store.clear_and_recreate()

        # Always upsert (idempotent — no duplicates even if called twice)
        chunks = load_documents(self.knowledge_base_path)
        self._store.upsert_chunks(chunks)
        self._is_ready = True
        return self._store.count()

    def count(self) -> int:
        """Return the number of chunks currently in the collection."""
        return self._store.count()

    # ── Retrieve ───────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """
        Retrieve the top-k most relevant knowledge chunks for a query.

        SCORE SEMANTICS:
          Distance values are cosine distances (0 = identical, 2 = opposite).
          Results are ordered ascending by distance (most relevant first).

        Args:
            query: Natural-language query string.
            top_k: Number of results to return. Must be >= 1.

        Returns:
            List of RetrievalResult objects, ordered by relevance (closest first).

        Raises:
            ValueError: if top_k < 1 or query is empty.
            RuntimeError: if the collection has not been built or is empty.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}.")

        collection = self._store.get_collection()
        n_docs = collection.count()

        if n_docs == 0:
            raise RuntimeError(
                "Knowledge base collection is empty. "
                "Run retriever.load_or_build() or "
                "python scripts/build_knowledge_base.py first."
            )

        effective_k = min(top_k, n_docs)
        raw = collection.query(
            query_texts=[query],
            n_results=effective_k,
            include=["documents", "metadatas", "distances"],
        )

        results: list[RetrievalResult] = []
        if raw and raw.get("documents"):
            docs   = raw["documents"][0]
            metas  = raw["metadatas"][0]
            dists  = raw["distances"][0]
            for doc, meta, dist in zip(docs, metas, dists):
                results.append(RetrievalResult(
                    content=doc,
                    source=meta.get("source", "unknown"),
                    title=meta.get("title", ""),
                    section=meta.get("section", ""),
                    chunk_index=int(meta.get("chunk_index", -1)),
                    distance=float(dist),
                    metadata=dict(meta),
                ))

        # Results are already ordered by distance (ascending) by ChromaDB
        return results
