"""
vector_store.py — Phase 9: ChromaDB Vector Store

RESPONSIBILITIES:
  1. Create or open a persistent ChromaDB collection.
  2. Upsert document chunks (idempotent — rebuilding never creates duplicates).
  3. Count documents in the collection.
  4. Explicitly clear/rebuild the collection when requested.
  5. Expose a clean interface for the retriever.

STORAGE:
  Persists to: chroma_db/ (project root, git-ignored).
  Collection: industrial_health_knowledge

DESIGN:
  - All operations use upsert, not add. This ensures idempotency:
    running build_knowledge_base.py twice never duplicates chunks.
  - The collection name is a module constant — one collection for the Phase 9
    knowledge base.
  - Chunk IDs are provided by documents.py (deterministic SHA-256 hashes).
  - Model objects are never stored in the vector store.
  - ChromaDB metadata values must be str | int | float | bool. The store
    enforces this by coercing metadata before upserting.

NO LLM:
  No LLM is used in this module. Embedding is handled by sentence-transformers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from industrial_health.rag.documents import DocumentChunk
from industrial_health.rag.embeddings import EmbeddingConfig, get_embedding_function


# ── Constants ──────────────────────────────────────────────────────────────────

COLLECTION_NAME: str = "industrial_health_knowledge"

#: Default persistence directory (relative to project root).
#: git-ignored via chroma_db/ in .gitignore.
DEFAULT_CHROMA_PATH: str = "chroma_db"

#: Maximum batch size for ChromaDB upsert calls (avoids memory pressure).
UPSERT_BATCH_SIZE: int = 100


# ── KnowledgeVectorStore ──────────────────────────────────────────────────────

class KnowledgeVectorStore:
    """
    ChromaDB-backed vector store for the Phase 9 industrial knowledge base.

    Args:
        chroma_path:      Local directory for ChromaDB persistence.
        embedding_config: EmbeddingConfig for sentence-transformers model.
        collection_name:  ChromaDB collection name.
    """

    def __init__(
        self,
        chroma_path: Path | str = DEFAULT_CHROMA_PATH,
        embedding_config: EmbeddingConfig | None = None,
        collection_name: str = COLLECTION_NAME,
    ):
        self.chroma_path     = Path(chroma_path)
        self.collection_name = collection_name
        self._embedding_cfg  = embedding_config or EmbeddingConfig()
        self._client: chromadb.PersistentClient | None = None
        self._collection = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def _get_client(self) -> chromadb.PersistentClient:
        if self._client is None:
            self.chroma_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.chroma_path))
        return self._client

    def get_or_create_collection(self):
        """
        Open the existing collection or create it if it does not exist.

        Returns the ChromaDB collection object.
        """
        if self._collection is None:
            client = self._get_client()
            ef = get_embedding_function(self._embedding_cfg)
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def clear_and_recreate(self) -> None:
        """
        Explicitly delete and recreate the collection.

        This is the ONLY safe way to wipe the collection. It is only called
        when --rebuild is explicitly passed to the build script.

        Does NOT silently delete the collection during normal operation.
        """
        client = self._get_client()
        try:
            client.delete_collection(name=self.collection_name)
        except Exception:
            pass  # Collection may not exist yet
        self._collection = None
        self.get_or_create_collection()

    # ── Data operations ────────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> int:
        """
        Upsert document chunks into the collection.

        Idempotent: upserting the same chunk twice with the same ID replaces
        the existing entry instead of duplicating it.

        Args:
            chunks: List of DocumentChunk objects from documents.py.

        Returns:
            Number of chunks upserted.

        Raises:
            ValueError: if chunks list is empty.
        """
        if not chunks:
            raise ValueError("Cannot upsert an empty list of chunks.")

        collection = self.get_or_create_collection()

        # Process in batches to avoid memory pressure
        total = 0
        for i in range(0, len(chunks), UPSERT_BATCH_SIZE):
            batch = chunks[i : i + UPSERT_BATCH_SIZE]
            ids        = [c.chunk_id for c in batch]
            documents  = [c.content  for c in batch]
            metadatas  = [_coerce_metadata(c.metadata) for c in batch]

            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            total += len(batch)

        return total

    def count(self) -> int:
        """Return the number of chunks currently in the collection."""
        collection = self.get_or_create_collection()
        return collection.count()

    def get_collection(self):
        """Return the raw ChromaDB collection object (for use by retriever)."""
        return self.get_or_create_collection()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _coerce_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """
    Ensure all metadata values are ChromaDB-compatible types.

    ChromaDB requires metadata values to be: str | int | float | bool.
    Any other type is converted to str.
    """
    coerced = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            coerced[k] = v
        else:
            coerced[k] = str(v)
    return coerced
