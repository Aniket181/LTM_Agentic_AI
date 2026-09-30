"""
retriever.py — RAG Knowledge Retriever (ChromaDB)

Loads local markdown knowledge base files, chunks them,
embeds them using sentence-transformers, and stores in ChromaDB.
Retrieves relevant chunks for maintenance recommendations.

Knowledge base structure:
  knowledge_base/
    bearing_faults.md
    maintenance_guidelines.md
    troubleshooting.md
    equipment_conditions.md

This is a simple, local RAG system — no cloud required.
"""

import hashlib
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from loguru import logger


CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 100    # overlap between chunks
COLLECTION_NAME = "bearing_knowledge"
CHROMA_DB_PATH = "chroma_db"


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


class KnowledgeRetriever:
    """
    ChromaDB-backed knowledge retriever for bearing maintenance information.

    Uses sentence-transformers (all-MiniLM-L6-v2) for local embedding —
    no API key required.
    """

    def __init__(
        self,
        knowledge_base_path: Path,
        chroma_db_path: Optional[Path] = None,
    ):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.chroma_db_path = Path(chroma_db_path or CHROMA_DB_PATH)
        self._client = None
        self._collection = None

    def _get_embedding_function(self):
        """Use sentence-transformers for local embeddings (no API key needed)."""
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

    def _get_client(self):
        if self._client is None:
            self._client = chromadb.PersistentClient(path=str(self.chroma_db_path))
        return self._client

    def load_or_build(self) -> None:
        """Load existing ChromaDB collection or build it from knowledge base files."""
        client = self._get_client()
        ef = self._get_embedding_function()

        # Check if collection already exists with content
        try:
            collection = client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=ef,
            )
            if collection.count() > 0:
                logger.info(
                    f"Loaded existing knowledge base: {collection.count()} chunks"
                )
                self._collection = collection
                return
        except Exception:
            pass

        # Build collection from markdown files
        logger.info("Building knowledge base from markdown files...")
        self._collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
        )
        self._ingest_knowledge_base()

    def _ingest_knowledge_base(self) -> None:
        """Load all markdown files and ingest into ChromaDB."""
        md_files = list(self.knowledge_base_path.glob("*.md"))
        if not md_files:
            logger.warning(f"No markdown files found in {self.knowledge_base_path}")
            return

        all_chunks = []
        all_ids = []
        all_metadata = []

        for md_file in md_files:
            text = md_file.read_text(encoding="utf-8")
            chunks = _chunk_text(text)
            for i, chunk in enumerate(chunks):
                chunk_id = hashlib.md5(f"{md_file.name}_{i}".encode()).hexdigest()
                all_chunks.append(chunk)
                all_ids.append(chunk_id)
                all_metadata.append({
                    "source": md_file.name,
                    "chunk_index": i,
                })

        if all_chunks:
            self._collection.add(
                documents=all_chunks,
                ids=all_ids,
                metadatas=all_metadata,
            )
            logger.info(
                f"Ingested {len(all_chunks)} chunks from {len(md_files)} files"
            )

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Retrieve top-k most relevant knowledge chunks for a query.

        Args:
            query: Natural language query.
            top_k: Number of chunks to retrieve.

        Returns:
            List of dicts with 'content', 'source', 'distance'.
        """
        if self._collection is None:
            raise RuntimeError("Call load_or_build() before retrieve()")

        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self._collection.count()),
        )

        retrieved = []
        if results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]
            for doc, meta, dist in zip(docs, metas, dists):
                retrieved.append({
                    "content":  doc,
                    "source":   meta.get("source", "unknown"),
                    "distance": float(dist),
                })

        logger.info(f"Retrieved {len(retrieved)} chunks for query: '{query[:50]}...'")
        return retrieved
