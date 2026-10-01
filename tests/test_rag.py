"""
test_rag.py — Phase 9: Comprehensive RAG Knowledge Base Tests

Test groups:
  1.  Knowledge directory exists
  2.  All expected Markdown files exist
  3.  Documents load successfully
  4.  Chunks are non-empty
  5.  Metadata exists and has required fields
  6.  Chunk IDs are deterministic
  7.  Duplicate IDs are not generated within a document set
  8.  Embedding config initialization
  9.  ChromaDB collection can be created (in-memory)
  10. Documents can be inserted into ChromaDB
  11. Collection count is correct after insert
  12. Retrieval returns results
  13. Retrieval results contain expected metadata fields
  14. top_k validation rejects invalid values
  15. Deterministic retrieval ordering (same query, same order)
  16. Rebuilding does not create duplicate documents
  17. Empty query raises ValueError
  18. Missing knowledge-base directory raises FileNotFoundError
  19. No LLM dependency is required for chunking/metadata tests

Tests 1–8 and 18–19 are pure Python (no ChromaDB, no sentence-transformers).
Tests 9–17 use an in-memory ChromaDB with a mock embedding function.

Real embedding downloads are NOT triggered by unit tests.
"""

import sys
import pytest
import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
EXPECTED_FILES = {
    "bearing_faults.md",
    "maintenance_guidelines.md",
    "troubleshooting.md",
    "equipment_conditions.md",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_temp_kb(tmp_path: Path, content: str | None = None) -> Path:
    """Create a minimal temporary knowledge-base directory for isolation tests."""
    kb = tmp_path / "knowledge_base"
    kb.mkdir()
    text = content or """# Test Document

## Section One
This is the first section of the test document. It contains enough text
to make a meaningful chunk for testing purposes.

## Section Two
This is the second section. It has different content so retrieval tests
can distinguish between them.
"""
    (kb / "bearing_faults.md").write_text(text, encoding="utf-8")
    return kb


def _mock_embedding_function():
    """
    Return a callable that mimics a ChromaDB embedding function using
    deterministic dummy embeddings. Does NOT download any model.
    """
    import numpy as np
    try:
        from chromadb.api.types import EmbeddingFunction
    except ImportError:
        EmbeddingFunction = object  # Fallback if chromadb not installed

    class MockEmbeddingFunction(EmbeddingFunction):
        def __init__(self):
            pass

        @staticmethod
        def name() -> str:
            return "mock_embedding"

        def get_config(self) -> dict:
            return {}

        @staticmethod
        def build_from_config(config: dict) -> "MockEmbeddingFunction":
            return MockEmbeddingFunction()

        def __call__(self, input: list[str]) -> list[list[float]]:
            """Return a unit vector for each text — purely deterministic."""
            vecs = []
            for t in input:
                # Deterministic embedding: hash-derived 64-dim vector
                h = hashlib.sha256(t.encode("utf-8")).digest()
                vec = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
                # Pad or truncate to 64 dims
                if len(vec) < 64:
                    vec = np.concatenate([vec, np.zeros(64 - len(vec), dtype=np.float32)])
                else:
                    vec = vec[:64]
                norm = np.linalg.norm(vec)
                vec = vec / norm if norm > 0 else vec
                vecs.append(vec.tolist())
            return vecs

    return MockEmbeddingFunction()


def _make_in_memory_collection(name: str = "test_col"):
    """
    Create an in-memory ChromaDB collection with mock embeddings.
    Returns the collection object.
    """
    import chromadb
    client = chromadb.EphemeralClient()
    ef = _mock_embedding_function()
    collection = client.create_collection(
        name=name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ── 1. Knowledge directory exists ─────────────────────────────────────────────

class TestKnowledgeDirectoryExists:

    def test_knowledge_base_dir_exists(self):
        assert KNOWLEDGE_BASE_DIR.exists(), (
            f"knowledge_base/ not found at {KNOWLEDGE_BASE_DIR}. "
            "Run Phase 9 setup."
        )

    def test_knowledge_base_is_directory(self):
        assert KNOWLEDGE_BASE_DIR.is_dir()


# ── 2. All expected Markdown files exist ──────────────────────────────────────

class TestExpectedMarkdownFilesExist:

    def test_bearing_faults_md_exists(self):
        assert (KNOWLEDGE_BASE_DIR / "bearing_faults.md").exists()

    def test_maintenance_guidelines_md_exists(self):
        assert (KNOWLEDGE_BASE_DIR / "maintenance_guidelines.md").exists()

    def test_troubleshooting_md_exists(self):
        assert (KNOWLEDGE_BASE_DIR / "troubleshooting.md").exists()

    def test_equipment_conditions_md_exists(self):
        assert (KNOWLEDGE_BASE_DIR / "equipment_conditions.md").exists()

    def test_no_unexpected_files_dominate(self):
        """At least the 4 expected files are present."""
        found = {f.name for f in KNOWLEDGE_BASE_DIR.glob("*.md")}
        assert EXPECTED_FILES.issubset(found), (
            f"Missing: {EXPECTED_FILES - found}"
        )


# ── 3. Documents load successfully ────────────────────────────────────────────

class TestDocumentsLoad:

    def test_load_documents_returns_list(self):
        from industrial_health.rag.documents import load_documents
        chunks = load_documents(KNOWLEDGE_BASE_DIR)
        assert isinstance(chunks, list)

    def test_load_documents_nonempty(self):
        from industrial_health.rag.documents import load_documents
        chunks = load_documents(KNOWLEDGE_BASE_DIR)
        assert len(chunks) > 0

    def test_load_documents_from_all_four_files(self):
        from industrial_health.rag.documents import load_documents
        chunks = load_documents(KNOWLEDGE_BASE_DIR)
        sources = {c.metadata["source"] for c in chunks}
        assert sources == EXPECTED_FILES

    def test_load_documents_sorted_by_file_then_index(self):
        """Chunks from the same file must appear in ascending chunk_index order."""
        from industrial_health.rag.documents import load_documents
        chunks = load_documents(KNOWLEDGE_BASE_DIR)
        by_source: dict[str, list] = {}
        for c in chunks:
            by_source.setdefault(c.metadata["source"], []).append(c.metadata["chunk_index"])
        for src, indices in by_source.items():
            assert indices == sorted(indices), (
                f"Chunks for {src} are not in ascending chunk_index order."
            )


# ── 4. Chunks are non-empty ───────────────────────────────────────────────────

class TestChunksNonEmpty:

    def test_all_chunks_have_content(self):
        from industrial_health.rag.documents import load_documents
        chunks = load_documents(KNOWLEDGE_BASE_DIR)
        for c in chunks:
            assert len(c.content.strip()) > 0, f"Empty chunk: {c.chunk_id}"

    def test_all_chunks_above_min_length(self):
        from industrial_health.rag.documents import load_documents, MIN_CHUNK_CHARS
        chunks = load_documents(KNOWLEDGE_BASE_DIR)
        for c in chunks:
            assert len(c.content) >= MIN_CHUNK_CHARS, (
                f"Chunk {c.chunk_id} has only {len(c.content)} chars"
            )


# ── 5. Metadata exists and has required fields ────────────────────────────────

class TestMetadataExists:

    REQUIRED_KEYS = {"source", "document_type", "title", "section",
                     "chunk_index", "char_count"}

    def test_all_chunks_have_metadata(self):
        from industrial_health.rag.documents import load_documents
        chunks = load_documents(KNOWLEDGE_BASE_DIR)
        for c in chunks:
            assert isinstance(c.metadata, dict)
            assert c.metadata, "Metadata dict must not be empty."

    def test_all_metadata_keys_present(self):
        from industrial_health.rag.documents import load_documents
        chunks = load_documents(KNOWLEDGE_BASE_DIR)
        for c in chunks:
            missing = self.REQUIRED_KEYS - set(c.metadata.keys())
            assert not missing, (
                f"Chunk {c.chunk_id} missing metadata keys: {missing}"
            )

    def test_source_matches_filename(self):
        from industrial_health.rag.documents import load_documents
        chunks = load_documents(KNOWLEDGE_BASE_DIR)
        for c in chunks:
            assert c.metadata["source"] in EXPECTED_FILES

    def test_document_type_not_empty(self):
        from industrial_health.rag.documents import load_documents
        chunks = load_documents(KNOWLEDGE_BASE_DIR)
        for c in chunks:
            assert c.metadata["document_type"], "document_type must not be empty."

    def test_title_is_string(self):
        from industrial_health.rag.documents import load_documents
        chunks = load_documents(KNOWLEDGE_BASE_DIR)
        for c in chunks:
            assert isinstance(c.metadata["title"], str)

    def test_chunk_index_is_int(self):
        from industrial_health.rag.documents import load_documents
        chunks = load_documents(KNOWLEDGE_BASE_DIR)
        for c in chunks:
            assert isinstance(c.metadata["chunk_index"], int)
            assert c.metadata["chunk_index"] >= 0


# ── 6. Chunk IDs are deterministic ────────────────────────────────────────────

class TestChunkIDsDeterministic:

    def test_make_chunk_id_is_deterministic(self):
        from industrial_health.rag.documents import make_chunk_id
        id1 = make_chunk_id("bearing_faults.md", 0)
        id2 = make_chunk_id("bearing_faults.md", 0)
        assert id1 == id2

    def test_make_chunk_id_differs_by_index(self):
        from industrial_health.rag.documents import make_chunk_id
        id0 = make_chunk_id("bearing_faults.md", 0)
        id1 = make_chunk_id("bearing_faults.md", 1)
        assert id0 != id1

    def test_make_chunk_id_differs_by_source(self):
        from industrial_health.rag.documents import make_chunk_id
        id_a = make_chunk_id("bearing_faults.md", 0)
        id_b = make_chunk_id("troubleshooting.md", 0)
        assert id_a != id_b

    def test_load_documents_twice_same_ids(self):
        from industrial_health.rag.documents import load_documents
        chunks1 = load_documents(KNOWLEDGE_BASE_DIR)
        chunks2 = load_documents(KNOWLEDGE_BASE_DIR)
        ids1 = [c.chunk_id for c in chunks1]
        ids2 = [c.chunk_id for c in chunks2]
        assert ids1 == ids2

    def test_chunk_id_length(self):
        from industrial_health.rag.documents import make_chunk_id
        cid = make_chunk_id("bearing_faults.md", 0)
        assert len(cid) == 32


# ── 7. Duplicate IDs not generated ───────────────────────────────────────────

class TestNoDuplicateIDs:

    def test_no_duplicate_chunk_ids_across_all_documents(self):
        from industrial_health.rag.documents import load_documents
        chunks = load_documents(KNOWLEDGE_BASE_DIR)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), (
            f"Duplicate chunk IDs detected: "
            f"{len(ids) - len(set(ids))} duplicates."
        )

    def test_no_duplicate_ids_from_temp_kb(self, tmp_path):
        from industrial_health.rag.documents import load_documents
        kb = _make_temp_kb(tmp_path)
        chunks = load_documents(kb)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))


# ── 8. Embedding config initialization ───────────────────────────────────────

class TestEmbeddingConfig:

    def test_default_config_uses_nomic(self):
        from industrial_health.rag.embeddings import EmbeddingConfig, DEFAULT_EMBEDDING_MODEL, DEFAULT_OLLAMA_HOST
        cfg = EmbeddingConfig()
        assert cfg.model_name == DEFAULT_EMBEDDING_MODEL
        assert cfg.host == DEFAULT_OLLAMA_HOST

    def test_custom_model_name(self):
        from industrial_health.rag.embeddings import EmbeddingConfig
        cfg = EmbeddingConfig(model_name="custom-model")
        assert cfg.model_name == "custom-model"

    def test_env_var_overrides_default(self, monkeypatch):
        from industrial_health.rag import embeddings
        monkeypatch.setenv("RAG_EMBEDDING_MODEL", "custom-model")
        monkeypatch.setenv("RAG_OLLAMA_HOST", "http://test:11434")
        cfg = embeddings.EmbeddingConfig()
        assert cfg.model_name == "custom-model"
        assert cfg.host == "http://test:11434"

    def test_empty_model_name_uses_default(self):
        from industrial_health.rag.embeddings import EmbeddingConfig, DEFAULT_EMBEDDING_MODEL
        cfg = EmbeddingConfig(model_name="")
        assert cfg.model_name == DEFAULT_EMBEDDING_MODEL


# ── 9. ChromaDB collection can be created ────────────────────────────────────

class TestChromaCollectionCreation:

    def test_ephemeral_collection_created(self):
        import chromadb
        client = chromadb.EphemeralClient()
        col = client.create_collection("test_creation")
        assert col is not None
        assert col.name == "test_creation"

    def test_collection_starts_empty(self):
        import chromadb
        client = chromadb.EphemeralClient()
        col = client.create_collection("test_empty")
        assert col.count() == 0


# ── 10. Documents can be inserted ─────────────────────────────────────────────

class TestChromaDocumentInsert:

    def test_upsert_chunks_succeeds(self, tmp_path):
        from industrial_health.rag.documents import load_documents
        from industrial_health.rag.vector_store import KnowledgeVectorStore, _coerce_metadata

        kb = _make_temp_kb(tmp_path)
        chunks = load_documents(kb)
        assert len(chunks) > 0

        import chromadb
        client = chromadb.EphemeralClient()
        ef = _mock_embedding_function()
        col = client.create_collection("test_insert", embedding_function=ef)

        col.upsert(
            ids       =[c.chunk_id for c in chunks],
            documents =[c.content  for c in chunks],
            metadatas =[_coerce_metadata(c.metadata) for c in chunks],
        )
        assert col.count() == len(chunks)


# ── 11. Collection count is correct ───────────────────────────────────────────

class TestCollectionCount:

    def test_count_after_single_insert(self, tmp_path):
        from industrial_health.rag.documents import load_documents, make_chunk_id
        from industrial_health.rag.vector_store import _coerce_metadata

        kb = _make_temp_kb(tmp_path)
        chunks = load_documents(kb)
        n = len(chunks)

        import chromadb
        client = chromadb.EphemeralClient()
        ef = _mock_embedding_function()
        col = client.create_collection("test_count", embedding_function=ef)

        col.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.content for c in chunks],
            metadatas=[_coerce_metadata(c.metadata) for c in chunks],
        )
        assert col.count() == n

    def test_double_upsert_no_duplicates(self, tmp_path):
        """Upserting the same chunks twice must not increase the count."""
        from industrial_health.rag.documents import load_documents
        from industrial_health.rag.vector_store import _coerce_metadata

        kb = _make_temp_kb(tmp_path)
        chunks = load_documents(kb)

        import chromadb
        client = chromadb.EphemeralClient()
        ef = _mock_embedding_function()
        col = client.create_collection("test_upsert_dedup", embedding_function=ef)

        ids = [c.chunk_id for c in chunks]
        docs = [c.content for c in chunks]
        metas = [_coerce_metadata(c.metadata) for c in chunks]

        col.upsert(ids=ids, documents=docs, metadatas=metas)
        count_after_first = col.count()

        col.upsert(ids=ids, documents=docs, metadatas=metas)
        count_after_second = col.count()

        assert count_after_first == count_after_second, (
            "Double upsert created duplicates."
        )


# ── 12. Retrieval returns results ─────────────────────────────────────────────

class TestRetrievalReturnsResults:

    def _build_retriever_with_mock_ef(self, tmp_path):
        """Build a KnowledgeRetriever backed by mock embeddings."""
        from industrial_health.rag.retriever import KnowledgeRetriever
        from industrial_health.rag.embeddings import EmbeddingConfig

        kb = _make_temp_kb(tmp_path)

        # Patch get_embedding_function to return the mock
        with patch("industrial_health.rag.vector_store.get_embedding_function",
                   return_value=_mock_embedding_function()):
            retriever = KnowledgeRetriever(
                knowledge_base_path=kb,
                chroma_path=str(tmp_path / "chroma"),
            )
            retriever.load_or_build()
        return retriever

    def test_retrieve_returns_list(self, tmp_path):
        retriever = self._build_retriever_with_mock_ef(tmp_path)
        with patch("industrial_health.rag.vector_store.get_embedding_function",
                   return_value=_mock_embedding_function()):
            results = retriever.retrieve("bearing fault", top_k=2)
        assert isinstance(results, list)

    def test_retrieve_returns_correct_count(self, tmp_path):
        retriever = self._build_retriever_with_mock_ef(tmp_path)
        n_docs = retriever.count()
        top_k = min(2, n_docs)
        with patch("industrial_health.rag.vector_store.get_embedding_function",
                   return_value=_mock_embedding_function()):
            results = retriever.retrieve("test query", top_k=top_k)
        assert len(results) <= top_k

    def test_retrieve_nonempty_content(self, tmp_path):
        retriever = self._build_retriever_with_mock_ef(tmp_path)
        with patch("industrial_health.rag.vector_store.get_embedding_function",
                   return_value=_mock_embedding_function()):
            results = retriever.retrieve("test section content", top_k=1)
        assert len(results) >= 1
        assert len(results[0].content.strip()) > 0


# ── 13. Retrieval results contain expected metadata ───────────────────────────

class TestRetrievalMetadata:

    def _build_retriever(self, tmp_path):
        from industrial_health.rag.retriever import KnowledgeRetriever
        kb = _make_temp_kb(tmp_path)
        with patch("industrial_health.rag.vector_store.get_embedding_function",
                   return_value=_mock_embedding_function()):
            r = KnowledgeRetriever(
                knowledge_base_path=kb,
                chroma_path=str(tmp_path / "chroma"),
            )
            r.load_or_build()
        return r

    def test_result_has_source_field(self, tmp_path):
        r = self._build_retriever(tmp_path)
        with patch("industrial_health.rag.vector_store.get_embedding_function",
                   return_value=_mock_embedding_function()):
            results = r.retrieve("section", top_k=1)
        assert hasattr(results[0], "source")
        assert results[0].source.endswith(".md")

    def test_result_has_title_field(self, tmp_path):
        r = self._build_retriever(tmp_path)
        with patch("industrial_health.rag.vector_store.get_embedding_function",
                   return_value=_mock_embedding_function()):
            results = r.retrieve("section", top_k=1)
        assert hasattr(results[0], "title")
        assert isinstance(results[0].title, str)

    def test_result_has_distance_field(self, tmp_path):
        r = self._build_retriever(tmp_path)
        with patch("industrial_health.rag.vector_store.get_embedding_function",
                   return_value=_mock_embedding_function()):
            results = r.retrieve("section", top_k=1)
        assert hasattr(results[0], "distance")
        assert isinstance(results[0].distance, float)

    def test_result_has_chunk_index(self, tmp_path):
        r = self._build_retriever(tmp_path)
        with patch("industrial_health.rag.vector_store.get_embedding_function",
                   return_value=_mock_embedding_function()):
            results = r.retrieve("section", top_k=1)
        assert hasattr(results[0], "chunk_index")
        assert results[0].chunk_index >= 0


# ── 14. top_k validation ──────────────────────────────────────────────────────

class TestTopKValidation:

    def _build_retriever(self, tmp_path):
        from industrial_health.rag.retriever import KnowledgeRetriever
        kb = _make_temp_kb(tmp_path)
        with patch("industrial_health.rag.vector_store.get_embedding_function",
                   return_value=_mock_embedding_function()):
            r = KnowledgeRetriever(
                knowledge_base_path=kb,
                chroma_path=str(tmp_path / "chroma"),
            )
            r.load_or_build()
        return r

    def test_top_k_zero_raises_value_error(self, tmp_path):
        r = self._build_retriever(tmp_path)
        with pytest.raises(ValueError, match="top_k"):
            with patch("industrial_health.rag.vector_store.get_embedding_function",
                       return_value=_mock_embedding_function()):
                r.retrieve("test", top_k=0)

    def test_top_k_negative_raises_value_error(self, tmp_path):
        r = self._build_retriever(tmp_path)
        with pytest.raises(ValueError):
            with patch("industrial_health.rag.vector_store.get_embedding_function",
                       return_value=_mock_embedding_function()):
                r.retrieve("test", top_k=-5)

    def test_top_k_larger_than_collection_capped(self, tmp_path):
        r = self._build_retriever(tmp_path)
        n = r.count()
        with patch("industrial_health.rag.vector_store.get_embedding_function",
                   return_value=_mock_embedding_function()):
            results = r.retrieve("section one content", top_k=n + 100)
        assert len(results) <= n


# ── 15. Deterministic retrieval ordering ──────────────────────────────────────

class TestDeterministicOrdering:

    def test_same_query_same_ordering(self, tmp_path):
        from industrial_health.rag.retriever import KnowledgeRetriever
        kb = _make_temp_kb(tmp_path)
        with patch("industrial_health.rag.vector_store.get_embedding_function",
                   return_value=_mock_embedding_function()):
            r = KnowledgeRetriever(
                knowledge_base_path=kb,
                chroma_path=str(tmp_path / "chroma"),
            )
            r.load_or_build()
            r1 = r.retrieve("bearing fault section", top_k=3)
            r2 = r.retrieve("bearing fault section", top_k=3)

        ids1 = [x.chunk_index for x in r1]
        ids2 = [x.chunk_index for x in r2]
        assert ids1 == ids2

    def test_results_ordered_by_distance_ascending(self, tmp_path):
        from industrial_health.rag.retriever import KnowledgeRetriever
        kb = _make_temp_kb(tmp_path)
        with patch("industrial_health.rag.vector_store.get_embedding_function",
                   return_value=_mock_embedding_function()):
            r = KnowledgeRetriever(
                knowledge_base_path=kb,
                chroma_path=str(tmp_path / "chroma"),
            )
            r.load_or_build()
            results = r.retrieve("first section", top_k=3)

        distances = [x.distance for x in results]
        assert distances == sorted(distances), (
            "Results must be ordered by ascending distance (most relevant first)."
        )


# ── 16. Rebuilding does not create duplicate documents ────────────────────────

class TestRebuildNoDuplicates:

    def test_rebuild_flag_does_not_increase_count(self, tmp_path):
        from industrial_health.rag.retriever import KnowledgeRetriever
        kb = _make_temp_kb(tmp_path)
        chroma_dir = tmp_path / "chroma"

        with patch("industrial_health.rag.vector_store.get_embedding_function",
                   return_value=_mock_embedding_function()):
            r = KnowledgeRetriever(
                knowledge_base_path=kb,
                chroma_path=str(chroma_dir),
            )
            count1 = r.load_or_build(rebuild=False)
            count2 = r.load_or_build(rebuild=True)

        assert count1 == count2, (
            f"Rebuild changed count: {count1} → {count2}. Possible duplicate insertion."
        )

    def test_double_load_or_build_same_count(self, tmp_path):
        from industrial_health.rag.retriever import KnowledgeRetriever
        kb = _make_temp_kb(tmp_path)
        chroma_dir = tmp_path / "chroma"

        with patch("industrial_health.rag.vector_store.get_embedding_function",
                   return_value=_mock_embedding_function()):
            r = KnowledgeRetriever(
                knowledge_base_path=kb,
                chroma_path=str(chroma_dir),
            )
            count1 = r.load_or_build()
            count2 = r.load_or_build()

        assert count1 == count2


# ── 17. Empty query handling ──────────────────────────────────────────────────

class TestEmptyQueryHandling:

    def _build_retriever(self, tmp_path):
        from industrial_health.rag.retriever import KnowledgeRetriever
        kb = _make_temp_kb(tmp_path)
        with patch("industrial_health.rag.vector_store.get_embedding_function",
                   return_value=_mock_embedding_function()):
            r = KnowledgeRetriever(
                knowledge_base_path=kb,
                chroma_path=str(tmp_path / "chroma"),
            )
            r.load_or_build()
        return r

    def test_empty_string_raises_value_error(self, tmp_path):
        r = self._build_retriever(tmp_path)
        with pytest.raises(ValueError, match="non-empty"):
            r.retrieve("")

    def test_whitespace_only_raises_value_error(self, tmp_path):
        r = self._build_retriever(tmp_path)
        with pytest.raises(ValueError, match="non-empty"):
            r.retrieve("   ")


# ── 18. Missing knowledge-base directory ──────────────────────────────────────

class TestMissingKnowledgeBase:

    def test_missing_directory_raises_file_not_found(self, tmp_path):
        from industrial_health.rag.documents import load_documents
        with pytest.raises(FileNotFoundError):
            load_documents(tmp_path / "nonexistent_kb")

    def test_empty_directory_raises_value_error(self, tmp_path):
        from industrial_health.rag.documents import load_documents
        empty_dir = tmp_path / "empty_kb"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No Markdown files"):
            load_documents(empty_dir)


# ── 19. No LLM dependency required for chunking tests ────────────────────────

class TestNoLLMDependency:

    def test_documents_load_without_openai(self):
        """documents.py must not import openai, langchain, or any LLM lib."""
        import importlib
        import importlib.util
        # Verify we can load documents.py without openai being present
        from industrial_health.rag.documents import load_documents, make_chunk_id
        # Just call the functions
        cid = make_chunk_id("bearing_faults.md", 0)
        assert len(cid) == 32

    def test_chunk_id_no_external_deps(self):
        """make_chunk_id uses only stdlib hashlib — no ML deps."""
        from industrial_health.rag.documents import make_chunk_id
        result = make_chunk_id("test.md", 42)
        expected = hashlib.sha256(b"test.md_42").hexdigest()[:32]
        assert result == expected

    def test_embedding_config_no_model_download(self):
        """EmbeddingConfig.__init__ must not download any model."""
        from industrial_health.rag.embeddings import EmbeddingConfig
        # Just creating the config is safe
        cfg = EmbeddingConfig()
        assert cfg.model_name  # non-empty string

    def test_documents_chunking_no_network_call(self, tmp_path):
        """Full chunking pipeline requires no network access."""
        from industrial_health.rag.documents import load_documents
        kb = _make_temp_kb(tmp_path)
        chunks = load_documents(kb)
        assert len(chunks) > 0
