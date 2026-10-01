# RAG Knowledge Base — Phase 9

## 1. Purpose

Phase 9 adds a **local Retrieval-Augmented Generation (RAG) knowledge layer**
to the existing agentic pipeline. Given a natural-language query about bearing
health, fault types, troubleshooting, or maintenance, it retrieves the most
relevant technical passages from a curated Markdown knowledge base.

> **This is a retrieval layer, not a complete RAG answer-generation system.**
> **No LLM is used in Phase 9.**

The retrieved passages are intended to be surfaced to:
- the operator via a future dashboard (Phase 10)
- an LLM-based explanation generator (Phase 10+)
- the existing deterministic health report

> **Knowledge-base content is engineering reference material. It does not
> replace equipment-specific engineering inspection or certified maintenance
> procedures.**
>
> **Retrieval relevance is not equivalent to fault confirmation.**

---

## 2. Architecture

```
knowledge_base/              ← Curated Markdown source (Git-tracked)
    bearing_faults.md
    maintenance_guidelines.md
    troubleshooting.md
    equipment_conditions.md
         │
         ▼
documents.py                 ← Load, chunk, assign deterministic IDs
         │
         ▼
embeddings.py                ← EmbeddingConfig (Ollama host & model from env)
         │
         ▼ Ollama (local API, no external API key)
         │
         ▼
vector_store.py              ← ChromaDB PersistentClient + upsert
         │
         ▼
chroma_db/                   ← Persisted locally (Git-ignored)
    industrial_health_knowledge
         │
         ▼
retriever.py                 ← KnowledgeRetriever.retrieve(query, top_k)
         │
         ▼
RetrievalResult              ← content, source, title, section, distance
```

---

## 3. Knowledge Sources

| File | Content |
|------|---------|
| `bearing_faults.md` | Fault types (OR, IR, RE, cage), vibration signatures, severity stages, IMS dataset notes |
| `maintenance_guidelines.md` | Vibration monitoring, lubrication, alignment, inspection, escalation principles |
| `troubleshooting.md` | Symptom-based diagnosis (high RMS, high kurtosis, unusual noise, contamination, EDM) |
| `equipment_conditions.md` | Normal/degrading/critical conditions, sensor quality, health-score interpretation limitations |

> These documents contain engineering reference information. They do not
> present IMS dataset experimental results as validated clinical benchmarks.

---

## 4. Chunking Strategy

**Method:** Heading-aware section splitting.

1. The document is split at every `## ` (level-2) or deeper heading.
2. Each section becomes at least one chunk.
3. Sections exceeding `MAX_SECTION_CHARS = 800` are split further on
   blank lines (paragraph boundaries), then on sentence boundaries.
4. Chunks shorter than `MIN_CHUNK_CHARS = 20` are discarded.

**Rationale:** The knowledge-base documents are organised by section headings.
Keeping sections intact produces semantically coherent chunks. Character-limit
splitting ensures no single chunk exceeds ChromaDB/embedding model token limits.

**Determinism:** The same document always produces the same sections in the
same order. Chunk indices are assigned 0, 1, 2, ... in document order.

---

## 5. Embedding Model

| Property | Value |
|----------|-------|
| Provider | **Ollama** |
| Default model | `nomic-embed-text` |
| API key | **None required** |
| Configuration | `RAG_EMBEDDING_PROVIDER`, `RAG_EMBEDDING_MODEL`, `RAG_OLLAMA_HOST` |

To use a different model, set in `.env`:
```
RAG_EMBEDDING_MODEL=custom-model
RAG_OLLAMA_HOST=http://localhost:11434
```

The model name and host are defined with defaults in `embeddings.py`.
No other file hard-codes the model name. Ensure the Ollama server is running locally and the model is downloaded before building the knowledge base.

---

## 6. ChromaDB Structure

| Property | Value |
|----------|-------|
| Client | `PersistentClient` |
| Persistence path | `chroma_db/` (project root, Git-ignored) |
| Collection name | `industrial_health_knowledge` |
| Distance metric | Cosine (`hnsw:space = cosine`) |
| Metadata stored | `source`, `document_type`, `title`, `section`, `chunk_index`, `char_count` |

The persistence directory is created automatically on first use.

---

## 7. Metadata Schema

Every chunk in ChromaDB carries:

```json
{
    "source":        "bearing_faults.md",
    "document_type": "bearing_faults",
    "title":         "Bearing Fault Types and Characteristics",
    "section":       "Outer Race Fault (OR Fault)",
    "chunk_index":   3,
    "char_count":    412
}
```

---

## 8. Chunk ID Scheme

```
chunk_id = SHA-256( f"{filename}_{chunk_index}" )[:32]
```

- **Deterministic:** same filename + same chunk_index → same ID across runs.
- **Upsert-safe:** rebuilding the collection with the same documents produces
  identical IDs, so ChromaDB upsert replaces rather than duplicates chunks.
- **Collision-resistant:** 128-bit entropy from truncated SHA-256.

---

## 9. Retrieval Interface

```python
from industrial_health.rag import KnowledgeRetriever

retriever = KnowledgeRetriever(
    knowledge_base_path="knowledge_base",  # default
    chroma_path="chroma_db",              # default
)
retriever.load_or_build()                 # idempotent; use rebuild=True to clear

results = retriever.retrieve(
    query="outer race bearing fault symptoms",
    top_k=5,
)
for r in results:
    print(r.source, r.section, r.distance)
    print(r.preview(150))
```

**RetrievalResult fields:**

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str` | Raw chunk text |
| `source` | `str` | Source `.md` filename |
| `title` | `str` | Document H1 title |
| `section` | `str` | H2 section the chunk belongs to |
| `chunk_index` | `int` | 0-based position within document |
| `distance` | `float` | Cosine distance (0 = closest) |
| `metadata` | `dict` | Full ChromaDB metadata |

---

## 10. Build Instructions

**Prerequisites:** Ollama must be installed and running locally, and the `nomic-embed-text` model must be pulled.

```bash
# First run:
python scripts/build_knowledge_base.py

# Idempotent rebuild (no duplicates):
python scripts/build_knowledge_base.py

# Force clear + rebuild:
python scripts/build_knowledge_base.py --rebuild

# Use a different model:
python scripts/build_knowledge_base.py --model all-mpnet-base-v2

# Run Phase 9 tests (no model download needed):
pytest tests/test_rag.py -v

# Run full test suite:
pytest -q
```

---

## 11. Query Examples

```python
retriever.retrieve("outer race bearing fault symptoms", top_k=3)
retriever.retrieve("high RMS vibration troubleshooting", top_k=5)
retriever.retrieve("bearing maintenance inspection", top_k=4)
retriever.retrieve("degrading equipment condition", top_k=3)
retriever.retrieve("kurtosis interpretation", top_k=3)
```

---

## 12. Limitations

1. **Retrieval only:** Phase 9 retrieves passages. It does not generate
   natural-language answers. Answer synthesis requires an LLM (Phase 10+).

2. **No LLM in Phase 9:** The pipeline is fully local. No OpenAI, Gemini,
   or other LLM API is called.

3. **Static knowledge base:** The Markdown documents are manually curated.
   They do not update automatically when new experiments are run.

4. **Not fault confirmation:** Retrieved passages are reference material.
   A retrieved paragraph about "outer race fault" is not confirmation that
   the monitored equipment has an outer race fault.

5. **Embedding quality:** `nomic-embed-text` is a general-purpose model
   not fine-tuned on vibration or bearing data. Retrieval quality may be
   lower than a domain-adapted model.

6. **Test 1 only (current scope):** Phase 9 provides knowledge retrieval
   independent of test dataset. The retriever works for any test_id query.

---

## 13. Future Integration with Agents

In Phase 10+, the Phase 8 OrchestratorAgent can pass the health/fault/anomaly
state to a `KnowledgeAgent` that:

1. Formulates a retrieval query from the structured state (e.g.
   `f"bearing fault symptom {state.final_health_status} RMS"`)
2. Calls `KnowledgeRetriever.retrieve()`
3. Passes retrieved passages as context to an LLM for explanation generation
4. Writes the LLM explanation back to AgentState

This keeps the retrieval layer (Phase 9) independent of the LLM layer (Phase 10+),
following the existing phase-boundary conventions.
