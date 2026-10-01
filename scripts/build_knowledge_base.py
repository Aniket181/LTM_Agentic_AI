"""
build_knowledge_base.py — Phase 9: Build ChromaDB Knowledge Base

Discovers Markdown documents in knowledge_base/, chunks them, embeds them
using a local sentence-transformers model, and upserts into ChromaDB.

Usage:
    python scripts/build_knowledge_base.py            # idempotent upsert
    python scripts/build_knowledge_base.py --rebuild  # clear + rebuild

DO NOT PROCEED TO PHASE 10 WITHOUT APPROVAL.
"""

import sys
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from industrial_health.rag.documents import load_documents, KNOWN_DOC_TYPES
from industrial_health.rag.embeddings import EmbeddingConfig
from industrial_health.rag.retriever import KnowledgeRetriever


KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
CHROMA_PATH        = PROJECT_ROOT / "chroma_db"


def sep(title: str = "", w: int = 60) -> None:
    if title:
        print(f"\n{'─' * w}\n  {title}\n{'─' * w}")
    else:
        print("─" * w)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 9 — Build / Rebuild RAG Knowledge Base"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Explicitly clear and recreate the ChromaDB collection.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Override embedding model name (e.g. all-MiniLM-L6-v2).",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Phase 9 — Knowledge Base Build")
    print("=" * 60)

    # ── Validate knowledge_base/ ───────────────────────────────────────────
    sep("VALIDATION")
    if not KNOWLEDGE_BASE_DIR.exists():
        print(f"  ✗ knowledge_base/ not found at {KNOWLEDGE_BASE_DIR}")
        sys.exit(1)

    md_files = sorted(KNOWLEDGE_BASE_DIR.glob("*.md"))
    expected = set(KNOWN_DOC_TYPES.keys())
    found    = {f.name for f in md_files}
    missing  = expected - found

    for f in md_files:
        print(f"  ✓ {f.name}  ({f.stat().st_size:,} bytes)")
    if missing:
        print(f"\n  ✗ Missing expected files: {missing}")
        sys.exit(1)
    print(f"\n  {len(md_files)} documents found.")

    # ── Load and chunk ─────────────────────────────────────────────────────
    sep("CHUNKING")
    t0 = time.monotonic()
    chunks = load_documents(KNOWLEDGE_BASE_DIR)
    chunk_time = time.monotonic() - t0

    # Per-document breakdown
    from collections import Counter
    source_counts = Counter(c.metadata["source"] for c in chunks)
    for src, cnt in sorted(source_counts.items()):
        print(f"  {src:<40} {cnt:>3} chunks")

    print(f"\n  Total chunks:   {len(chunks)}")
    print(f"  Chunk time:     {chunk_time:.3f}s")

    # ── Configure embedding ────────────────────────────────────────────────
    sep("EMBEDDING MODEL")
    cfg = EmbeddingConfig(model_name=args.model) if args.model else EmbeddingConfig()
    print(f"  Provider:      Ollama")
    print(f"  Model:         {cfg.model_name}")
    print(f"  Ollama host:   {cfg.host}")
    print(f"  Source:        {'--model arg' if args.model else 'RAG_EMBEDDING_MODEL env / default'}")
    print(f"  ChromaDB path: {CHROMA_PATH}")
    print(f"  Collection:    industrial_health_knowledge")

    # ── Build/Upsert ───────────────────────────────────────────────────────
    sep("UPSERTING INTO CHROMADB")
    if args.rebuild:
        print("  --rebuild flag set: clearing existing collection first.")
    else:
        print("  Idempotent upsert (no --rebuild): existing chunks will be updated in-place.")

    retriever = KnowledgeRetriever(
        knowledge_base_path=KNOWLEDGE_BASE_DIR,
        chroma_path=CHROMA_PATH,
        embedding_config=cfg,
    )

    print("  Loading embedding model via Ollama...")
    t1 = time.monotonic()
    try:
        count = retriever.load_or_build(rebuild=args.rebuild)
    except Exception as e:
        print(f"\n  ✗ Build FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    build_time = time.monotonic() - t1

    # ── Summary ────────────────────────────────────────────────────────────
    sep("SUMMARY")
    print(f"""
  Knowledge Base Build
  ────────────────────
  Documents:        {len(md_files)}
  Chunks ingested:  {len(chunks)}
  Collection count: {count}
  Collection:       industrial_health_knowledge
  Provider:         Ollama
  Embedding model:  {cfg.model_name}
  Ollama host:      {cfg.host}
  Persistent path:  {CHROMA_PATH}
  Build time:       {build_time:.1f}s
  Status:           SUCCESS
""")

    # ── Quick smoke test ───────────────────────────────────────────────────
    sep("SMOKE TEST")
    test_queries = [
        "outer race bearing fault symptoms",
        "high RMS vibration troubleshooting",
    ]
    for q in test_queries:
        results = retriever.retrieve(q, top_k=2)
        print(f"  Query: '{q}'")
        for r in results:
            print(f"    Source: {r.source}  |  Section: {r.section}")
            print(f"    Distance: {r.distance:.4f}")
            print(f"    Preview:  {r.preview(100)}")
        print()

    print("=" * 60)
    print("  PHASE 9 KNOWLEDGE BASE BUILD COMPLETE")
    print("  STOP — Do NOT proceed to Phase 10 without approval.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
