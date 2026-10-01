"""
documents.py — Phase 9: Knowledge Document Loader & Chunker

RESPONSIBILITIES:
  1. Discover Markdown documents in knowledge_base/.
  2. Load each document and extract its title from the first H1 heading.
  3. Split the document into heading-aware sections (each section = one chunk).
  4. Attach structured metadata to every chunk.
  5. Assign deterministic, content-independent chunk IDs based on source
     filename and chunk index.

CHUNKING STRATEGY:
  Heading-aware section splitting:
    - The document is split at every line that starts with "##" (level-2
      heading or deeper). This keeps each major section logically together.
    - If a section exceeds MAX_SECTION_CHARS, it is further split on
      paragraph boundaries (blank lines), then on sentence boundaries.
    - This is deterministic: the same document always produces the same chunks.

  Rationale:
    The knowledge-base documents are short technical Markdown files organised
    by section heading. Keeping sections intact produces semantically coherent
    chunks that retrieve well for section-level queries.

METADATA SCHEMA (per chunk):
  {
      "source":        "<filename>.md",
      "document_type": "bearing_faults" | "maintenance_guidelines" | ...,
      "title":         "<document H1 title>",
      "section":       "<H2 section heading or 'preamble'>",
      "chunk_index":   <int, 0-based within document>,
      "char_count":    <int>,
  }

DETERMINISM:
  Chunk IDs are computed as SHA-256( "<filename>_<chunk_index>" ).
  The same document + same chunk_index always produces the same ID,
  regardless of content changes that do not change ordering.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Configuration ─────────────────────────────────────────────────────────────

#: Maximum characters per chunk. Sections exceeding this are split further.
MAX_SECTION_CHARS: int = 800

#: Minimum characters for a chunk to be kept (avoids near-empty chunks).
MIN_CHUNK_CHARS: int = 20

#: Expected knowledge-base filenames (used for document_type extraction).
KNOWN_DOC_TYPES: dict[str, str] = {
    "bearing_faults.md":         "bearing_faults",
    "maintenance_guidelines.md": "maintenance_guidelines",
    "troubleshooting.md":        "troubleshooting",
    "equipment_conditions.md":   "equipment_conditions",
}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """
    A single chunk extracted from a knowledge-base Markdown document.

    Attributes:
        chunk_id:  Deterministic SHA-256 ID derived from source + chunk_index.
        content:   Raw text content of the chunk (may include Markdown).
        metadata:  Structured metadata dict (source, title, section, etc.).
    """
    chunk_id:  str
    content:   str
    metadata:  dict


# ── Public API ────────────────────────────────────────────────────────────────

def load_documents(knowledge_base_dir: Path) -> list[DocumentChunk]:
    """
    Load all Markdown documents from knowledge_base_dir and return chunks.

    Args:
        knowledge_base_dir: Path to the directory containing *.md files.

    Returns:
        List of DocumentChunk objects, ordered by file then chunk index.

    Raises:
        FileNotFoundError: if knowledge_base_dir does not exist.
        ValueError: if no Markdown files are found.
    """
    knowledge_base_dir = Path(knowledge_base_dir)
    if not knowledge_base_dir.exists():
        raise FileNotFoundError(
            f"Knowledge base directory not found: {knowledge_base_dir}"
        )

    md_files = sorted(knowledge_base_dir.glob("*.md"))
    if not md_files:
        raise ValueError(
            f"No Markdown files found in {knowledge_base_dir}"
        )

    all_chunks: list[DocumentChunk] = []
    for md_path in md_files:
        chunks = _chunk_document(md_path)
        all_chunks.extend(chunks)

    return all_chunks


def make_chunk_id(source_filename: str, chunk_index: int) -> str:
    """
    Generate a deterministic chunk ID.

    ID = SHA-256( "<source_filename>_<chunk_index>" )[:32]

    Using the first 32 hex chars of SHA-256 gives 128-bit collision resistance,
    which is more than sufficient for a small local knowledge base.

    Args:
        source_filename: Filename of the source document (e.g. "bearing_faults.md").
        chunk_index:     0-based index of this chunk within the document.

    Returns:
        32-character lowercase hex string.
    """
    raw = f"{source_filename}_{chunk_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


# ── Internal functions ────────────────────────────────────────────────────────

def _extract_title(lines: list[str]) -> str:
    """Return the text of the first H1 heading, or the filename stem."""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            return stripped[2:].strip()
    return ""


def _extract_document_type(filename: str) -> str:
    """Return the document_type string for a given filename."""
    return KNOWN_DOC_TYPES.get(filename, Path(filename).stem)


def _split_into_sections(lines: list[str]) -> list[tuple[str, str]]:
    """
    Split document lines into (section_heading, section_text) pairs.

    The preamble (text before the first ## heading) is assigned section_heading
    equal to the document title (or 'preamble').

    Returns:
        List of (heading, text) tuples.
    """
    sections: list[tuple[str, str]] = []
    current_heading = "preamble"
    current_lines: list[str] = []

    for line in lines:
        # Match ## or deeper heading (but not # which is the doc title)
        if re.match(r"^#{2,}\s", line):
            # Save current section
            text = "\n".join(current_lines).strip()
            if text:
                sections.append((current_heading, text))
            # Start new section
            current_heading = line.strip().lstrip("#").strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    # Flush last section
    text = "\n".join(current_lines).strip()
    if text:
        sections.append((current_heading, text))

    return sections


def _split_long_section(section_text: str, max_chars: int = MAX_SECTION_CHARS) -> list[str]:
    """
    Split a section that exceeds max_chars into smaller chunks.

    Strategy:
      1. Try splitting on blank lines (paragraph boundaries).
      2. If any paragraph still exceeds max_chars, split on ". " (sentences).
      3. Any remainder is kept as a trailing chunk.

    Always deterministic.
    """
    if len(section_text) <= max_chars:
        return [section_text]

    # Step 1: split on blank lines
    paragraphs = re.split(r"\n\s*\n", section_text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 <= max_chars:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            if len(para) > max_chars:
                # Step 2: split on sentences
                sentences = re.split(r"(?<=\.)\s+", para)
                sent_chunk = ""
                for sent in sentences:
                    if len(sent_chunk) + len(sent) + 1 <= max_chars:
                        sent_chunk = (sent_chunk + " " + sent).strip()
                    else:
                        if sent_chunk:
                            chunks.append(sent_chunk)
                        sent_chunk = sent
                if sent_chunk:
                    chunks.append(sent_chunk)
                current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS] or [section_text]


def _chunk_document(md_path: Path) -> list[DocumentChunk]:
    """
    Load a single Markdown file and return its list of DocumentChunks.

    Chunking is heading-aware:
      - Each H2+ section becomes at least one chunk.
      - Sections longer than MAX_SECTION_CHARS are split further.
    """
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    filename = md_path.name

    title = _extract_title(lines)
    doc_type = _extract_document_type(filename)

    sections = _split_into_sections(lines)

    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for section_heading, section_text in sections:
        sub_chunks = _split_long_section(section_text, MAX_SECTION_CHARS)
        for sub in sub_chunks:
            if len(sub.strip()) < MIN_CHUNK_CHARS:
                continue
            chunk_id = make_chunk_id(filename, chunk_index)
            metadata = {
                "source":        filename,
                "document_type": doc_type,
                "title":         title,
                "section":       section_heading,
                "chunk_index":   chunk_index,
                "char_count":    len(sub),
            }
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                content=sub,
                metadata=metadata,
            ))
            chunk_index += 1

    return chunks
