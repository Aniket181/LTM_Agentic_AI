"""
embeddings.py — Phase 9: Ollama Embedding Configuration

RESPONSIBILITIES:
  Provide a single, configurable embedding function for ChromaDB using
  Ollama (local, no API key required).

CONFIGURATION:
  The configuration is read from environment variables:
      RAG_EMBEDDING_PROVIDER=ollama
      RAG_EMBEDDING_MODEL=nomic-embed-text
      RAG_OLLAMA_HOST=http://localhost:11434

  If not set, it defaults to the above Ollama configuration.

DESIGN:
  - The EmbeddingConfig dataclass is the single source of truth for model
    selection.
  - get_embedding_function() returns a ChromaDB-compatible embedding function
    created once and reused across all vector-store operations.
  - OllamaEmbeddingFunction strictly adheres to ChromaDB's EmbeddingFunction
    protocol.

NO LLM GENERATION:
  Ollama is used purely for generating embeddings (nomic-embed-text).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env if present (follows existing project convention)
load_dotenv()


# ── Configuration ─────────────────────────────────────────────────────────────

#: Default embedding model. Optimized for text retrieval.
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


@dataclass
class EmbeddingConfig:
    """
    Configuration for the local embedding model via Ollama.

    Reads from environment variables.
    Falls back to defaults if not set.

    Attributes:
        model_name: Ollama model name (e.g., nomic-embed-text).
        host: Ollama server host URL.
    """
    model_name: str = ""
    host: str = ""

    def __post_init__(self) -> None:
        if not self.model_name:
            self.model_name = os.environ.get("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        if not self.host:
            self.host = os.environ.get("RAG_OLLAMA_HOST", DEFAULT_OLLAMA_HOST)


def get_embedding_function(config: EmbeddingConfig | None = None):
    """
    Return a ChromaDB-compatible Ollama embedding function.

    The function is created from the model specified in EmbeddingConfig.
    If config is None, a default EmbeddingConfig is used.

    Args:
        config: EmbeddingConfig instance. Uses defaults if None.

    Returns:
        OllamaEmbeddingFunction instance.

    Raises:
        ImportError: if chromadb or ollama is not installed.
    """
    if config is None:
        config = EmbeddingConfig()

    try:
        from chromadb.api.types import EmbeddingFunction
    except ImportError as e:
        raise ImportError(
            "chromadb is required for Phase 9 RAG. "
            "Install with: pip install chromadb>=0.4.0"
        ) from e

    class OllamaEmbeddingFunction(EmbeddingFunction):
        """
        Custom wrapper for Ollama that strictly conforms to
        ChromaDB's EmbeddingFunction interface, avoiding signature validation errors.
        """
        def __init__(self, model_name: str, host: str):
            try:
                from ollama import Client
            except ImportError as exc:
                raise ImportError(
                    "ollama is required for local embeddings. "
                    "Install with: pip install ollama>=0.1.0"
                ) from exc
            
            self.model_name = model_name
            self.host = host
            self._client = Client(host=host)

        @staticmethod
        def name() -> str:
            return "OllamaEmbedding"

        def get_config(self) -> dict:
            return {"model_name": self.model_name, "host": self.host}

        @staticmethod
        def build_from_config(config: dict) -> "OllamaEmbeddingFunction":
            return OllamaEmbeddingFunction(
                model_name=config.get("model_name", DEFAULT_EMBEDDING_MODEL),
                host=config.get("host", DEFAULT_OLLAMA_HOST)
            )

        def __call__(self, input: list[str]) -> list[list[float]]:
            # 'input' must be the exact parameter name to pass ChromaDB signature checks
            embeddings = []
            for text in input:
                try:
                    response = self._client.embeddings(model=self.model_name, prompt=text)
                    if 'embedding' not in response:
                        raise RuntimeError(f"Ollama response missing 'embedding' for model {self.model_name}.")
                    embeddings.append(response['embedding'])
                except Exception as e:
                    raise RuntimeError(f"Failed to generate embedding with Ollama model '{self.model_name}' at {self.host}. Error: {e}")
            return embeddings

    return OllamaEmbeddingFunction(model_name=config.model_name, host=config.host)
