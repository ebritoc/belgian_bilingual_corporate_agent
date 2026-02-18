"""
Retriever abstraction for the ColBERT retrieval backend.

Defines a Protocol for retrievers and a factory function that returns
a ColBERTRetriever instance.
"""
from dataclasses import dataclass
from typing import Protocol, Optional, List, Dict, Any
import numpy as np


@dataclass
class RetrievedPassage:
    """A retrieved passage with token-level embeddings for MaxSimE explanations."""
    content: str
    metadata: Dict[str, Any]  # bank, year, language, page
    score: float
    # Token-level embeddings provided by ColBERT
    query_embeddings: Optional[np.ndarray] = None  # (Q_tokens, dim)
    doc_embeddings: Optional[np.ndarray] = None    # (D_tokens, dim)
    query_token_strings: Optional[List[str]] = None
    doc_token_strings: Optional[List[str]] = None


class Retriever(Protocol):
    """Protocol defining the retriever interface."""

    def index_documents(
        self,
        chunks: List[str],
        chunk_ids: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        """Index documents into the retrieval system."""
        ...

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        language_filter: Optional[str] = None
    ) -> List[RetrievedPassage]:
        """Retrieve relevant passages for a query."""
        ...

    def collection_info(self) -> Dict[str, Any]:
        """Get information about the indexed collection."""
        ...


def create_retriever(config: Optional[Dict[str, Any]] = None) -> Retriever:
    """
    Factory function returning a ColBERTRetriever.

    Args:
        config: Configuration dict with ColBERT settings (colbert_model,
                index_folder, index_name). See ColBERTRetriever for details.

    Returns:
        A ColBERTRetriever instance.
    """
    config = config or {}
    retriever_type = config.get('retriever_type', 'colbert')
    if retriever_type != 'colbert':
        raise ValueError(
            f"Unknown retriever type: {retriever_type!r}. "
            "Only 'colbert' is supported."
        )
    from colbert_retriever import ColBERTRetriever
    return ColBERTRetriever(config)
