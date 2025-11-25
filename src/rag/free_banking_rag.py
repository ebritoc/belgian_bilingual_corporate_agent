"""Deprecated shim for legacy import path.

Use `from agents.rag_agent import RAGAgent` instead of
`from rag.free_banking_rag import FreeBankingRAG`.
This file intentionally lightweight to avoid heavy imports.
"""
import warnings

try:
    from agents.rag_agent import RAGAgent
except Exception as e:  # pragma: no cover
    RAGAgent = None
    _import_error = e
else:
    _import_error = None


class FreeBankingRAG:
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "FreeBankingRAG is deprecated. Import RAGAgent from agents.rag_agent instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if RAGAgent is None:
            raise ImportError(f"Failed to import RAGAgent for shim: {_import_error}")
        self._agent = RAGAgent(*args, **kwargs)

    def index_documents(self, pdf_dir):
        return self._agent.index_documents(pdf_dir)

    def answer_question(self, question, n_results=8):
        return self._agent.answer_question(question, n_results=n_results)
