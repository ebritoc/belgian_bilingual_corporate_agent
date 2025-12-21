"""RAG Agent facade

Delegates responsibilities to agentic modules via RAGOrchestrator.
Public API remains: index_documents and answer_question.
"""
from typing import Optional, Dict, Any

from .rag_orchestrator import RAGOrchestrator


class RAGAgent:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._orch = RAGOrchestrator(self._config)

    # Public API (kept stable)
    def index_documents(self, pdf_directory: str):
        return self._orch.index_documents(pdf_directory)

    def answer_question(self, user_question: str, n_results: int = 8, use_llm: bool = True, bilingual_check: Optional[bool] = None):
        print(f"\n{'='*70}")
        print(f"Question: {user_question}")
        print(f"{'='*70}\n")

        detected_lang = self._orch._detect_language(user_question)
        print(f"Detected language: {detected_lang.upper()}\n")

        if bilingual_check is None:
            bilingual_check = self._config.get('enable_bilingual_check', False)

        print("Performing hybrid search (vector + keyword)...")
        if bilingual_check:
            exp = self.detect_and_expand_query(user_question)
            multi = self.multi_retrieve(exp['variants'], n_per_lang=max(2, n_results//3))
            results = multi['merged']
            consistency = self.compare_across_languages(multi['buckets'])
        else:
            results = self.hybrid_retrieve(user_question, n_results=n_results)
            consistency = {'status':'skip','confidence':0.0,'notes':'Bilingual check disabled'}

        docs = results.get('documents', [])
        metas = results.get('metadatas', [])
        if not docs:
            return {"answer": "No relevant information found.", "sources": [], "language": detected_lang}

        ctx_blocks = []
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            ctx_blocks.append(f"[Source {i+1}: {meta['bank']}, Page {meta['page']}, {meta['language'].upper()}]\n{doc}")
        context = "\n\n---\n\n".join(ctx_blocks)

        warn = ""
        if consistency.get('status') == 'discrepancy':
            warn = f"\n\nNote: Cross-language consistency check flagged a potential discrepancy ({consistency.get('notes')})."
        system_prompt, user_prompt = self._orch.generator.build_prompts(context, user_question, consistency_note=warn)

        if use_llm:
            print("Generating answer with local LLM...")
            answer = self._orch.generator.generate(user_prompt, system_prompt)
        else:
            print("LLM disabled; returning sources only.")
            answer = "LLM disabled. Review sources below."

        return {'answer': answer, 'sources': metas, 'language': detected_lang, 'num_sources': len(docs), 'consistency': consistency}

    def collection_info(self) -> Dict[str, Any]:
        return self._orch.retrieval.collection_info()

    # Backwards-compat small helpers (used in tests/extensions)
    def hybrid_retrieve(self, query: str, n_results: int = 8, language_filter: Optional[str] = None):
        return self._orch.retrieval.hybrid_search(query, n_results=n_results, language_filter=language_filter)

    def detect_and_expand_query(self, text: str) -> Dict[str, Any]:
        return self._orch.detect_and_expand_query(text)

    def multi_retrieve(self, variants, n_per_lang: int = 4):
        return self._orch.retrieval.multi_retrieve(variants, n_per_lang=n_per_lang)

    def compare_across_languages(self, buckets):
        return self._orch.compare_numeric_across_languages(buckets)

    def _detect_language(self, text: str) -> str:
        return self._orch._detect_language(text)
