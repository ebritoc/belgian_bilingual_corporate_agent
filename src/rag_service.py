"""
Simplified RAG Service for Belgian Banking Reports.

Single-file implementation combining document indexing, retrieval, and generation.
"""
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any
from retriever import Retriever, RetrievedPassage, create_retriever
from bilingual_validator import BilingualValidator


class RAGService:
    """Main RAG service for indexing PDFs and answering questions."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        retriever: Optional[Retriever] = None
    ):
        """
        Initialize RAG Service.

        Args:
            config: Optional configuration dict with keys:
                - retriever_type: 'chroma' or 'colbert' (default: 'chroma')
                - chroma_path: Path to ChromaDB persistent storage (default: './banking_db_v2')
                - embedding_model: SentenceTransformer model name (default: 'paraphrase-multilingual-mpnet-base-v2')
                - ollama_url: Ollama API endpoint (default: 'http://localhost:11434')
                - model_name: Ollama model name (default: 'qwen2.5:7b')
                - ollama_timeout: Request timeout in seconds (default: 180)
            retriever: Optional pre-configured Retriever instance. If not provided,
                       one will be created based on config.
        """
        self.config = config or {}
        self.ollama_url = self.config.get('ollama_url', 'http://localhost:11434')
        self.model_name = self.config.get('model_name', 'qwen2.5:7b')
        self.ollama_timeout = int(self.config.get('ollama_timeout', 180))

        # Use provided retriever or create one lazily
        self._retriever = retriever
        self._retriever_config = config
        self._bilingual_validator = None

    def _ensure_initialized(self):
        """Initialize retriever if not already done."""
        if self._retriever is None:
            self._retriever = create_retriever(self._retriever_config)

    def index_documents(self, pdf_directory: str):
        """
        Index all PDF files in the given directory.

        Args:
            pdf_directory: Path to directory containing PDF files

        Expected filename format: BankName_Language.pdf
        Example: BNP_Paribas_Fortis_FR.pdf, KBC_Group_NL.pdf
        """
        self._ensure_initialized()

        from utils.pdf_parser import chunk_pdf

        pdf_dir = Path(pdf_directory)
        if not pdf_dir.exists():
            raise ValueError(f"Directory not found: {pdf_directory}")

        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in: {pdf_directory}")

        print(f"\nFound {len(pdf_files)} PDF files to index")

        all_chunks = []
        for pdf_file in pdf_files:
            # Parse filename: BankName_Language.pdf
            name = pdf_file.stem  # e.g., "BNP_Paribas_Fortis_FR"
            parts = name.rsplit('_', 1)  # Split from right to get language code

            if len(parts) == 2:
                bank = parts[0].replace('_', ' ')  # "BNP Paribas Fortis"
                language = parts[1].lower()  # "fr"
            else:
                # Fallback if filename doesn't match pattern
                bank = name.replace('_', ' ')
                language = 'unknown'

            print(f"\nProcessing: {pdf_file.name}")
            print(f"  Bank: {bank}, Language: {language}")

            chunks = chunk_pdf(pdf_file, bank=bank, language=language)
            print(f"  Extracted {len(chunks)} chunks")
            all_chunks.extend(chunks)

        # Prepare data for retriever
        texts = [c['text'] for c in all_chunks]
        metadatas = [c['metadata'] for c in all_chunks]
        chunk_ids = [f"chunk_{i}" for i in range(len(all_chunks))]

        print(f"\nIndexing {len(all_chunks)} total chunks...")
        self._retriever.index_documents(texts, chunk_ids, metadatas)

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        language_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieve relevant document chunks for a query.

        Args:
            query: Search query
            n_results: Number of results to return
            language_filter: Optional language code to filter by ('fr', 'nl', 'en')

        Returns:
            Dict with 'documents', 'metadatas', 'distances', and 'passages' keys
        """
        self._ensure_initialized()

        passages = self._retriever.retrieve(query, n_results, language_filter)

        # Convert to legacy dict format for backward compatibility
        return {
            'documents': [p.content for p in passages],
            'metadatas': [p.metadata for p in passages],
            'distances': [1.0 - p.score for p in passages],  # Convert score back to distance
            'passages': passages  # Include full RetrievedPassage objects for new features
        }

    def generate_answer(self, context: str, question: str) -> str:
        """
        Generate answer using Ollama LLM.

        Args:
            context: Retrieved document context
            question: User question

        Returns:
            Generated answer string
        """
        system_prompt = (
            "You are a financial analyst assistant specializing in European banking reports.\n\n"
            "Your task:\n"
            "- Answer questions using ONLY the provided context\n"
            "- Always cite sources (bank name, page number)\n"
            "- If information is not in the context, explicitly state this\n"
            "- Provide specific numbers and facts when available\n"
            "- Keep answers concise but complete\n\n"
            "Respond in the same language as the question."
        )

        user_prompt = (
            "Based on the following excerpts from banking reports, please answer the question.\n\n"
            f"Context:\n\n{context}\n\n---\n\n"
            f"Question: {question}\n\n"
            "Provide a clear, factual answer with citations."
        )

        try:
            payload = {
                "model": self.model_name,
                "prompt": user_prompt,
                "system": system_prompt,
                "stream": False,
                "options": {"temperature": 0.3, "top_p": 0.9}
            }

            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=self.ollama_timeout
            )

            if response.status_code == 200:
                return response.json().get('response', 'No response generated')
            else:
                try:
                    error_detail = response.json().get('error', response.text)
                except:
                    error_detail = response.text
                return f"Error: Ollama returned status {response.status_code}: {error_detail}"

        except requests.exceptions.Timeout:
            return "Error: Request to Ollama timed out. Try increasing the timeout or using a faster model."
        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to Ollama. Make sure Ollama is running at " + self.ollama_url
        except Exception as e:
            return f"Error: {str(e)}"

    def answer_question(self, question: str, n_results: int = 5, use_llm: bool = True,
                       enable_bilingual: bool = False) -> Dict[str, Any]:
        """
        Answer a question using RAG pipeline.

        Args:
            question: User question
            n_results: Number of document chunks to retrieve
            use_llm: Whether to use LLM for generation (if False, returns sources only)
            enable_bilingual: Enable bilingual validation and cross-language retrieval

        Returns:
            Dict with keys:
                - answer: Generated answer or sources description
                - sources: List of source metadata dicts
                - num_sources: Number of sources retrieved
                - language: Detected language (if bilingual enabled)
                - consistency: Consistency check results (if bilingual enabled)
        """
        self._ensure_initialized()

        print(f"\n{'='*70}")
        print(f"Question: {question}")
        print(f"{'='*70}\n")

        # Initialize bilingual validator if needed
        detected_lang = None
        consistency = None

        if enable_bilingual:
            if self._bilingual_validator is None:
                self._bilingual_validator = BilingualValidator()

            # Detect language and expand query
            detected_lang = self._bilingual_validator.detect_language(question)
            print(f"Detected language: {detected_lang.upper()}")

            expansion = self._bilingual_validator.expand_query(question)
            print(f"Expanded to {len(expansion['variants'])} language variants\n")

            # Retrieve from each language variant
            print(f"Performing multi-language retrieval...")
            language_results = {}
            all_docs = []
            all_metas = []
            all_passages = []

            for variant in expansion['variants']:
                lang = variant['lang']
                text = variant['text']
                # Retrieve n_results per variant so dedup has enough candidates
                lang_results = self.retrieve(text, n_results=n_results)
                language_results[lang] = lang_results

                # Collect all docs, metadata, and passage objects
                all_docs.extend(lang_results['documents'])
                all_metas.extend(lang_results['metadatas'])
                all_passages.extend(lang_results['passages'])

            # Check numeric consistency
            consistency = self._bilingual_validator.check_numeric_consistency(language_results)
            print(f"Consistency check: {consistency['status']} (confidence: {consistency['confidence']:.2f})")

            # Deduplicate and limit results, keeping passage objects in sync
            seen_texts = set()
            docs = []
            metas = []
            passages = []
            for doc, meta, passage in zip(all_docs, all_metas, all_passages):
                if doc not in seen_texts and len(docs) < n_results:
                    docs.append(doc)
                    metas.append(meta)
                    passages.append(passage)
                    seen_texts.add(doc)

        else:
            # Simple retrieval without bilingual features
            print(f"Retrieving {n_results} relevant passages...")
            results = self.retrieve(question, n_results=n_results)
            docs = results['documents']
            metas = results['metadatas']
            passages = results['passages']

        if not docs:
            return {
                "answer": "No relevant information found in the indexed documents.",
                "sources": [],
                "num_sources": 0
            }

        print(f"[OK] Retrieved {len(docs)} passages\n")

        # Build context from retrieved chunks
        context_blocks = []
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            source_info = f"[Source {i+1}: {meta['bank']}, Page {meta['page']}, {meta['language'].upper()}]"
            context_blocks.append(f"{source_info}\n{doc}")

        context = "\n\n---\n\n".join(context_blocks)

        # Generate answer or return sources
        if use_llm:
            print("Generating answer with LLM...")
            answer = self.generate_answer(context, question)
        else:
            print("LLM disabled; returning sources only.")
            answer = "LLM disabled. Retrieved sources:\n\n" + context

        result = {
            'answer': answer,
            'sources': metas,
            'documents': docs,
            'passages': passages,
            'num_sources': len(docs)
        }

        # Add bilingual information if enabled
        if enable_bilingual:
            result['language'] = detected_lang
            result['consistency'] = consistency

        return result

    def collection_info(self) -> Dict[str, Any]:
        """
        Get information about the indexed collection.

        Returns:
            Dict with collection statistics
        """
        self._ensure_initialized()
        return self._retriever.collection_info()
