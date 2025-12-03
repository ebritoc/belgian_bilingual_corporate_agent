"""RAG Agent

Lightweight agent that lazy-loads heavy dependencies and exposes a
minimal public API: index_documents(pdf_dir) and answer_question(question).

This file avoids importing heavy libraries at module import time so tests
and other agents can import it cheaply.
"""
from typing import Optional, List, Dict, Any
from pathlib import Path
import os


class RAGAgent:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Create the RAGAgent.

        config keys (optional):
        - 'ollama_url' (default 'http://localhost:11434')
        - 'model_name' (default 'qwen2.5:7b')
        - 'chroma_path' (default './banking_db_v2')
        - 'embedding_model' (default 'paraphrase-multilingual-mpnet-base-v2')
        """
        config = config or {}
        self.ollama_url = config.get('ollama_url', 'http://localhost:11434')
        self.model_name = config.get('model_name', 'qwen2.5:7b')
        self.chroma_path = config.get('chroma_path', './banking_db_v2')
        self.embedding_model_name = config.get('embedding_model', 'paraphrase-multilingual-mpnet-base-v2')
        self.ollama_timeout = int(config.get('ollama_timeout', 180))
        self.enable_bilingual_check = bool(config.get('enable_bilingual_check', False))

        # Lazy attributes
        self._embedding_model = None
        self._chroma_client = None
        self.collection = None

        # Simple keyword index for hybrid retrieval
        self.keyword_index: Dict[str, str] = {}

    # ------------------- Lazy initializers -------------------
    def _ensure_embedding_model(self):
        if self._embedding_model is None:
            # Import here to avoid heavy imports at module import time
            from sentence_transformers import SentenceTransformer
            print("Loading multilingual embedding model...")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)

    def _ensure_chroma_client(self):
        if self._chroma_client is None:
            import chromadb
            self._chroma_client = chromadb.PersistentClient(path=str(self.chroma_path))

    def _create_embedding_function(self):
        # Inner wrapper that conforms to Chroma's EmbeddingFunction interface
        class MultilingualEmbedding:
            def __init__(self, model):
                self.model = model

            def name(self):
                return "default"

            def __call__(self, input):
                # Normalize to list of strings
                if isinstance(input, str):
                    inputs = [input]
                else:
                    inputs = list(input)
                embs = self.model.encode(inputs, show_progress_bar=False)
                processed = []
                for v in embs:
                    try:
                        processed.append([float(x) for x in v.tolist()])
                    except Exception:
                        processed.append([float(x) for x in v])
                return processed  # list[list[float]]

            def embed_documents(self, input):
                if isinstance(input, str):
                    input = [input]
                return self.__call__(input)  # list[list[float]]

            def embed_query(self, input):
                # Chroma expects list[list[float]] even for single query
                if isinstance(input, str):
                    input = [input]
                return self.__call__(input)  # list[list[float]]

        return MultilingualEmbedding(self._embedding_model)

    def _ensure_collection(self):
        if self.collection is None:
            self._ensure_embedding_model()
            self._ensure_chroma_client()
            embedding_fn = self._create_embedding_function()
            # Create or get collection
            self.collection = self._chroma_client.get_or_create_collection(
                name="banking_reports_improved",
                embedding_function=embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )

    # ------------------- Utilities / Public -------------------
    def collection_info(self) -> Dict[str, Any]:
        """Return diagnostic info about the collection and embedding.

        This is useful to detect DB mismatches before indexing.
        """
        info = {
            'chroma_path': str(Path(self.chroma_path).absolute()),
            'embedding_model': self.embedding_model_name,
            'collection_name': 'banking_reports_improved'
        }
        try:
            if self.collection is not None:
                info['collection_exists'] = True
        except Exception:
            info['collection_exists'] = False
        return info

    # ------------------- Core functionality -------------------
    def extract_pdf_content(self, pdf_path: Path, bank: str, language: str):
        # Import pypdf here
        from pypdf import PdfReader

        reader = PdfReader(pdf_path)
        chunks = []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text or not text.strip():
                continue

            clean_text = self._advanced_clean(text)
            if not clean_text or len(clean_text) < 100:
                continue

            page_chunks = self._semantic_chunk(clean_text, max_size=1000, min_size=200)
            for chunk in page_chunks:
                if self._is_quality_chunk(chunk):
                    chunks.append({
                        'text': chunk,
                        'metadata': {
                            'bank': bank,
                            'language': language,
                            'page': page_num + 1,
                            'source': pdf_path.name
                        }
                    })

        return chunks

    def _advanced_clean(self, text: str) -> str:
        import re
        lines = text.splitlines()
        cleaned = []
        skip_patterns = [r'^\s*\d+\s*$', r'^(TABLE OF CONTENTS|CONTENU|INHOUD|INDEX)', r'^\s*\.{3,}', r'^\s*_{3,}', r'^\s*-{3,}', r'^\s*(RAPPORT|ANNUEL|ANNUAL|REPORT|JAARVERSLAG)\s*$']
        for line in lines:
            line = line.strip()
            if not line or len(line) < 3:
                continue
            if any(__import__('re').match(p, line, __import__('re').IGNORECASE) for p in skip_patterns):
                continue
            if len(line) < 50 and line.isupper() and not any(c.isdigit() for c in line):
                continue
            alpha_ratio = sum(c.isalpha() for c in line) / len(line) if line else 0
            if alpha_ratio < 0.4:
                continue
            cleaned.append(line)
        return ' '.join(cleaned)

    def _semantic_chunk(self, text: str, max_size=1000, min_size=200) -> List[str]:
        import re
        text = re.sub(r'\s+', ' ', text).strip()
        segments = re.split(r'(?<=[.!?])\s+(?=[A-ZÀ-Ö])|(?:\n\s*){2,}', text)
        chunks = []
        current = ""
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            if len(current) + len(seg) + 1 < max_size:
                current += " " + seg if current else seg
            else:
                if len(current) >= min_size:
                    chunks.append(current)
                    current = seg
                else:
                    current += " " + seg if current else seg
                    if len(current) >= min_size:
                        chunks.append(current)
                        current = ""
        if len(current) >= min_size:
            chunks.append(current)
        return chunks

    def _is_quality_chunk(self, chunk: str) -> bool:
        import re
        from collections import Counter
        if len(chunk) < 150:
            return False
        words = re.findall(r'\b[a-zA-ZÀ-ÖØ-öø-ÿ]{3,}\b', chunk)
        if len(words) < 20:
            return False
        word_counts = Counter(w.lower() for w in words)
        most_common_freq = word_counts.most_common(1)[0][1] if word_counts else 0
        if most_common_freq > len(words) * 0.3:
            return False
        financial_terms = r'\b(million|billion|ratio|capital|revenue|profit|loss|assets|equity|risk|performance|résultat|bénéfice|actif|capitaux|risque|winst|verlies|vermogen)\b'
        if not __import__('re').search(financial_terms, chunk, __import__('re').IGNORECASE):
            sentences = re.split(r'[.!?]+', chunk)
            avg_sentence_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
            if avg_sentence_len < 5:
                return False
        return True

    def index_documents(self, pdf_directory: str):
        from pathlib import Path
        pdf_directory = Path(pdf_directory)
        pdf_configs = [
            ('BNP_Paribas_Fortis_FR.pdf', 'BNP Paribas Fortis', 'fr'),
            ('BNP_Paribas_Fortis_NL.pdf', 'BNP Paribas Fortis', 'nl'),
            ('BNP_Paribas_Fortis_EN.pdf', 'BNP Paribas Fortis', 'en'),
            ('KBC_Group_FR.pdf', 'KBC Group', 'fr'),
            ('KBC_Group_NL.pdf', 'KBC Group', 'nl'),
            ('KBC_Group_EN.pdf', 'KBC Group', 'en'),
        ]

        all_chunks = []
        for filename, bank, lang in pdf_configs:
            pdf_path = pdf_directory / filename
            if pdf_path.exists():
                print(f"Processing {filename}...")
                chunks = self.extract_pdf_content(pdf_path, bank, lang)
                all_chunks.extend(chunks)
            else:
                print(f"Warning: {filename} not found, skipping...")

        if not all_chunks:
            print("No documents to index!")
            return

        print(f"\nIndexing {len(all_chunks)} chunks with multilingual embeddings...")

        for i, chunk in enumerate(all_chunks):
            doc_id = f"chunk_{i}"
            self.keyword_index[doc_id] = chunk['text'].lower()

        # Ensure collection ready
        self._ensure_collection()

        batch_size = 100
        for start_idx in range(0, len(all_chunks), batch_size):
            end_idx = min(start_idx + batch_size, len(all_chunks))
            batch = all_chunks[start_idx:end_idx]
            self.collection.add(
                documents=[c['text'] for c in batch],
                metadatas=[c['metadata'] for c in batch],
                ids=[f"chunk_{i}" for i in range(start_idx, end_idx)]
            )
            print(f"  ✓ Indexed chunks {start_idx}-{end_idx-1}")

        print(f"✓ Successfully indexed {len(all_chunks)} chunks!")

    def _keyword_search(self, query: str, top_k: int = 10) -> List[str]:
        import re
        query_terms = set(re.findall(r'\b\w{3,}\b', query.lower()))
        scores = {}
        for doc_id, text in self.keyword_index.items():
            matches = sum(1 for term in query_terms if term in text)
            if matches > 0:
                score = matches / len(query_terms)
                scores[doc_id] = score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, score in sorted_docs[:top_k]]

    def hybrid_retrieve(self, query: str, n_results: int = 8, language_filter: Optional[str] = None):
        where_filter = {"language": language_filter} if language_filter else None
        self._ensure_collection()

        vector_results = self.collection.query(
            query_texts=[query],
            n_results=n_results * 2,
            where=where_filter,
            include=['documents', 'metadatas', 'distances']
        )

        keyword_ids = self._keyword_search(query, top_k=n_results)

        vector_ids = set(vector_results['ids'][0]) if vector_results.get('ids') else set()
        keyword_id_set = set(keyword_ids)

        combined_ids = list(vector_ids & keyword_id_set)
        combined_ids += [id for id in vector_results['ids'][0] if id not in combined_ids][:n_results - len(combined_ids)]
        combined_ids += [id for id in keyword_ids if id not in combined_ids][:n_results - len(combined_ids)]

        if not combined_ids:
            return vector_results

        final_results = self.collection.get(
            ids=combined_ids[:n_results],
            include=['documents', 'metadatas']
        )

        return final_results

    def query_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        import requests
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "top_p": 0.9}
            }
            if system_prompt:
                payload["system"] = system_prompt
            response = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=self.ollama_timeout)
            if response.status_code == 200:
                return response.json().get('response', '')
            return f"Error: Ollama returned status {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"

    def answer_question(self, user_question: str, n_results: int = 8, use_llm: bool = True, bilingual_check: Optional[bool] = None):
        print(f"\n{'='*70}")
        print(f"Question: {user_question}")
        print(f"{'='*70}\n")

        detected_lang = self._detect_language(user_question)
        print(f"Detected language: {detected_lang.upper()}\n")

        print("Performing hybrid search (vector + keyword)...")
        if bilingual_check is None:
            bilingual_check = self.enable_bilingual_check
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

        context_blocks = []
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            context_blocks.append(f"[Source {i+1}: {meta['bank']}, Page {meta['page']}, {meta['language'].upper()}]\n{doc}")

        context = "\n\n---\n\n".join(context_blocks)

        system_prompt = """You are a financial analyst assistant specializing in European banking reports.

Your task:
- Answer questions using ONLY the provided context
- Always cite sources (bank name, page number)
- If information is not in the context, explicitly state this
- Provide specific numbers and facts when available
- Keep answers concise but complete

Respond in the same language as the question.

If there is a cross-language consistency warning, mention it briefly at the end.
"""

        warning = ""
        if consistency.get('status') == 'discrepancy':
            warning = f"\n\nNote: Cross-language consistency check flagged a potential discrepancy ({consistency.get('notes')})."

        user_prompt = f"""Based on the following excerpts from banking reports, please answer the question.

Context:

{context}

---

Question: {user_question}

Provide a clear, factual answer with citations.{warning}"""

        if use_llm:
            print("Generating answer with local LLM...")
            answer = self.query_ollama(user_prompt, system_prompt)
        else:
            print("LLM disabled; returning sources only.")
            answer = "LLM disabled. Review sources below."
        return {'answer': answer, 'sources': metas, 'language': detected_lang, 'num_sources': len(docs), 'consistency': consistency}

    # ------------------- Bilingual Consistency -------------------
    def detect_and_expand_query(self, text: str) -> Dict[str, Any]:
        """Detect language and create FR/NL/EN variants for cross-language retrieval."""
        detected = self._detect_language(text)
        def translate_tiny(q: str, src: str, target: str) -> str:
            ql = q.lower()
            mappings = {
                ('fr','nl'): [ ('résultat annuel','jaarresultaat'), ('résultat net','nettowinst') ],
                ('nl','fr'): [ ('jaarresultaat','résultat annuel'), ('nettowinst','résultat net') ],
                ('en','fr'): [ ('annual result','résultat annuel'), ('net profit','résultat net') ],
                ('en','nl'): [ ('annual result','jaarresultaat'), ('net profit','nettowinst') ],
            }
            text_out = q
            for a,b in mappings.get((src,target), []):
                if a in ql:
                    text_out = text_out.lower().replace(a, b)
            return text_out
        variants: List[Dict[str,str]] = []
        if detected == 'fr':
            variants = [ {'lang':'fr','text':text}, {'lang':'nl','text':translate_tiny(text,'fr','nl')}, {'lang':'en','text':translate_tiny(text,'fr','en')} ]
        elif detected == 'nl':
            variants = [ {'lang':'nl','text':text}, {'lang':'fr','text':translate_tiny(text,'nl','fr')}, {'lang':'en','text':translate_tiny(text,'nl','en')} ]
        else:
            variants = [ {'lang':'en','text':text}, {'lang':'fr','text':translate_tiny(text,'en','fr')}, {'lang':'nl','text':translate_tiny(text,'en','nl')} ]
        return {'detected': detected, 'variants': variants}

    def multi_retrieve(self, variants: List[Dict[str,str]], n_per_lang: int = 4):
        buckets: Dict[str, Any] = {}
        merged_docs: List[str] = []
        merged_metas: List[Dict[str,Any]] = []
        seen: set = set()
        for v in variants:
            lang = v['lang']
            q = v['text']
            res = self.hybrid_retrieve(q, n_results=n_per_lang, language_filter=lang)
            docs = res.get('documents', [])
            metas = res.get('metadatas', [])
            buckets[lang] = {'documents': docs, 'metadatas': metas}
            # Normalize and merge by source+page
            for i, (doc, meta) in enumerate(zip(docs, metas)):
                sid = f"{meta.get('source','')}:{meta.get('page','')}:{lang}"
                if sid in seen:
                    continue
                seen.add(sid)
                merged_docs.append(doc)
                merged_metas.append(meta)
        return {'merged': {'documents': merged_docs, 'metadatas': merged_metas}, 'buckets': buckets}

    def compare_across_languages(self, buckets: Dict[str,Any]) -> Dict[str, Any]:
        import re
        def extract_numbers(t: str) -> List[str]:
            return re.findall(r"\b\d+[\d.,]*\b", t)
        nums: Dict[str,set] = {}
        for lang, data in buckets.items():
            docs = data.get('documents', [])
            flat: List[str] = []
            for d in docs:
                if isinstance(d, list):
                    flat.extend(d)
                else:
                    flat.append(d)
            s = set()
            for d in flat[:3]:
                for n in extract_numbers(d):
                    s.add(n)
            nums[lang] = s
        if len(nums) < 2:
            return {'status':'ok','confidence':0.5,'notes':'Insufficient language coverage'}
        sets = list(nums.values())
        common = set.intersection(*sets) if sets else set()
        if common:
            return {'status':'ok','confidence':0.8,'notes':'Numeric overlap across languages'}
        any_nums = any(len(s)>0 for s in sets)
        if any_nums:
            return {'status':'discrepancy','confidence':0.7,'notes':'Different numeric figures across languages'}
        return {'status':'ok','confidence':0.6,'notes':'No numeric figures detected'}

    def _detect_language(self, text: str) -> str:
        text_lower = text.lower()
        import re
        fr_count = len(re.findall(r'\b(le|la|les|des|un|une|du|avec|pour|que|qui|dans|sur|est|sont|ont|leur|cette)\b', text_lower))
        nl_count = len(re.findall(r'\b(de|het|een|van|en|met|voor|dat|die|zijn|is|was|op|in|bij|heeft|werd)\b', text_lower))
        en_count = len(re.findall(r'\b(the|is|are|and|with|for|that|this|from|have|was|were|has|been)\b', text_lower))
        max_count = max(fr_count, nl_count, en_count)
        if max_count == 0:
            return 'en'
        elif fr_count == max_count:
            return 'fr'
        elif nl_count == max_count:
            return 'nl'
        else:
            return 'en'
