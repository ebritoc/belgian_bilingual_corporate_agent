from typing import Optional, List, Dict, Any


class RetrievalAgent:
    def __init__(self, chroma_path: str, embedding_model_name: str):
        self.chroma_path = chroma_path
        self.embedding_model_name = embedding_model_name
        self._embedding_model = None
        self._chroma_client = None
        self.collection = None
        self.keyword_index: Dict[str, str] = {}

    # ---- Lazy init ----
    def _ensure_embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            print("Loading multilingual embedding model...")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)

    def _ensure_chroma_client(self):
        if self._chroma_client is None:
            import chromadb
            self._chroma_client = chromadb.PersistentClient(path=str(self.chroma_path))

    def _create_embedding_function(self):
        class MultilingualEmbedding:
            def __init__(self, model):
                self.model = model
            def name(self):
                return "default"
            def __call__(self, input):
                if isinstance(input, str):
                    inputs = [input]
                else:
                    inputs = list(input)
                embs = self.model.encode(inputs, show_progress_bar=False)
                out = []
                for v in embs:
                    try:
                        out.append([float(x) for x in v.tolist()])
                    except Exception:
                        out.append([float(x) for x in v])
                return out
            def embed_documents(self, input):
                if isinstance(input, str):
                    input = [input]
                return self.__call__(input)
            def embed_query(self, input):
                if isinstance(input, str):
                    input = [input]
                return self.__call__(input)
        return MultilingualEmbedding(self._embedding_model)

    def ensure_ready(self):
        if self.collection is None:
            self._ensure_embedding_model()
            self._ensure_chroma_client()
            emb_fn = self._create_embedding_function()
            self.collection = self._chroma_client.get_or_create_collection(
                name="banking_reports_improved",
                embedding_function=emb_fn,
                metadata={"hnsw:space": "cosine"}
            )

    # ---- Indexing ----
    def index(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        self.ensure_ready()
        for i, c in enumerate(chunks):
            self.keyword_index[f"chunk_{i}"] = c['text'].lower()
        for start in range(0, len(chunks), batch_size):
            end = min(start + batch_size, len(chunks))
            batch = chunks[start:end]
            self.collection.add(
                documents=[c['text'] for c in batch],
                metadatas=[c['metadata'] for c in batch],
                ids=[f"chunk_{i}" for i in range(start, end)]
            )
            print(f"  ✓ Indexed chunks {start}-{end-1}")

    # ---- Retrieval ----
    def _keyword_search(self, query: str, top_k: int = 10) -> List[str]:
        import re
        terms = set(re.findall(r'\b\w{3,}\b', query.lower()))
        scores: Dict[str, float] = {}
        for doc_id, text in self.keyword_index.items():
            matches = sum(1 for t in terms if t in text)
            if matches > 0:
                scores[doc_id] = matches / max(len(terms), 1)
        return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]

    def hybrid_search(self, query: str, n_results: int = 8, language_filter: Optional[str] = None) -> Dict[str, Any]:
        self.ensure_ready()
        where = {"language": language_filter} if language_filter else None
        vector = self.collection.query(
            query_texts=[query], n_results=n_results*2, where=where,
            include=['documents','metadatas','distances']
        )
        keyword_ids = self._keyword_search(query, top_k=n_results)
        vector_ids = set(vector['ids'][0]) if vector.get('ids') else set()
        combined = list(vector_ids & set(keyword_ids))
        combined += [i for i in vector['ids'][0] if i not in combined][:n_results-len(combined)]
        combined += [i for i in keyword_ids if i not in combined][:n_results-len(combined)]
        if not combined:
            return vector
        final = self.collection.get(ids=combined[:n_results], include=['documents','metadatas'])
        return final

    def multi_retrieve(self, variants: List[Dict[str,str]], n_per_lang: int = 4) -> Dict[str, Any]:
        buckets: Dict[str, Any] = {}
        merged_docs: List[str] = []
        merged_metas: List[Dict[str, Any]] = []
        seen = set()
        for v in variants:
            lang = v['lang']
            q = v['text']
            res = self.hybrid_search(q, n_results=n_per_lang, language_filter=lang)
            docs = res.get('documents', [])
            metas = res.get('metadatas', [])
            buckets[lang] = {'documents': docs, 'metadatas': metas}
            for doc, meta in zip(docs, metas):
                sid = f"{meta.get('source','')}:{meta.get('page','')}:{lang}"
                if sid in seen:
                    continue
                seen.add(sid)
                merged_docs.append(doc)
                merged_metas.append(meta)
        return {'merged': {'documents': merged_docs, 'metadatas': merged_metas}, 'buckets': buckets}

    # ---- Diagnostics ----
    def collection_info(self) -> Dict[str, Any]:
        return {
            'chroma_path': str(self.chroma_path),
            'embedding_model': self.embedding_model_name,
            'collection_name': 'banking_reports_improved',
            'collection_exists': self.collection is not None
        }
