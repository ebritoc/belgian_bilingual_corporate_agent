"""
ColBERT-based retriever using colbert-ai and FAISS.

Provides late interaction retrieval with FAISS indexing and token-level
embeddings for MaxSimE explanations.
"""
import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import torch

from retriever import RetrievedPassage

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ColBERTRetriever:
    """
    ColBERT retriever using FAISS for indexing.

    Key features:
    - Late interaction retrieval via ColBERT
    - FAISS flat index for accurate search
    - Returns raw token embeddings for MaxSimE explanations
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ColBERT retriever.

        Args:
            config: Configuration dict with keys:
                - colbert_model: Model name/path (default: 'answerdotai/answerai-colbert-small-v1')
                - index_folder: Directory for index (default: './colbert_index')
                - index_name: Name for the index (default: 'belgian_banks')
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS is required for ColBERTRetriever. "
                "Install it with: pip install faiss-cpu"
            )

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers is required for ColBERTRetriever. "
                "Install it with: pip install transformers"
            )

        self.config = config or {}
        self.model_name = self.config.get(
            'colbert_model',
            'answerdotai/answerai-colbert-small-v1'
        )
        self.index_folder = self.config.get('index_folder', './colbert_index')
        self.index_name = self.config.get('index_name', 'belgian_banks')
        self.metadata_path = os.path.join(self.index_folder, 'metadata.json')
        self.index_path = os.path.join(self.index_folder, f'{self.index_name}.faiss')
        self.embeddings_path = os.path.join(self.index_folder, f'{self.index_name}_embeddings.npz')

        # Device selection
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Initialize model and tokenizer
        print(f"Loading ColBERT model: {self.model_name}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

        # Track chunk data
        self._chunk_texts: Dict[str, str] = {}
        self._chunk_metadatas: Dict[str, Dict[str, Any]] = {}
        self._chunk_ids: List[str] = []
        self._doc_embeddings: Dict[str, np.ndarray] = {}

        # FAISS index
        self._index = None

        # Try to load existing index
        self._load_index_if_exists()

    def _encode_batch(
        self,
        texts: List[str],
        is_query: bool = False,
        max_length: int = 512
    ) -> List[np.ndarray]:
        """
        Encode texts using ColBERT model.

        Returns list of token embeddings (each shape: [num_tokens, embedding_dim])
        """
        embeddings_list = []

        # Process in batches
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            inputs = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            # Get token embeddings (last hidden state)
            token_embeddings = outputs.last_hidden_state

            # Normalize embeddings (ColBERT style)
            token_embeddings = torch.nn.functional.normalize(token_embeddings, p=2, dim=-1)

            # Process each item in batch
            for j, emb in enumerate(token_embeddings):
                # Get attention mask to identify real tokens
                mask = inputs['attention_mask'][j]
                # Keep only non-padded tokens
                real_emb = emb[mask.bool()].cpu().numpy()
                embeddings_list.append(real_emb)

        return embeddings_list

    def _compute_maxsim_score(
        self,
        query_emb: np.ndarray,
        doc_emb: np.ndarray
    ) -> float:
        """
        Compute MaxSim score between query and document embeddings.

        For each query token, find max similarity with any doc token,
        then sum across query tokens.
        """
        # query_emb: [num_query_tokens, dim]
        # doc_emb: [num_doc_tokens, dim]

        # Compute similarity matrix
        sim_matrix = np.dot(query_emb, doc_emb.T)  # [num_query, num_doc]

        # MaxSim: for each query token, take max over doc tokens
        max_sims = sim_matrix.max(axis=1)

        # Sum (or average) across query tokens
        return float(max_sims.sum())

    def _load_index_if_exists(self) -> None:
        """Load existing FAISS index and metadata if available."""
        if (os.path.exists(self.index_path) and
            os.path.exists(self.metadata_path) and
            os.path.exists(self.embeddings_path)):
            print(f"Loading existing index from: {self.index_folder}")
            try:
                # Load FAISS index
                self._index = faiss.read_index(self.index_path)

                # Load metadata
                self._load_metadata()

                # Load embeddings
                data = np.load(self.embeddings_path, allow_pickle=True)
                self._doc_embeddings = data['embeddings'].item()

                print(f"  Loaded {len(self._chunk_texts)} chunks")
            except Exception as e:
                print(f"  Warning: Could not load index: {e}")
                self._index = None

    def _save_metadata(self) -> None:
        """Save chunk texts and metadata to JSON sidecar file."""
        os.makedirs(self.index_folder, exist_ok=True)
        data = {
            'texts': self._chunk_texts,
            'metadatas': self._chunk_metadatas,
            'chunk_ids': self._chunk_ids,
            'model_name': self.model_name,
        }
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_metadata(self) -> None:
        """Load chunk texts and metadata from JSON sidecar file."""
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self._chunk_texts = data.get('texts', {})
        self._chunk_metadatas = data.get('metadatas', {})
        self._chunk_ids = data.get('chunk_ids', [])

    def index_documents(
        self,
        chunks: List[str],
        chunk_ids: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        """
        Index documents using FAISS.

        Args:
            chunks: List of text chunks to index
            chunk_ids: Unique IDs for each chunk
            metadatas: Metadata dicts for each chunk

        Note: This creates a new index, overwriting any existing one.
        """
        print(f"Encoding {len(chunks)} chunks with ColBERT...")

        # Encode all documents
        doc_embeddings_list = self._encode_batch(chunks, is_query=False)

        # Store embeddings per chunk
        self._doc_embeddings = {}
        for chunk_id, emb in zip(chunk_ids, doc_embeddings_list):
            self._doc_embeddings[chunk_id] = emb

        # Create mean-pooled embeddings for FAISS index (for initial retrieval)
        # Then re-rank with full MaxSim
        embedding_dim = doc_embeddings_list[0].shape[1]
        mean_embeddings = np.zeros((len(chunks), embedding_dim), dtype=np.float32)

        for i, emb in enumerate(doc_embeddings_list):
            mean_embeddings[i] = emb.mean(axis=0)

        # Normalize for cosine similarity
        faiss.normalize_L2(mean_embeddings)

        # Create FAISS index
        print(f"Building FAISS index at: {self.index_folder}/{self.index_name}")
        os.makedirs(self.index_folder, exist_ok=True)

        self._index = faiss.IndexFlatIP(embedding_dim)  # Inner product = cosine for normalized vectors
        self._index.add(mean_embeddings)

        # Save FAISS index
        faiss.write_index(self._index, self.index_path)

        # Store texts and metadata
        self._chunk_texts = dict(zip(chunk_ids, chunks))
        self._chunk_metadatas = dict(zip(chunk_ids, metadatas))
        self._chunk_ids = list(chunk_ids)
        self._save_metadata()

        # Save embeddings
        np.savez(self.embeddings_path, embeddings=self._doc_embeddings)

        print(f"\n[SUCCESS] Indexed {len(chunks)} chunks")

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        language_filter: Optional[str] = None
    ) -> List[RetrievedPassage]:
        """
        Retrieve relevant passages for a query.

        Args:
            query: Search query
            n_results: Number of results to return
            language_filter: Optional language code filter (post-retrieval filtering)

        Returns:
            List of RetrievedPassage objects with token embeddings for MaxSimE
        """
        if self._index is None:
            raise RuntimeError(
                "No index loaded. Run index_documents() first or ensure "
                f"index exists at {self.index_folder}"
            )

        # Encode query
        query_embeddings = self._encode_batch([query], is_query=True)
        query_emb = query_embeddings[0]

        # Get mean-pooled query for FAISS search
        query_mean = query_emb.mean(axis=0, keepdims=True).astype(np.float32)
        faiss.normalize_L2(query_mean)

        # Retrieve more candidates for re-ranking and filtering
        fetch_n = max(n_results * 5, 20)
        scores, indices = self._index.search(query_mean, fetch_n)

        # Get query tokens for display
        query_tokens = self._tokenizer.tokenize(query)

        # Re-rank with MaxSim and collect results
        candidates = []
        for idx, initial_score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self._chunk_ids):
                continue

            chunk_id = self._chunk_ids[idx]
            metadata = self._chunk_metadatas.get(chunk_id, {})

            # Check language filter
            if language_filter and metadata.get('language') != language_filter:
                continue

            doc_text = self._chunk_texts.get(chunk_id, '')
            if not doc_text:
                continue

            # Get stored document embeddings
            doc_emb = self._doc_embeddings.get(chunk_id)
            if doc_emb is None:
                continue

            # Compute full MaxSim score
            maxsim_score = self._compute_maxsim_score(query_emb, doc_emb)

            # Get document tokens for display
            doc_tokens = self._tokenizer.tokenize(doc_text)

            candidates.append({
                'chunk_id': chunk_id,
                'content': doc_text,
                'metadata': metadata,
                'score': maxsim_score,
                'query_emb': query_emb,
                'doc_emb': doc_emb,
                'query_tokens': query_tokens,
                'doc_tokens': doc_tokens,
            })

        # Sort by MaxSim score and take top n
        candidates.sort(key=lambda x: x['score'], reverse=True)
        candidates = candidates[:n_results]

        # Convert to RetrievedPassage objects
        passages = []
        for c in candidates:
            passages.append(RetrievedPassage(
                content=c['content'],
                metadata=c['metadata'],
                score=c['score'],
                query_embeddings=c['query_emb'],
                doc_embeddings=c['doc_emb'],
                query_token_strings=c['query_tokens'],
                doc_token_strings=c['doc_tokens'],
            ))

        return passages

    def collection_info(self) -> Dict[str, Any]:
        """Get information about the indexed collection."""
        return {
            'name': self.index_name,
            'count': len(self._chunk_texts),
            'path': self.index_folder,
            'type': 'colbert',
            'model': self.model_name,
            'index_exists': self._index is not None,
        }
