"""
Multilingual Banking RAG System using Free/Open-Source LLMs
Uses: Ollama (local LLM), ChromaDB, PyPDF2
"""

import chromadb
from pypdf import PdfReader
from pathlib import Path
import re
import requests
import json

class FreeBankingRAG:
    def __init__(self, ollama_url="http://localhost:11434"):
        """
        Initialize RAG system with Ollama (free local LLM)
        Install Ollama from: https://ollama.ai
        Then run: ollama pull qwen2.5:7b
        """
        self.ollama_url = ollama_url
        self.model_name = "qwen2.5:7b"  # Qwen 2.5 7B model
        # Initialize ChromaDB (free, local vector database)
        self.chroma_client = chromadb.PersistentClient(path="./banking_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name="banking_reports",
            metadata={"hnsw:space": "cosine"}
        )


    def extract_pdf_content(self, pdf_path, bank, language):
        """Extract and chunk PDF content with metadata"""
        reader = PdfReader(pdf_path)
        chunks = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text or not text.strip():
                continue

            # Clean header/footer noisy lines before chunking
            clean_text = self._clean_text(text)

            # Split into semantic chunks
            page_chunks = self._intelligent_chunk(clean_text, max_size=500)

            for i, chunk in enumerate(page_chunks):
                if len(chunk.strip()) > 50:  # Skip very short chunks
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

    def _clean_text(self, text):
        """Remove typical PDF noise: short all-caps headers, page numbers, repeated titles."""
        lines = text.splitlines()
        cleaned_lines = []
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            # Drop lines that are only page numbers
            if re.match(r'^\s*\d+\s*$', s):
                continue
            # Drop short all-caps lines that are likely headers/titles
            letters = re.sub(r'[^A-Za-z]', '', s)
            if letters:
                up_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            else:
                up_ratio = 0.0
            if len(s) < 100 and up_ratio > 0.65:
                # common header words that may be useful could be kept, we drop most
                continue
            # Drop common repeated words that indicate TOC or header
            if re.match(r'^(RAPPORT|ANNUEL|TABLE|CONTENU|CONTENTS|PAGE|INDEX)\b', s.upper()):
                continue
            cleaned_lines.append(s)
        return " ".join(cleaned_lines)

    def _intelligent_chunk(self, text, max_size=800):
        """Split text into semantic chunks"""
        # Clean up text
        text = re.sub(r'\s+', ' ', text).strip()

        # Break into sentence-like units
        sentences = re.split(r'(?<=[.!?])\s+|\n\n+', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If the sentence itself is very long, split it into subparts
            if len(sentence) > max_size * 1.5:
                # naive split on commas to avoid overly long single chunks
                parts = [p.strip() for p in re.split(r',\s+', sentence) if p.strip()]
                for part in parts:
                    if len(current_chunk) + len(part) < max_size:
                        current_chunk += part + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = part + " "
                continue

            if len(current_chunk) + len(sentence) < max_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        # Filter out chunks that are dominated by very short words or non-informative content
        filtered = []
        for c in chunks:
            alpha_chars = len(re.sub(r'[^A-Za-zÀ-ÖØ-öø-ÿ]', '', c))
            if alpha_chars < 30:  # skip chunks with too little alpha content
                continue
            filtered.append(c)
        return filtered


    def index_documents(self, pdf_directory):
        """Process all PDFs and add to vector DB"""
        pdf_configs = [
            # Filenames must match those in data/fetched_reports
            ('BNP_Paribas_Fortis_FR.pdf', 'BNP Paribas Fortis', 'fr'),
            ('BNP_Paribas_Fortis_NL.pdf', 'BNP Paribas Fortis', 'nl'),
            ('BNP_Paribas_Fortis_EN.pdf', 'BNP Paribas Fortis', 'en'),
            ('KBC_Group_FR.pdf', 'KBC Group', 'fr'),
            ('KBC_Group_NL.pdf', 'KBC Group', 'nl'),
            ('KBC_Group_EN.pdf', 'KBC Group', 'en'),
        ]
        all_chunks = []
        for filename, bank, lang in pdf_configs:
            pdf_path = Path(pdf_directory) / filename
            if pdf_path.exists():
                print(f"Processing {filename}...")
                chunks = self.extract_pdf_content(pdf_path, bank, lang)
                all_chunks.extend(chunks)
                print(f"  -> Extracted {len(chunks)} chunks")
            else:
                print(f"Warning: {filename} not found, skipping...")
        if not all_chunks:
            print("No documents to index!")
            return

        # Batch add to ChromaDB to avoid internal max-batch errors
        total = len(all_chunks)
        batch_size = 4000  # safe default; lower if you still hit limits
        print(f"\nIndexing {total} total chunks in batches of {batch_size}...")

        start_idx = 0
        try:
            while start_idx < total:
                end_idx = min(start_idx + batch_size, total)
                batch = all_chunks[start_idx:end_idx]
                docs = [c['text'] for c in batch]
                metas = [c['metadata'] for c in batch]
                ids = [f"chunk_{i}" for i in range(start_idx, end_idx)]
                self.collection.add(
                    documents=docs,
                    metadatas=metas,
                    ids=ids
                )
                print(f"  ✓ Indexed chunks {start_idx}..{end_idx-1}")
                start_idx = end_idx
        except Exception as e:
            print(f"Error while indexing to ChromaDB: {e}")
            return

        print(f"✓ Successfully indexed {total} chunks!")

    def retrieve_context(self, query, n_results=5, language_filter=None, bank_filter=None):
        """Retrieve relevant chunks from vector DB and print diagnostics (includes distances)"""
        where_filter = {}
        if language_filter:
            where_filter["language"] = language_filter
        if bank_filter:
            where_filter["bank"] = bank_filter

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter if where_filter else None,
            include=['documents', 'metadatas', 'distances']
        )

        # Print diagnostics so you can inspect relevance & distances
        docs = results.get('documents', [[]])[0]
        metas = results.get('metadatas', [[]])[0]
        dists = results.get('distances', [[]])[0] if 'distances' in results else []
        print(f"[chroma] query returned {len(docs)} hits")
        if dists:
            print(f"[chroma] distances: {[round(float(d),4) for d in dists]}")
        for i, (meta, doc) in enumerate(zip(metas, docs)):
            snippet = (doc[:250].replace("\n", " ") + '...') if doc else ''
            print(f"  hit {i+1}: bank={meta.get('bank')} lang={meta.get('language')} page={meta.get('page')} src={meta.get('source')} snippet={snippet}")

        return results

    def query_ollama(self, prompt, system_prompt=None):
        """Query local Ollama LLM"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            if system_prompt:
                payload["system"] = system_prompt
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=120
            )
            if response.status_code == 200:
                return response.json()['response']
            else:
                return f"Error: Ollama returned status {response.status_code}"
        except requests.exceptions.ConnectionError:
            return ("Error: Cannot connect to Ollama. "
                   "Make sure Ollama is running (ollama serve) and "
                   f"model '{self.model_name}' is installed (ollama pull {self.model_name})")
        except Exception as e:
            return f"Error querying Ollama: {str(e)}"

    def _detect_language(self, text):
        """Simple language detection"""
        text_lower = text.lower()
        fr_indicators = len(re.findall(r'\b(le|la|les|un|une|des|et|avec|pour|que|qui|dans|sur|est|sont)\b', text_lower))
        nl_indicators = len(re.findall(r'\b(de|het|een|van|en|met|voor|dat|die|zijn|is|op|in)\b', text_lower))
        en_indicators = len(re.findall(r'\b(the|is|are|and|with|for|that|this|from|have)\b', text_lower))
        max_score = max(fr_indicators, nl_indicators, en_indicators)
        if max_score == 0:
            return 'en'  # default
        elif fr_indicators == max_score:
            return 'fr'
        elif nl_indicators == max_score:
            return 'nl'
        else:
            return 'en'

    def answer_question(self, user_question, n_results=6):
        print(f"\n{'='*60}")
        print(f"Question: {user_question}")
        print(f"{'='*60}\n")
        detected_lang = self._detect_language(user_question)
        print(f"Detected language: {detected_lang.upper()}")
        print("Searching documents...")
        retrieval_results = self.retrieve_context(
            user_question, 
            n_results=n_results
        )
        # Print retrieved chunks and metadata for debugging / transparency
        docs = retrieval_results.get('documents', [[]])[0]
        metas = retrieval_results.get('metadatas', [[]])[0]
        if not docs:
            return "No relevant information found in the documents."

        print(f"\nRetrieved {len(docs)} documents/passages:\n")
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            bank = meta.get('bank', 'unknown')
            page = meta.get('page', '?')
            lang = meta.get('language', '').upper()
            source = meta.get('source', '')
            snippet = (doc[:300].strip() + '...') if len(doc) > 300 else doc.strip()
            print(f"[{i+1}] {bank} | Page: {page} | Lang: {lang} | Source: {source}")
            print(f"    {snippet}\n")
        context_blocks = []
        for i, (doc, metadata) in enumerate(zip(docs, metas)):
            context_blocks.append(
                f"[Source {i+1}: {metadata.get('bank')}, Page {metadata.get('page')}, Language: {metadata.get('language','').upper()}]\n{doc}"
            )
        context = "\n\n---\n\n".join(context_blocks)
        print(f"Found {len(context_blocks)} relevant passages\n")
        system_prompt = """You are a financial analyst assistant. Answer questions based ONLY on the provided context from banking reports. \nIf the information is not in the context, say so clearly. \nAlways cite your sources (bank name and page number).\nKeep your answer concise and factual."""
        user_prompt = f"""Context from banking reports:\n\n{context}\n\n---\n\nQuestion: {user_question}\n\nAnswer the question based on the context above. Cite your sources."""
        print("Generating answer with local LLM...")
        answer = self.query_ollama(user_prompt, system_prompt)
        return {
            'answer': answer,
            'sources': retrieval_results['metadatas'][0],
            'language': detected_lang,
            'num_sources': len(context_blocks)
        }
