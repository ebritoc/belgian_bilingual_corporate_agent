from typing import Optional, List, Dict, Any
from pathlib import Path

from .pdf_ingestion_agent import PDFIngestion
from .retrieval_agent import RetrievalAgent
from .generation_agent import GenerationAgent


class RAGOrchestrator:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.retrieval = RetrievalAgent(
            chroma_path=cfg.get('chroma_path', './banking_db_v2'),
            embedding_model_name=cfg.get('embedding_model', 'paraphrase-multilingual-mpnet-base-v2'),
        )
        self.ingestion = PDFIngestion()
        self.generator = GenerationAgent(
            ollama_url=cfg.get('ollama_url', 'http://localhost:11434'),
            model_name=cfg.get('model_name', 'qwen2.5:7b'),
            timeout=int(cfg.get('ollama_timeout', 180)),
        )
        self.enable_bilingual_check = bool(cfg.get('enable_bilingual_check', False))

    # ---- Language utils ----
    def _detect_language(self, text: str) -> str:
        import re
        t = text.lower()
        fr = len(re.findall(r'\b(le|la|les|des|un|une|du|avec|pour|que|qui|dans|sur|est|sont|ont|leur|cette)\b', t))
        nl = len(re.findall(r'\b(de|het|een|van|en|met|voor|dat|die|zijn|is|was|op|in|bij|heeft|werd)\b', t))
        en = len(re.findall(r'\b(the|is|are|and|with|for|that|this|from|have|was|were|has|been)\b', t))
        m = max(fr, nl, en)
        if m == 0:
            return 'en'
        return 'fr' if fr == m else ('nl' if nl == m else 'en')

    def detect_and_expand_query(self, text: str) -> Dict[str, Any]:
        detected = self._detect_language(text)
        def map_terms(q: str, src: str, tgt: str) -> str:
            ql = q.lower()
            mappings = {
                ('fr','nl'): [('résultat annuel','jaarresultaat'), ('résultat net','nettowinst')],
                ('nl','fr'): [('jaarresultaat','résultat annuel'), ('nettowinst','résultat net')],
                ('en','fr'): [('annual result','résultat annuel'), ('net profit','résultat net')],
                ('en','nl'): [('annual result','jaarresultaat'), ('net profit','nettowinst')],
            }
            out = q
            for a,b in mappings.get((src,tgt), []):
                if a in ql:
                    out = out.lower().replace(a, b)
            return out
        if detected == 'fr':
            variants = [{'lang':'fr','text':text}, {'lang':'nl','text':map_terms(text,'fr','nl')}, {'lang':'en','text':map_terms(text,'fr','en')}]
        elif detected == 'nl':
            variants = [{'lang':'nl','text':text}, {'lang':'fr','text':map_terms(text,'nl','fr')}, {'lang':'en','text':map_terms(text,'nl','en')}]
        else:
            variants = [{'lang':'en','text':text}, {'lang':'fr','text':map_terms(text,'en','fr')}, {'lang':'nl','text':map_terms(text,'en','nl')}]
        return {'detected': detected, 'variants': variants}

    def compare_numeric_across_languages(self, buckets: Dict[str,Any]) -> Dict[str, Any]:
        import re
        def nums(t: str):
            return set(re.findall(r"\b\d+[\d.,]*\b", t))
        per_lang = {}
        for lang, data in buckets.items():
            docs = data.get('documents', [])
            flat: List[str] = []
            for d in docs:
                if isinstance(d, list):
                    flat.extend(d)
                else:
                    flat.append(d)
            bag = set()
            for d in flat[:3]:
                bag |= nums(d)
            per_lang[lang] = bag
        if len(per_lang) < 2:
            return {'status':'ok','confidence':0.5,'notes':'Insufficient language coverage'}
        sets = list(per_lang.values())
        common = set.intersection(*sets) if sets else set()
        if common:
            return {'status':'ok','confidence':0.8,'notes':'Numeric overlap across languages'}
        if any(len(s)>0 for s in sets):
            return {'status':'discrepancy','confidence':0.7,'notes':'Different numeric figures across languages'}
        return {'status':'ok','confidence':0.6,'notes':'No numeric figures detected'}

    # ---- Orchestration ----
    def index_documents(self, pdf_directory: str):
        pdf_directory = Path(pdf_directory)
        pdfs = [
            ('BNP_Paribas_Fortis_FR.pdf', 'BNP Paribas Fortis', 'fr'),
            ('BNP_Paribas_Fortis_NL.pdf', 'BNP Paribas Fortis', 'nl'),
            ('BNP_Paribas_Fortis_EN.pdf', 'BNP Paribas Fortis', 'en'),
            ('KBC_Group_FR.pdf', 'KBC Group', 'fr'),
            ('KBC_Group_NL.pdf', 'KBC Group', 'nl'),
            ('KBC_Group_EN.pdf', 'KBC Group', 'en'),
        ]
        all_chunks: List[Dict[str,Any]] = []
        for filename, bank, lang in pdfs:
            p = pdf_directory / filename
            if p.exists():
                print(f"Processing {filename}...")
                all_chunks.extend(self.ingestion.extract_chunks(p, bank, lang))
            else:
                print(f"Warning: {filename} not found, skipping...")
        if not all_chunks:
            print("No documents to index!")
            return
        print(f"\nIndexing {len(all_chunks)} chunks with multilingual embeddings...")
        self.retrieval.index(all_chunks)
        print(f"✓ Successfully indexed {len(all_chunks)} chunks!")

    def answer_question(self, user_question: str, n_results: int = 8, use_llm: bool = True, bilingual_check: Optional[bool] = None) -> Dict[str, Any]:
        print(f"\n{'='*70}")
        print(f"Question: {user_question}")
        print(f"{'='*70}\n")

        detected_lang = self._detect_language(user_question)
        print(f"Detected language: {detected_lang.upper()}\n")

        if bilingual_check is None:
            bilingual_check = self.enable_bilingual_check

        print("Performing hybrid search (vector + keyword)...")
        if bilingual_check:
            exp = self.detect_and_expand_query(user_question)
            multi = self.retrieval.multi_retrieve(exp['variants'], n_per_lang=max(2, n_results//3))
            results = multi['merged']
            consistency = self.compare_numeric_across_languages(multi['buckets'])
        else:
            results = self.retrieval.hybrid_search(user_question, n_results=n_results)
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
        system_prompt, user_prompt = self.generator.build_prompts(context, user_question, consistency_note=warn)

        if use_llm:
            print("Generating answer with local LLM...")
            answer = self.generator.generate(user_prompt, system_prompt)
        else:
            print("LLM disabled; returning sources only.")
            answer = "LLM disabled. Review sources below."

        return {'answer': answer, 'sources': metas, 'language': detected_lang, 'num_sources': len(docs), 'consistency': consistency}
