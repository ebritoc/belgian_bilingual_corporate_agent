from typing import List, Dict
from pathlib import Path


class PDFIngestion:
    def advanced_clean(self, text: str) -> str:
        import re
        lines = text.splitlines()
        cleaned = []
        skip_patterns = [
            r'^\s*\d+\s*$',
            r'^(TABLE OF CONTENTS|CONTENU|INHOUD|INDEX)',
            r'^\s*\.{3,}', r'^\s*_{3,}', r'^\s*-{3,}',
            r'^\s*(RAPPORT|ANNUEL|ANNUAL|REPORT|JAARVERSLAG)\s*$'
        ]
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

    def semantic_chunk(self, text: str, max_size=1000, min_size=200) -> List[str]:
        import re
        text = re.sub(r'\s+', ' ', text).strip()
        segments = re.split(r'(?<=[.!?])\s+(?=[A-ZÀ-Ö])|(?:\n\s*){2,}', text)
        chunks: List[str] = []
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

    def is_quality_chunk(self, chunk: str) -> bool:
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

    def extract_chunks(self, pdf_path: Path, bank: str, language: str) -> List[Dict]:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        chunks = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text or not text.strip():
                continue
            clean_text = self.advanced_clean(text)
            if not clean_text or len(clean_text) < 100:
                continue
            page_chunks = self.semantic_chunk(clean_text, max_size=1000, min_size=200)
            for chunk in page_chunks:
                if self.is_quality_chunk(chunk):
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
