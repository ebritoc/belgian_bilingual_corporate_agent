"""Simple PDF parsing and chunking utility."""
import re
from typing import List, Dict
from pathlib import Path
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: Path) -> List[Dict[str, any]]:
    """
    Extract text from PDF and return pages with basic metadata.

    Args:
        pdf_path: Path to PDF file

    Returns:
        List of dicts with 'text' and 'page_num' keys
    """
    reader = PdfReader(pdf_path)
    pages = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            # Basic cleaning: normalize whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 50:  # Skip nearly empty pages
                pages.append({
                    'text': text,
                    'page_num': page_num + 1
                })

    return pages


def simple_chunk(text: str, max_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Simple text chunking with overlap.
    Tries to split on sentence boundaries when possible.

    Args:
        text: Text to chunk
        max_size: Maximum chunk size in characters
        overlap: Overlap between chunks in characters

    Returns:
        List of text chunks
    """
    if len(text) <= max_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + max_size

        # If we're not at the end, try to split on sentence boundary
        if end < len(text):
            # Look for sentence end in last 200 chars of chunk
            search_start = max(start, end - 200)
            match = None
            for pattern in [r'[.!?]\s+', r'\n\n', r'\n']:
                matches = list(re.finditer(pattern, text[search_start:end]))
                if matches:
                    match = matches[-1]
                    break

            if match:
                end = search_start + match.end()

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start forward, accounting for overlap
        start = max(start + 1, end - overlap)

    return chunks


def chunk_pdf(pdf_path: Path, bank: str, language: str, max_chunk_size: int = 1000) -> List[Dict]:
    """
    Extract and chunk a PDF file.

    Args:
        pdf_path: Path to PDF file
        bank: Bank name (e.g., "BNP Paribas Fortis", "KBC Group")
        language: Language code ('fr', 'nl', 'en')
        max_chunk_size: Maximum chunk size in characters

    Returns:
        List of dicts with 'text' and 'metadata' keys
    """
    pages = extract_text_from_pdf(pdf_path)
    all_chunks = []

    for page_data in pages:
        text = page_data['text']
        page_num = page_data['page_num']

        # Chunk the page text
        chunks = simple_chunk(text, max_size=max_chunk_size)

        for chunk in chunks:
            all_chunks.append({
                'text': chunk,
                'metadata': {
                    'bank': bank,
                    'language': language,
                    'page': page_num,
                    'source': pdf_path.name
                }
            })

    return all_chunks
