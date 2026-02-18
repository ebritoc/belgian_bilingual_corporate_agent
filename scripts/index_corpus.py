#!/usr/bin/env python3
"""
One-time corpus indexing script.

Reads all PDFs from data/fetched_reports/, chunks them, encodes with ColBERT,
and builds a FAISS index.

Usage:
    python scripts/index_corpus.py
    python scripts/index_corpus.py --model jinaai/jina-colbert-v2  # on GPU machine
"""
import argparse
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import yaml

from utils.pdf_parser import chunk_pdf


def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from YAML file."""
    config_file = Path(__file__).parent.parent / config_path
    if config_file.exists():
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(
        description='Index PDF corpus for retrieval',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python scripts/index_corpus.py
  python scripts/index_corpus.py --model jinaai/jina-colbert-v2
  python scripts/index_corpus.py --pdf-dir ./my_pdfs
        '''
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model name/path (overrides config.yaml)'
    )
    parser.add_argument(
        '--pdf-dir',
        type=str,
        default=None,
        help='PDF directory (default: from config.yaml or "./data/fetched_reports")'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=None,
        help='Chunk size in characters (default: from config.yaml or 1000)'
    )
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=None,
        help='Chunk overlap in characters (default: from config.yaml or 100)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    pdf_dir = args.pdf_dir or config.get('data', {}).get('pdf_directory', './data/fetched_reports')
    chunk_size = args.chunk_size or config.get('chunking', {}).get('chunk_size', 1000)
    chunk_overlap = args.chunk_overlap or config.get('chunking', {}).get('chunk_overlap', 100)

    # Build retriever config
    retriever_config = {
        'retriever_type': 'colbert',
        **config.get('retriever', {}),
    }

    if args.model:
        retriever_config['colbert_model'] = args.model

    model = retriever_config.get('colbert_model', 'answerdotai/answerai-colbert-small-v1')

    print("=" * 70)
    print("Corpus Indexing")
    print("=" * 70)
    print(f"PDF directory: {pdf_dir}")
    print(f"Chunk size: {chunk_size}, overlap: {chunk_overlap}")
    print(f"ColBERT model: {model}")
    print("=" * 70)
    print()

    # Find PDFs
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        print(f"Error: PDF directory not found: {pdf_dir}")
        sys.exit(1)

    pdf_files = list(pdf_path.glob("*.pdf"))
    if not pdf_files:
        print(f"Error: No PDF files found in: {pdf_dir}")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF files")

    # Extract and chunk all PDFs
    start_time = time.time()
    all_chunks = []
    all_chunk_ids = []
    all_metadatas = []

    for pdf_file in pdf_files:
        # Parse filename: BankName_Language.pdf
        name = pdf_file.stem
        parts = name.rsplit('_', 1)

        if len(parts) == 2:
            bank = parts[0].replace('_', ' ')
            language = parts[1].lower()
        else:
            bank = name.replace('_', ' ')
            language = 'unknown'

        print(f"\nProcessing: {pdf_file.name}")
        print(f"  Bank: {bank}, Language: {language}")

        chunks = chunk_pdf(
            pdf_file,
            bank=bank,
            language=language,
            chunk_size=chunk_size,
            overlap=chunk_overlap
        )
        print(f"  Extracted {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            # Generate unique chunk ID
            page = chunk['metadata'].get('page', 0)
            chunk_id = f"{bank.replace(' ', '_')}_{language}_p{page}_c{i}"

            all_chunks.append(chunk['text'])
            all_chunk_ids.append(chunk_id)
            all_metadatas.append(chunk['metadata'])

    extraction_time = time.time() - start_time
    print(f"\nExtraction complete: {len(all_chunks)} chunks in {extraction_time:.1f}s")

    # Create retriever and index
    print(f"\nCreating {retriever_type} index...")
    start_time = time.time()

    from retriever import create_retriever
    retriever = create_retriever(retriever_config)
    retriever.index_documents(all_chunks, all_chunk_ids, all_metadatas)

    indexing_time = time.time() - start_time
    print(f"\nIndexing complete in {indexing_time:.1f}s")

    # Show final stats
    info = retriever.collection_info()
    print("\n" + "=" * 70)
    print("Index Statistics")
    print("=" * 70)
    for key, value in info.items():
        print(f"  {key}: {value}")
    print("=" * 70)


if __name__ == '__main__':
    main()
