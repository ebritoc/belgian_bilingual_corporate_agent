"""CLI for the Belgian Bilingual Corporate RAG System."""
import sys
import os
import argparse
from pathlib import Path

import yaml

# Ensure src is on sys.path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from rag_service import RAGService


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'config.yaml'
    else:
        config_path = Path(config_path)

    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def main():
    print("Belgian Bilingual Corporate RAG System")
    print("=" * 70)
    print("Commands:")
    print("  python scripts/rag_cli.py index")
    print("  python scripts/rag_cli.py query <your question>")
    print("=" * 70)

    parser = argparse.ArgumentParser(description="RAG CLI for Belgian banking reports")
    parser.add_argument('mode', choices=['index', 'query', 'info'], help='Mode: index, query, or info')
    parser.add_argument('question', nargs='*', help='Question to ask (for query mode)')

    # Configuration options
    parser.add_argument('--config', dest='config', default=None,
                        help='Path to config.yaml (default: config.yaml)')
    parser.add_argument('--retriever', dest='retriever', choices=['chroma', 'colbert'], default=None,
                        help='Retriever type (default: from config.yaml)')
    parser.add_argument('--ollama-url', dest='ollama_url', default=None,
                        help='Ollama API endpoint (default: http://localhost:11434)')
    parser.add_argument('--model', dest='model', default=None,
                        help='Ollama model name (default: qwen2.5:7b)')
    parser.add_argument('--timeout', dest='timeout', type=int, default=None,
                        help='Ollama request timeout in seconds (default: 180)')

    # Query options
    parser.add_argument('--no-llm', dest='no_llm', action='store_true',
                        help='Disable LLM generation; return sources only')
    parser.add_argument('--n-results', dest='n_results', type=int, default=None,
                        help='Number of passages to retrieve (default: from config or 5)')
    parser.add_argument('--bilingual-check', dest='bilingual_check', action='store_true',
                        help='Enable bilingual consistency checking')

    args = parser.parse_args()

    # Load config from YAML
    yaml_config = load_config(args.config)

    # Build configuration, merging YAML and CLI args
    config = {}

    # Retriever settings
    retriever_config = yaml_config.get('retriever', {})
    config['retriever_type'] = args.retriever or retriever_config.get('type', 'chroma')
    config.update(retriever_config)

    # LLM settings
    llm_config = yaml_config.get('llm', {})
    config['ollama_url'] = args.ollama_url or llm_config.get('ollama_url', 'http://localhost:11434')
    config['model_name'] = args.model or llm_config.get('model_name', 'qwen2.5:7b')
    config['ollama_timeout'] = args.timeout or llm_config.get('timeout', 180)

    # Retrieval settings
    retrieval_config = yaml_config.get('retrieval', {})
    n_results = args.n_results or retrieval_config.get('n_results', 5)

    # Data settings
    data_config = yaml_config.get('data', {})
    pdf_dir = data_config.get('pdf_directory', 'data/fetched_reports')

    print(f"Using retriever: {config['retriever_type']}")

    # Initialize RAG service
    rag = RAGService(config)

    # Execute command
    if args.mode == 'index':
        print("\n[INDEXING] Processing PDFs from:", pdf_dir)
        try:
            rag.index_documents(pdf_dir)
            print("\n[SUCCESS] Indexing complete!\n")
        except Exception as e:
            print(f"\n[ERROR] Error during indexing: {e}\n")
            sys.exit(1)

    elif args.mode == 'info':
        print("\n[INFO] Collection information:")
        try:
            info = rag.collection_info()
            print(f"  Collection name: {info['name']}")
            print(f"  Total chunks: {info['count']}")
            print(f"  Storage path: {info['path']}")
        except Exception as e:
            print(f"\n[ERROR] Error getting collection info: {e}\n")
            sys.exit(1)

    elif args.mode == 'query':
        if not args.question:
            print("\n[ERROR] Please provide a question to ask.")
            print("Example: python scripts/rag_cli.py query \"What is KBC's capital ratio?\"")
            sys.exit(1)

        question = ' '.join(args.question)

        try:
            result = rag.answer_question(
                question,
                n_results=n_results,
                use_llm=(not args.no_llm),
                enable_bilingual=args.bilingual_check
            )

            # Display results
            print("\n" + "="*70)
            print("ANSWER:")
            print("="*70)
            print(result['answer'])
            print("\n" + "="*70)
            print(f"Retrieved {result['num_sources']} source passages")

            # Show bilingual information if available
            if 'language' in result:
                print(f"Detected language: {result['language'].upper()}")

            if 'consistency' in result:
                c = result['consistency']
                print(f"Consistency: {c['status']} (confidence: {c['confidence']:.2f}) - {c['notes']}")

            # Show source details
            if result['sources']:
                print("\nSources:")
                for i, src in enumerate(result['sources'], 1):
                    print(f"  {i}. {src['bank']} (Page {src['page']}, {src['language'].upper()})")

            print("="*70)

        except Exception as e:
            print(f"\n[ERROR] Error during query: {e}\n")
            sys.exit(1)


if __name__ == "__main__":
    main()
