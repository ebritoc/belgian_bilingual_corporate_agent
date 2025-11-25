import sys
import os
# Ensure src is on sys.path for local imports (keeps previous behavior)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from agents.rag_agent import RAGAgent

def main():
    print("Multilingual Banking RAG System (CLI)")
    print("=" * 60)
    print("1. To index PDFs, run: python scripts/rag_cli.py index")
    print("2. To ask a question, run: python scripts/rag_cli.py query <your question>")
    print("=" * 60)

    import argparse
    parser = argparse.ArgumentParser(description="RAG CLI for banking reports")
    parser.add_argument('mode', choices=['index', 'query'], help='Mode: index or query')
    parser.add_argument('question', nargs='*', help='Question to ask (for query mode)')
    parser.add_argument('--ollama-url', dest='ollama_url', default=None, help='Override Ollama base URL (e.g., http://localhost:11434)')
    parser.add_argument('--model', dest='model', default=None, help='Override model name (e.g., qwen2.5:7b)')
    parser.add_argument('--timeout', dest='timeout', type=int, default=None, help='Ollama request timeout in seconds')
    parser.add_argument('--no-llm', dest='no_llm', action='store_true', help='Disable LLM generation; only retrieve sources')
    parser.add_argument('--n-results', dest='n_results', type=int, default=8, help='Number of passages to retrieve')
    args = parser.parse_args()

    # Instantiate RAGAgent with optional config from CLI flags
    config = {}
    if args.ollama_url:
        config['ollama_url'] = args.ollama_url
    if args.model:
        config['model_name'] = args.model
    if args.timeout:
        config['ollama_timeout'] = args.timeout
    rag = RAGAgent(config)
    pdf_dir = "data/fetched_reports"

    if args.mode == 'index':
        print("\n[INDEXING] Processing PDFs...")
        rag.index_documents(pdf_dir)
        print("\n[INDEXING] Complete!\n")
    elif args.mode == 'query':
        if not args.question:
            print("Please provide a question to ask.")
            return
        question = ' '.join(args.question)
        result = rag.answer_question(question, n_results=args.n_results, use_llm=(not args.no_llm))
        print("\n" + "="*60)
        print("ANSWER:")
        print("="*60)
        print(result['answer'])
        print("\n" + "="*60)
        print(f"Sources: {result['num_sources']} passages from reports")
        print("="*60)

if __name__ == "__main__":
    main()
