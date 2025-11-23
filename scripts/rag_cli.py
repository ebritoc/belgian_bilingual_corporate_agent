import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from rag.free_banking_rag import FreeBankingRAG

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
    args = parser.parse_args()

    rag = FreeBankingRAG()
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
        result = rag.answer_question(question)
        print("\n" + "="*60)
        print("ANSWER:")
        print("="*60)
        print(result['answer'])
        print("\n" + "="*60)
        print(f"Sources: {result['num_sources']} passages from reports")
        print("="*60)

if __name__ == "__main__":
    main()
