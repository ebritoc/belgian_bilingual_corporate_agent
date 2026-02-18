#!/usr/bin/env python3
"""
Evaluation pipeline for the Belgian Bilingual Corporate Agent.

Runs retrieval and consistency experiments on the benchmark dataset.

Usage:
    python scripts/run_evaluation.py --no-llm
    python scripts/run_evaluation.py --facts 10 --no-llm
    python scripts/run_evaluation.py --output-dir results_v2
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def _compute_ndcg(rank: Optional[int], k: int) -> float:
    """
    nDCG@k for a single binary-relevant document.

    With binary relevance and one relevant document, IDCG@k = 1.0 (best rank = 1),
    so nDCG@k = 1 / log2(rank + 1) if rank <= k, else 0.0.
    """
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


@dataclass
class FactResult:
    """Result for a single benchmark fact."""
    fact_id: int
    question_fr: str
    question_nl: str
    ground_truth: str
    category: str
    difficulty: str

    # Retrieval results
    retrieved_fr: bool
    retrieved_nl: bool
    rank_fr: Optional[int]       # 1-based rank of first hit, or None
    rank_nl: Optional[int]

    recall_at_3_fr: float
    recall_at_3_nl: float
    recall_at_5_fr: float
    recall_at_5_nl: float
    ndcg_at_3_fr: float
    ndcg_at_3_nl: float
    ndcg_at_5_fr: float
    ndcg_at_5_nl: float
    mrr_fr: float
    mrr_nl: float

    # Consistency results (ColBERT token alignment)
    consistency_score: Optional[float] = None
    alignment_overlap: Optional[float] = None
    flagged: Optional[bool] = None

    # LLM results (if enabled)
    answer_fr: Optional[str] = None
    answer_nl: Optional[str] = None
    answer_correct_fr: Optional[bool] = None
    answer_correct_nl: Optional[bool] = None


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


def load_benchmark(benchmark_path: str) -> dict:
    """Load the evaluation benchmark."""
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def check_ground_truth_in_results(
    ground_truth: str,
    documents: List[str],
    metadatas: List[dict],
    source_page: Optional[int] = None,
) -> tuple:
    """
    Check if ground truth is found in retrieved documents.

    Returns:
        (found: bool, rank: int or None)
        rank is 1-based position of first hit, or None if not found.
    """
    ground_truth_lower = ground_truth.lower().strip()

    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        if ground_truth_lower in doc.lower():
            if source_page and meta.get('page') != source_page:
                continue
            return True, i + 1

    return False, None


def evaluate_fact(
    fact: dict,
    rag_service,
    n_results: int = 5,
    use_llm: bool = False,
    check_consistency: bool = True,
) -> FactResult:
    """Evaluate a single benchmark fact."""
    fact_id = fact['id']
    question_fr = fact['question_fr']
    question_nl = fact['question_nl']
    ground_truth = fact['ground_truth']
    category = fact.get('category', 'unknown')
    difficulty = fact.get('difficulty', 'unknown')
    source_page_fr = fact.get('source_page_fr')
    source_page_nl = fact.get('source_page_nl')

    # Retrieve with FR query
    results_fr = rag_service.retrieve(question_fr, n_results=n_results)
    docs_fr = results_fr.get('documents', [])
    metas_fr = results_fr.get('metadatas', [])
    passages_fr = results_fr.get('passages', [])

    found_fr, rank_fr = check_ground_truth_in_results(
        ground_truth, docs_fr, metas_fr, source_page_fr
    )

    # Retrieve with NL query
    results_nl = rag_service.retrieve(question_nl, n_results=n_results)
    docs_nl = results_nl.get('documents', [])
    metas_nl = results_nl.get('metadatas', [])
    passages_nl = results_nl.get('passages', [])

    found_nl, rank_nl = check_ground_truth_in_results(
        ground_truth, docs_nl, metas_nl, source_page_nl
    )

    # Compute all retrieval metrics from rank
    recall_at_3_fr = 1.0 if (rank_fr is not None and rank_fr <= 3) else 0.0
    recall_at_3_nl = 1.0 if (rank_nl is not None and rank_nl <= 3) else 0.0
    recall_at_5_fr = 1.0 if (rank_fr is not None and rank_fr <= 5) else 0.0
    recall_at_5_nl = 1.0 if (rank_nl is not None and rank_nl <= 5) else 0.0
    ndcg_at_3_fr   = _compute_ndcg(rank_fr, 3)
    ndcg_at_3_nl   = _compute_ndcg(rank_nl, 3)
    ndcg_at_5_fr   = _compute_ndcg(rank_fr, 5)
    ndcg_at_5_nl   = _compute_ndcg(rank_nl, 5)
    mrr_fr         = (1.0 / rank_fr) if rank_fr is not None else 0.0
    mrr_nl         = (1.0 / rank_nl) if rank_nl is not None else 0.0

    # Check consistency if token embeddings available (ColBERT)
    consistency_score = None
    alignment_overlap = None
    flagged = None

    if check_consistency and passages_fr and passages_nl:
        has_embeddings = (
            any(getattr(p, 'query_embeddings', None) is not None for p in passages_fr) and
            any(getattr(p, 'query_embeddings', None) is not None for p in passages_nl)
        )

        if has_embeddings:
            try:
                from bilingual_validator import BilingualValidator
                validator = BilingualValidator()
                consistency_result = validator.check_token_alignment_consistency(
                    passages_fr, passages_nl
                )

                if consistency_result.get('results'):
                    cr = consistency_result['results']
                    consistency_score = sum(r.score for r in cr) / len(cr)
                    alignment_overlap = sum(r.alignment_overlap for r in cr) / len(cr)
                    flagged = any(r.flagged for r in cr)
            except Exception as e:
                print(f"  Warning: Consistency check failed: {e}")

    # LLM evaluation (if enabled)
    answer_fr = None
    answer_nl = None
    answer_correct_fr = None
    answer_correct_nl = None

    if use_llm:
        result_fr = rag_service.answer_question(
            question_fr, n_results=n_results, use_llm=True, enable_bilingual=False
        )
        answer_fr = result_fr.get('answer', '')
        answer_correct_fr = ground_truth.lower() in answer_fr.lower()

        result_nl = rag_service.answer_question(
            question_nl, n_results=n_results, use_llm=True, enable_bilingual=False
        )
        answer_nl = result_nl.get('answer', '')
        answer_correct_nl = ground_truth.lower() in answer_nl.lower()

    return FactResult(
        fact_id=fact_id,
        question_fr=question_fr,
        question_nl=question_nl,
        ground_truth=ground_truth,
        category=category,
        difficulty=difficulty,
        retrieved_fr=found_fr,
        retrieved_nl=found_nl,
        rank_fr=rank_fr,
        rank_nl=rank_nl,
        recall_at_3_fr=recall_at_3_fr,
        recall_at_3_nl=recall_at_3_nl,
        recall_at_5_fr=recall_at_5_fr,
        recall_at_5_nl=recall_at_5_nl,
        ndcg_at_3_fr=ndcg_at_3_fr,
        ndcg_at_3_nl=ndcg_at_3_nl,
        ndcg_at_5_fr=ndcg_at_5_fr,
        ndcg_at_5_nl=ndcg_at_5_nl,
        mrr_fr=mrr_fr,
        mrr_nl=mrr_nl,
        consistency_score=consistency_score,
        alignment_overlap=alignment_overlap,
        flagged=flagged,
        answer_fr=answer_fr,
        answer_nl=answer_nl,
        answer_correct_fr=answer_correct_fr,
        answer_correct_nl=answer_correct_nl,
    )


def compute_summary(results: List[FactResult]) -> dict:
    """Compute summary statistics from results."""
    n = len(results)
    if n == 0:
        return {}

    def avg(vals):
        return sum(vals) / n

    retrieval = {
        'recall_at_3_fr':  avg(r.recall_at_3_fr for r in results),
        'recall_at_3_nl':  avg(r.recall_at_3_nl for r in results),
        'recall_at_5_fr':  avg(r.recall_at_5_fr for r in results),
        'recall_at_5_nl':  avg(r.recall_at_5_nl for r in results),
        'ndcg_at_3_fr':    avg(r.ndcg_at_3_fr for r in results),
        'ndcg_at_3_nl':    avg(r.ndcg_at_3_nl for r in results),
        'ndcg_at_5_fr':    avg(r.ndcg_at_5_fr for r in results),
        'ndcg_at_5_nl':    avg(r.ndcg_at_5_nl for r in results),
        'mrr_fr':          avg(r.mrr_fr for r in results),
        'mrr_nl':          avg(r.mrr_nl for r in results),
    }

    # Consistency metrics (if available)
    cr = [r for r in results if r.consistency_score is not None]
    consistency = {
        'avg_score':    sum(r.consistency_score for r in cr) / len(cr),
        'avg_overlap':  sum(r.alignment_overlap for r in cr) / len(cr),
        'flagged_rate': sum(1 for r in cr if r.flagged) / len(cr),
    } if cr else None

    # LLM metrics (if available)
    lr = [r for r in results if r.answer_correct_fr is not None]
    llm = {
        'accuracy_fr': sum(1 for r in lr if r.answer_correct_fr) / len(lr),
        'accuracy_nl': sum(1 for r in lr if r.answer_correct_nl) / len(lr),
    } if lr else None

    # By difficulty
    by_difficulty = {}
    for difficulty in ['easy', 'medium', 'hard', 'adversarial']:
        dr = [r for r in results if r.difficulty == difficulty]
        if dr:
            dn = len(dr)
            by_difficulty[difficulty] = {
                'count':       dn,
                'recall_3_fr': sum(r.recall_at_3_fr for r in dr) / dn,
                'recall_3_nl': sum(r.recall_at_3_nl for r in dr) / dn,
                'recall_5_fr': sum(r.recall_at_5_fr for r in dr) / dn,
                'recall_5_nl': sum(r.recall_at_5_nl for r in dr) / dn,
                'ndcg_5_fr':   sum(r.ndcg_at_5_fr for r in dr) / dn,
                'ndcg_5_nl':   sum(r.ndcg_at_5_nl for r in dr) / dn,
            }

    return {
        'total_facts': n,
        'retrieval': retrieval,
        'consistency': consistency,
        'llm': llm,
        'by_difficulty': by_difficulty,
    }


def write_summary_markdown(summary: dict, output_path: Path) -> None:
    """Write summary as markdown file."""
    r = summary['retrieval']

    lines = [
        "# Evaluation Summary",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Retrieval Performance",
        "",
        "| Metric     |   FR    |   NL    |",
        "|------------|---------|---------|",
        f"| Recall@3   | {r['recall_at_3_fr']:.3f}  | {r['recall_at_3_nl']:.3f}  |",
        f"| Recall@5   | {r['recall_at_5_fr']:.3f}  | {r['recall_at_5_nl']:.3f}  |",
        f"| nDCG@3     | {r['ndcg_at_3_fr']:.3f}  | {r['ndcg_at_3_nl']:.3f}  |",
        f"| nDCG@5     | {r['ndcg_at_5_fr']:.3f}  | {r['ndcg_at_5_nl']:.3f}  |",
        f"| MRR        | {r['mrr_fr']:.3f}  | {r['mrr_nl']:.3f}  |",
        "",
    ]

    if summary.get('consistency'):
        c = summary['consistency']
        lines.extend([
            "## Consistency Metrics",
            "",
            f"- Average score: {c['avg_score']:.3f}",
            f"- Average overlap: {c['avg_overlap']:.3f}",
            f"- Flagged rate: {c['flagged_rate']:.1%}",
            "",
        ])

    if summary.get('llm'):
        lm = summary['llm']
        lines.extend([
            "## LLM Accuracy",
            "",
            f"- FR: {lm['accuracy_fr']:.1%}",
            f"- NL: {lm['accuracy_nl']:.1%}",
            "",
        ])

    if summary.get('by_difficulty'):
        lines.extend([
            "## By Difficulty",
            "",
            "| Difficulty  | Count | R@3 FR | R@3 NL | R@5 FR | R@5 NL | nDCG@5 FR | nDCG@5 NL |",
            "|-------------|-------|--------|--------|--------|--------|-----------|-----------|",
        ])
        for diff, d in summary['by_difficulty'].items():
            lines.append(
                f"| {diff:<11} | {d['count']:>5} "
                f"| {d['recall_3_fr']:.3f}  | {d['recall_3_nl']:.3f}  "
                f"| {d['recall_5_fr']:.3f}  | {d['recall_5_nl']:.3f}  "
                f"| {d['ndcg_5_fr']:.3f}     | {d['ndcg_5_nl']:.3f}     |"
            )
        lines.append("")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(
        description='Run evaluation on benchmark dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--no-llm', action='store_true',
                        help='Skip LLM evaluation (faster)')
    parser.add_argument('--benchmark', type=str, default=None,
                        help='Path to benchmark JSON')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--facts', type=int, default=None,
                        help='Limit to first N facts (for testing)')
    parser.add_argument('--n-results', type=int, default=5,
                        help='Number of passages to retrieve (default: 5)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.yaml')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup paths
    project_root = Path(__file__).parent.parent
    benchmark_path = args.benchmark or config.get('data', {}).get(
        'benchmark_file', 'tests/eval_benchmark.json'
    )
    benchmark_path = project_root / benchmark_path
    output_dir = project_root / args.output_dir
    output_dir.mkdir(exist_ok=True)

    # Load benchmark
    print("=" * 70)
    print("Belgian Bilingual Corporate Agent - Evaluation")
    print("=" * 70)

    benchmark = load_benchmark(benchmark_path)
    facts = benchmark.get('facts', [])

    if args.facts:
        facts = facts[:args.facts]

    retriever_type = config.get('retriever', {}).get('type', 'colbert')
    print(f"Benchmark: {benchmark_path}")
    print(f"Facts to evaluate: {len(facts)}")
    print(f"Retriever: {retriever_type}")
    print(f"LLM evaluation: {'No' if args.no_llm else 'Yes'}")
    print("=" * 70)

    # Build retriever config
    retriever_config = config.get('retriever', {}).copy()
    retriever_config['retriever_type'] = retriever_type
    retriever_config.update(config.get('llm', {}))

    # Initialize RAG service
    from rag_service import RAGService
    print("\nInitializing RAG service...")
    rag = RAGService(retriever_config)
    info = rag.collection_info()
    print(f"Collection: {info.get('count', 0)} chunks indexed")
    print()

    # Run evaluation
    results = []
    start_time = time.time()

    for i, fact in enumerate(facts):
        print(f"[{i+1}/{len(facts)}] Evaluating fact {fact['id']}: {fact.get('category', 'unknown')}...")

        try:
            result = evaluate_fact(
                fact,
                rag,
                n_results=args.n_results,
                use_llm=(not args.no_llm),
                check_consistency=True,
            )
            results.append(result)

            # Print per-fact summary line
            def rank_str(rank):
                return f"rank {rank}" if rank else "MISS"

            print(
                f"  FR: {rank_str(result.rank_fr)} "
                f"(R@3={result.recall_at_3_fr:.0f} nDCG@5={result.ndcg_at_5_fr:.2f} MRR={result.mrr_fr:.2f}) | "
                f"NL: {rank_str(result.rank_nl)} "
                f"(R@3={result.recall_at_3_nl:.0f} nDCG@5={result.ndcg_at_5_nl:.2f} MRR={result.mrr_nl:.2f})"
            )

            if result.consistency_score is not None:
                flag = " [FLAGGED]" if result.flagged else ""
                print(f"  Consistency: {result.consistency_score:.2f}{flag}")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    elapsed = time.time() - start_time
    print(f"\nEvaluation completed in {elapsed:.1f}s")

    # Compute summary
    summary = compute_summary(results)

    # Save results
    results_path = output_dir / 'results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'retriever': retriever_type,
                'n_results': args.n_results,
                'use_llm': not args.no_llm,
            },
            'summary': summary,
            'results': [asdict(r) for r in results],
        }, f, indent=2)
    print(f"Results saved to: {results_path}")

    # Save summary markdown
    summary_path = output_dir / 'summary.md'
    write_summary_markdown(summary, summary_path)
    print(f"Summary saved to: {summary_path}")

    # Print terminal summary
    r = summary['retrieval']
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total facts: {summary['total_facts']}")
    print(f"{'Metric':<12} {'FR':>8} {'NL':>8}")
    print(f"{'Recall@3':<12} {r['recall_at_3_fr']:>8.3f} {r['recall_at_3_nl']:>8.3f}")
    print(f"{'Recall@5':<12} {r['recall_at_5_fr']:>8.3f} {r['recall_at_5_nl']:>8.3f}")
    print(f"{'nDCG@3':<12} {r['ndcg_at_3_fr']:>8.3f} {r['ndcg_at_3_nl']:>8.3f}")
    print(f"{'nDCG@5':<12} {r['ndcg_at_5_fr']:>8.3f} {r['ndcg_at_5_nl']:>8.3f}")
    print(f"{'MRR':<12} {r['mrr_fr']:>8.3f} {r['mrr_nl']:>8.3f}")

    if summary.get('consistency'):
        c = summary['consistency']
        print(f"Consistency - Avg: {c['avg_score']:.3f}, Flagged: {c['flagged_rate']:.1%}")

    if summary.get('llm'):
        lm = summary['llm']
        print(f"LLM Accuracy - FR: {lm['accuracy_fr']:.1%}, NL: {lm['accuracy_nl']:.1%}")

    print("=" * 70)


if __name__ == '__main__':
    main()
