#!/usr/bin/env python3
"""
Benchmark validation script.

Validates the evaluation benchmark JSON file for:
- Required fields present
- Difficulty distribution
- Category distribution
- Bank/year coverage

Usage:
    python scripts/validate_benchmark.py
    python scripts/validate_benchmark.py --benchmark path/to/benchmark.json
"""
import argparse
import json
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any


REQUIRED_FIELDS = [
    'id',
    'question_fr',
    'question_nl',
    'ground_truth',
    'source_bank',
]

OPTIONAL_FIELDS = [
    'question_en',
    'ground_truth_type',
    'source_year',
    'source_page_fr',
    'source_page_nl',
    'source_language',
    'category',
    'difficulty',
    'notes',
]

VALID_DIFFICULTIES = ['easy', 'medium', 'hard', 'adversarial']
VALID_CATEGORIES = [
    'capital_ratio', 'balance_sheet', 'profitability', 'fee_income',
    'workforce', 'liquidity', 'digital', 'dividend', 'general', 'other'
]


def validate_fact(fact: dict, idx: int) -> List[str]:
    """Validate a single fact and return list of issues."""
    issues = []

    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in fact:
            issues.append(f"Fact {idx}: Missing required field '{field}'")
        elif not fact[field]:
            issues.append(f"Fact {idx}: Empty required field '{field}'")

    # Validate difficulty if present
    if 'difficulty' in fact:
        if fact['difficulty'] not in VALID_DIFFICULTIES:
            issues.append(
                f"Fact {idx}: Invalid difficulty '{fact['difficulty']}'. "
                f"Valid values: {VALID_DIFFICULTIES}"
            )

    # Validate category if present
    if 'category' in fact:
        if fact['category'] not in VALID_CATEGORIES:
            issues.append(
                f"Fact {idx}: Unknown category '{fact['category']}'. "
                f"Consider adding to VALID_CATEGORIES or use 'other'"
            )

    # Check question language consistency
    q_fr = fact.get('question_fr', '')
    q_nl = fact.get('question_nl', '')

    if q_fr and q_nl and q_fr == q_nl:
        issues.append(f"Fact {idx}: FR and NL questions are identical")

    return issues


def compute_statistics(facts: List[dict]) -> Dict[str, Any]:
    """Compute statistics about the benchmark."""
    stats = {
        'total_facts': len(facts),
        'by_difficulty': Counter(),
        'by_category': Counter(),
        'by_bank': Counter(),
        'by_year': Counter(),
        'by_bank_year': Counter(),
        'has_page_refs': 0,
        'has_english': 0,
    }

    for fact in facts:
        stats['by_difficulty'][fact.get('difficulty', 'unknown')] += 1
        stats['by_category'][fact.get('category', 'unknown')] += 1
        stats['by_bank'][fact.get('source_bank', 'unknown')] += 1

        year = fact.get('source_year')
        if year:
            stats['by_year'][year] += 1
            bank = fact.get('source_bank', 'unknown')
            stats['by_bank_year'][f"{bank}_{year}"] += 1

        if fact.get('source_page_fr') or fact.get('source_page_nl'):
            stats['has_page_refs'] += 1

        if fact.get('question_en'):
            stats['has_english'] += 1

    return stats


def check_distribution(stats: Dict[str, Any]) -> List[str]:
    """Check if distribution meets target criteria."""
    warnings = []
    total = stats['total_facts']

    if total < 20:
        warnings.append(f"Low fact count: {total} (target: 200+)")

    # Check difficulty distribution
    diff_dist = stats['by_difficulty']
    if diff_dist:
        easy_pct = diff_dist.get('easy', 0) / total
        medium_pct = diff_dist.get('medium', 0) / total
        hard_pct = diff_dist.get('hard', 0) / total
        adversarial_pct = diff_dist.get('adversarial', 0) / total

        if easy_pct < 0.2:
            warnings.append(f"Low 'easy' facts: {easy_pct:.0%} (target: ~30%)")
        if medium_pct < 0.2:
            warnings.append(f"Low 'medium' facts: {medium_pct:.0%} (target: ~30%)")
        if hard_pct < 0.15:
            warnings.append(f"Low 'hard' facts: {hard_pct:.0%} (target: ~25%)")

    # Check bank coverage
    bank_dist = stats['by_bank']
    if len(bank_dist) < 2:
        warnings.append(f"Low bank coverage: {len(bank_dist)} banks")

    return warnings


def print_report(stats: Dict[str, Any], issues: List[str], warnings: List[str]) -> None:
    """Print validation report."""
    print("=" * 70)
    print("BENCHMARK VALIDATION REPORT")
    print("=" * 70)

    print(f"\nTotal facts: {stats['total_facts']}")
    print(f"Facts with page references: {stats['has_page_refs']}")
    print(f"Facts with English questions: {stats['has_english']}")

    print("\n--- By Difficulty ---")
    for diff, count in sorted(stats['by_difficulty'].items()):
        pct = count / stats['total_facts'] * 100
        print(f"  {diff}: {count} ({pct:.1f}%)")

    print("\n--- By Category ---")
    for cat, count in sorted(stats['by_category'].items(), key=lambda x: -x[1]):
        pct = count / stats['total_facts'] * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")

    print("\n--- By Bank ---")
    for bank, count in sorted(stats['by_bank'].items(), key=lambda x: -x[1]):
        pct = count / stats['total_facts'] * 100
        print(f"  {bank}: {count} ({pct:.1f}%)")

    if stats['by_year']:
        print("\n--- By Year ---")
        for year, count in sorted(stats['by_year'].items()):
            print(f"  {year}: {count}")

    if issues:
        print("\n" + "=" * 70)
        print(f"ERRORS ({len(issues)})")
        print("=" * 70)
        for issue in issues[:20]:  # Limit output
            print(f"  - {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more")

    if warnings:
        print("\n" + "=" * 70)
        print(f"WARNINGS ({len(warnings)})")
        print("=" * 70)
        for warning in warnings:
            print(f"  - {warning}")

    print("\n" + "=" * 70)
    if issues:
        print("VALIDATION FAILED")
    elif warnings:
        print("VALIDATION PASSED WITH WARNINGS")
    else:
        print("VALIDATION PASSED")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Validate benchmark JSON')
    parser.add_argument('--benchmark', type=str, default='tests/eval_benchmark.json',
                        help='Path to benchmark JSON file')

    args = parser.parse_args()

    # Load benchmark
    benchmark_path = Path(__file__).parent.parent / args.benchmark
    if not benchmark_path.exists():
        print(f"Error: Benchmark file not found: {benchmark_path}")
        sys.exit(1)

    with open(benchmark_path, 'r', encoding='utf-8') as f:
        benchmark = json.load(f)

    facts = benchmark.get('facts', [])
    if not facts:
        print("Error: No facts found in benchmark")
        sys.exit(1)

    # Validate each fact
    issues = []
    for idx, fact in enumerate(facts):
        issues.extend(validate_fact(fact, idx))

    # Compute statistics
    stats = compute_statistics(facts)

    # Check distribution
    warnings = check_distribution(stats)

    # Print report
    print_report(stats, issues, warnings)

    # Exit with appropriate code
    sys.exit(1 if issues else 0)


if __name__ == '__main__':
    main()
