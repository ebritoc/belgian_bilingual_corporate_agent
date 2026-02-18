"""
Crosslingual Consistency Scorer using MaxSimE.

Compares how FR and NL queries align to the same document, producing a
consistency score that flags potential retrieval inconsistencies or
hallucination risks.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set, Tuple
import numpy as np

try:
    from maxsime import MaxSimExplainer, MaxSimExplanation
    MAXSIME_AVAILABLE = True
except ImportError:
    MAXSIME_AVAILABLE = False
    MaxSimExplanation = None

from retriever import RetrievedPassage


@dataclass
class ConsistencyResult:
    """Result of comparing FR and NL query alignments to a document."""
    score: float                          # 0.0 (inconsistent) to 1.0 (consistent)
    alignment_overlap: float              # Jaccard of matched doc token sets
    score_divergence: float               # normalized |MaxSim(q_fr, d) - MaxSim(q_nl, d)|
    matched_doc_tokens_fr: Set[int]       # doc token indices matched by FR query
    matched_doc_tokens_nl: Set[int]       # doc token indices matched by NL query
    explanation_fr: Optional[Any] = None  # MaxSimExplanation for FR
    explanation_nl: Optional[Any] = None  # MaxSimExplanation for NL
    flagged: bool = False                 # True if consistency < threshold

    @property
    def overlap_tokens(self) -> Set[int]:
        """Tokens matched by both FR and NL queries."""
        return self.matched_doc_tokens_fr & self.matched_doc_tokens_nl

    @property
    def fr_only_tokens(self) -> Set[int]:
        """Tokens matched only by FR query."""
        return self.matched_doc_tokens_fr - self.matched_doc_tokens_nl

    @property
    def nl_only_tokens(self) -> Set[int]:
        """Tokens matched only by NL query."""
        return self.matched_doc_tokens_nl - self.matched_doc_tokens_fr


class ConsistencyScorer:
    """
    Compare how FR and NL queries align to the same document chunk.

    High overlap in matched doc tokens → consistent → safe.
    Low overlap → inconsistent → flag for review.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        overlap_weight: float = 0.6,
        divergence_weight: float = 0.4
    ):
        """
        Initialize the consistency scorer.

        Args:
            threshold: Score below which results are flagged (default: 0.5)
            overlap_weight: Weight for alignment overlap in score (default: 0.6)
            divergence_weight: Weight for score divergence in score (default: 0.4)
        """
        if not MAXSIME_AVAILABLE:
            raise ImportError(
                "MaxSimE is required for ConsistencyScorer. "
                "Install it with: pip install git+https://github.com/ebritoc/MaxSimE.git"
            )

        self.threshold = threshold
        self.overlap_weight = overlap_weight
        self.divergence_weight = divergence_weight
        self.explainer = MaxSimExplainer()

    def score(
        self,
        query_fr_embeddings: np.ndarray,
        query_nl_embeddings: np.ndarray,
        doc_embeddings: np.ndarray,
        query_fr_tokens: List[str],
        query_nl_tokens: List[str],
        doc_tokens: List[str],
    ) -> ConsistencyResult:
        """
        Score consistency between FR and NL query alignments to a document.

        Args:
            query_fr_embeddings: FR query token embeddings (Q_fr_tokens, dim)
            query_nl_embeddings: NL query token embeddings (Q_nl_tokens, dim)
            doc_embeddings: Document token embeddings (D_tokens, dim)
            query_fr_tokens: FR query token strings
            query_nl_tokens: NL query token strings
            doc_tokens: Document token strings

        Returns:
            ConsistencyResult with score, explanations, and flag status
        """
        # Get MaxSimE explanations for both queries
        expl_fr = self.explainer.explain(
            query_fr_embeddings, doc_embeddings, query_fr_tokens, doc_tokens
        )
        expl_nl = self.explainer.explain(
            query_nl_embeddings, doc_embeddings, query_nl_tokens, doc_tokens
        )

        # Get matched doc token indices from each explanation
        matched_fr = set(expl_fr.matched_doc_indices)
        matched_nl = set(expl_nl.matched_doc_indices)

        # Compute Jaccard overlap
        intersection = matched_fr & matched_nl
        union = matched_fr | matched_nl
        alignment_overlap = len(intersection) / len(union) if union else 1.0

        # Compute normalized score divergence
        max_score = max(expl_fr.total_score, expl_nl.total_score, 1e-10)
        score_divergence = abs(expl_fr.total_score - expl_nl.total_score) / max_score

        # Combined consistency score
        consistency = (
            self.overlap_weight * alignment_overlap +
            self.divergence_weight * (1.0 - score_divergence)
        )

        return ConsistencyResult(
            score=consistency,
            alignment_overlap=alignment_overlap,
            score_divergence=score_divergence,
            matched_doc_tokens_fr=matched_fr,
            matched_doc_tokens_nl=matched_nl,
            explanation_fr=expl_fr,
            explanation_nl=expl_nl,
            flagged=consistency < self.threshold,
        )

    def score_passage_pair(
        self,
        passage_fr: RetrievedPassage,
        passage_nl: RetrievedPassage,
    ) -> Optional[ConsistencyResult]:
        """
        Score consistency for a pair of FR/NL retrieved passages.

        Both passages must have embeddings (from ColBERTRetriever).

        Args:
            passage_fr: French query retrieval result
            passage_nl: Dutch query retrieval result

        Returns:
            ConsistencyResult or None if embeddings not available
        """
        # Check embeddings are available
        if (passage_fr.query_embeddings is None or
            passage_fr.doc_embeddings is None or
            passage_nl.query_embeddings is None or
            passage_nl.doc_embeddings is None):
            return None

        # Check token strings are available
        if (passage_fr.query_token_strings is None or
            passage_fr.doc_token_strings is None or
            passage_nl.query_token_strings is None or
            passage_nl.doc_token_strings is None):
            return None

        # Both passages should reference the same document
        # Use the document embeddings and tokens from either (they should be the same)
        return self.score(
            query_fr_embeddings=passage_fr.query_embeddings,
            query_nl_embeddings=passage_nl.query_embeddings,
            doc_embeddings=passage_fr.doc_embeddings,
            query_fr_tokens=passage_fr.query_token_strings,
            query_nl_tokens=passage_nl.query_token_strings,
            doc_tokens=passage_fr.doc_token_strings,
        )

    def score_passages(
        self,
        passages_fr: List[RetrievedPassage],
        passages_nl: List[RetrievedPassage],
    ) -> List[Tuple[int, int, ConsistencyResult]]:
        """
        Score consistency for pairs of FR/NL retrieved passages.

        Matches passages by content similarity or metadata (same doc chunk).

        Args:
            passages_fr: List of passages from FR query
            passages_nl: List of passages from NL query

        Returns:
            List of (fr_idx, nl_idx, ConsistencyResult) tuples
        """
        results = []

        # Simple matching: pair by content hash or metadata
        for i, p_fr in enumerate(passages_fr):
            for j, p_nl in enumerate(passages_nl):
                # Check if same document chunk (by metadata)
                if self._is_same_chunk(p_fr.metadata, p_nl.metadata):
                    result = self.score_passage_pair(p_fr, p_nl)
                    if result is not None:
                        results.append((i, j, result))

        return results

    def _is_same_chunk(self, meta_fr: Dict, meta_nl: Dict) -> bool:
        """Check if two metadata dicts reference the same document chunk."""
        # Match by bank and page
        return (
            meta_fr.get('bank') == meta_nl.get('bank') and
            meta_fr.get('page') == meta_nl.get('page')
        )


def format_consistency_markdown(result: ConsistencyResult, doc_tokens: List[str] = None) -> str:
    """
    Format consistency result as markdown for display.

    Args:
        result: ConsistencyResult to format
        doc_tokens: Optional document token strings for display

    Returns:
        Markdown string with emoji status and details
    """
    icon = "🟢" if not result.flagged else "🔴"
    status = "Consistent" if not result.flagged else "Inconsistent"

    lines = [
        f"{icon} **{status}** (score: {result.score:.2f})",
        "",
        f"- Alignment overlap: {result.alignment_overlap:.1%}",
        f"- Score divergence: {result.score_divergence:.1%}",
    ]

    if doc_tokens:
        # Show matched tokens
        fr_tokens = [doc_tokens[i] for i in sorted(result.matched_doc_tokens_fr) if i < len(doc_tokens)]
        nl_tokens = [doc_tokens[i] for i in sorted(result.matched_doc_tokens_nl) if i < len(doc_tokens)]
        overlap_tokens = [doc_tokens[i] for i in sorted(result.overlap_tokens) if i < len(doc_tokens)]

        lines.append(f"- FR matched: {', '.join(fr_tokens[:10])}{'...' if len(fr_tokens) > 10 else ''}")
        lines.append(f"- NL matched: {', '.join(nl_tokens[:10])}{'...' if len(nl_tokens) > 10 else ''}")
        lines.append(f"- Shared: {', '.join(overlap_tokens[:10])}{'...' if len(overlap_tokens) > 10 else ''}")

    return '\n'.join(lines)
