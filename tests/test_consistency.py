"""
Tests for the ConsistencyScorer.

These tests verify the crosslingual consistency scoring logic.
"""
import sys
from pathlib import Path
import pytest
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Check if MaxSimE is available
try:
    from maxsime import MaxSimExplainer
    MAXSIME_AVAILABLE = True
except ImportError:
    MAXSIME_AVAILABLE = False

# Skip all tests if MaxSimE not available
pytestmark = pytest.mark.skipif(
    not MAXSIME_AVAILABLE,
    reason="MaxSimE not installed"
)


@pytest.fixture
def scorer():
    """Create a ConsistencyScorer instance."""
    from consistency import ConsistencyScorer
    return ConsistencyScorer(threshold=0.5)


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings for testing."""
    np.random.seed(42)
    dim = 128

    # Create realistic-ish embeddings
    # Query FR: 5 tokens
    query_fr = np.random.randn(5, dim).astype(np.float32)
    query_fr = query_fr / np.linalg.norm(query_fr, axis=1, keepdims=True)

    # Query NL: 5 tokens (similar to FR for testing)
    query_nl = query_fr + np.random.randn(5, dim).astype(np.float32) * 0.1
    query_nl = query_nl / np.linalg.norm(query_nl, axis=1, keepdims=True)

    # Document: 20 tokens
    doc = np.random.randn(20, dim).astype(np.float32)
    doc = doc / np.linalg.norm(doc, axis=1, keepdims=True)

    return {
        'query_fr': query_fr,
        'query_nl': query_nl,
        'doc': doc,
        'query_fr_tokens': ['quel', 'est', 'le', 'ratio', 'CET1'],
        'query_nl_tokens': ['wat', 'is', 'de', 'ratio', 'CET1'],
        'doc_tokens': [f'tok_{i}' for i in range(20)],
    }


class TestConsistencyScorer:
    """Tests for ConsistencyScorer."""

    def test_initialization(self, scorer):
        """Test scorer initializes correctly."""
        assert scorer.threshold == 0.5
        assert scorer.overlap_weight == 0.6
        assert scorer.divergence_weight == 0.4

    def test_score_returns_result(self, scorer, mock_embeddings):
        """Test that score() returns a ConsistencyResult."""
        from consistency import ConsistencyResult

        result = scorer.score(
            query_fr_embeddings=mock_embeddings['query_fr'],
            query_nl_embeddings=mock_embeddings['query_nl'],
            doc_embeddings=mock_embeddings['doc'],
            query_fr_tokens=mock_embeddings['query_fr_tokens'],
            query_nl_tokens=mock_embeddings['query_nl_tokens'],
            doc_tokens=mock_embeddings['doc_tokens'],
        )

        assert isinstance(result, ConsistencyResult)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.alignment_overlap <= 1.0
        assert 0.0 <= result.score_divergence <= 1.0
        assert isinstance(result.matched_doc_tokens_fr, set)
        assert isinstance(result.matched_doc_tokens_nl, set)
        assert isinstance(result.flagged, bool)

    def test_identical_queries_high_consistency(self, scorer, mock_embeddings):
        """Test that identical FR/NL queries produce high consistency."""
        # Use identical embeddings for FR and NL
        result = scorer.score(
            query_fr_embeddings=mock_embeddings['query_fr'],
            query_nl_embeddings=mock_embeddings['query_fr'],  # Same as FR
            doc_embeddings=mock_embeddings['doc'],
            query_fr_tokens=mock_embeddings['query_fr_tokens'],
            query_nl_tokens=mock_embeddings['query_fr_tokens'],
            doc_tokens=mock_embeddings['doc_tokens'],
        )

        # Identical queries should have perfect overlap
        assert result.alignment_overlap == 1.0
        assert result.score_divergence == 0.0
        assert result.score >= 0.9
        assert not result.flagged

    def test_orthogonal_queries_low_consistency(self, scorer):
        """Test that orthogonal FR/NL queries produce low consistency."""
        np.random.seed(123)
        dim = 128

        # Create orthogonal query embeddings
        query_fr = np.random.randn(5, dim).astype(np.float32)
        query_fr = query_fr / np.linalg.norm(query_fr, axis=1, keepdims=True)

        # Orthogonal: negate and add noise
        query_nl = -query_fr + np.random.randn(5, dim).astype(np.float32) * 0.5
        query_nl = query_nl / np.linalg.norm(query_nl, axis=1, keepdims=True)

        doc = np.random.randn(20, dim).astype(np.float32)
        doc = doc / np.linalg.norm(doc, axis=1, keepdims=True)

        result = scorer.score(
            query_fr_embeddings=query_fr,
            query_nl_embeddings=query_nl,
            doc_embeddings=doc,
            query_fr_tokens=['a', 'b', 'c', 'd', 'e'],
            query_nl_tokens=['f', 'g', 'h', 'i', 'j'],
            doc_tokens=[f'tok_{i}' for i in range(20)],
        )

        # Very different queries should have low overlap
        assert result.alignment_overlap < 0.5
        # Score should be lower, possibly flagged
        assert result.score < 0.8

    def test_threshold_flagging(self):
        """Test that results below threshold are flagged."""
        from consistency import ConsistencyScorer
        np.random.seed(456)
        dim = 128

        # Create scorer with high threshold
        scorer = ConsistencyScorer(threshold=0.8)

        # Create somewhat different queries
        query_fr = np.random.randn(5, dim).astype(np.float32)
        query_nl = np.random.randn(5, dim).astype(np.float32)
        doc = np.random.randn(20, dim).astype(np.float32)

        result = scorer.score(
            query_fr_embeddings=query_fr,
            query_nl_embeddings=query_nl,
            doc_embeddings=doc,
            query_fr_tokens=['a'] * 5,
            query_nl_tokens=['b'] * 5,
            doc_tokens=['c'] * 20,
        )

        # With random embeddings and high threshold, should likely be flagged
        if result.score < 0.8:
            assert result.flagged
        else:
            assert not result.flagged

    def test_overlap_tokens_property(self, scorer, mock_embeddings):
        """Test that overlap_tokens returns correct intersection."""
        result = scorer.score(
            query_fr_embeddings=mock_embeddings['query_fr'],
            query_nl_embeddings=mock_embeddings['query_nl'],
            doc_embeddings=mock_embeddings['doc'],
            query_fr_tokens=mock_embeddings['query_fr_tokens'],
            query_nl_tokens=mock_embeddings['query_nl_tokens'],
            doc_tokens=mock_embeddings['doc_tokens'],
        )

        # Check overlap is correct intersection
        expected_overlap = result.matched_doc_tokens_fr & result.matched_doc_tokens_nl
        assert result.overlap_tokens == expected_overlap

    def test_fr_only_nl_only_tokens(self, scorer, mock_embeddings):
        """Test fr_only_tokens and nl_only_tokens properties."""
        result = scorer.score(
            query_fr_embeddings=mock_embeddings['query_fr'],
            query_nl_embeddings=mock_embeddings['query_nl'],
            doc_embeddings=mock_embeddings['doc'],
            query_fr_tokens=mock_embeddings['query_fr_tokens'],
            query_nl_tokens=mock_embeddings['query_nl_tokens'],
            doc_tokens=mock_embeddings['doc_tokens'],
        )

        # FR only = FR - NL
        expected_fr_only = result.matched_doc_tokens_fr - result.matched_doc_tokens_nl
        assert result.fr_only_tokens == expected_fr_only

        # NL only = NL - FR
        expected_nl_only = result.matched_doc_tokens_nl - result.matched_doc_tokens_fr
        assert result.nl_only_tokens == expected_nl_only


class TestFormatConsistencyMarkdown:
    """Tests for format_consistency_markdown function."""

    def test_consistent_result_green(self, scorer, mock_embeddings):
        """Test that consistent results show green icon."""
        from consistency import format_consistency_markdown

        # Use identical queries for high consistency
        result = scorer.score(
            query_fr_embeddings=mock_embeddings['query_fr'],
            query_nl_embeddings=mock_embeddings['query_fr'],
            doc_embeddings=mock_embeddings['doc'],
            query_fr_tokens=mock_embeddings['query_fr_tokens'],
            query_nl_tokens=mock_embeddings['query_fr_tokens'],
            doc_tokens=mock_embeddings['doc_tokens'],
        )

        markdown = format_consistency_markdown(result)

        assert "🟢" in markdown
        assert "Consistent" in markdown
        assert "score:" in markdown

    def test_inconsistent_result_red(self):
        """Test that inconsistent results show red icon."""
        from consistency import ConsistencyResult, format_consistency_markdown

        # Create a flagged result directly
        result = ConsistencyResult(
            score=0.3,
            alignment_overlap=0.2,
            score_divergence=0.8,
            matched_doc_tokens_fr={0, 1, 2},
            matched_doc_tokens_nl={5, 6, 7},
            flagged=True,
        )

        markdown = format_consistency_markdown(result)

        assert "🔴" in markdown
        assert "Inconsistent" in markdown

    def test_with_doc_tokens(self):
        """Test markdown includes token details when provided."""
        from consistency import ConsistencyResult, format_consistency_markdown

        result = ConsistencyResult(
            score=0.7,
            alignment_overlap=0.6,
            score_divergence=0.2,
            matched_doc_tokens_fr={0, 1, 2},
            matched_doc_tokens_nl={1, 2, 3},
            flagged=False,
        )

        doc_tokens = ['capital', 'ratio', 'CET1', 'percent']
        markdown = format_consistency_markdown(result, doc_tokens)

        assert "FR matched:" in markdown
        assert "NL matched:" in markdown
        assert "Shared:" in markdown
