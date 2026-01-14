"""
End-to-end integration tests for RAG system.

These tests use real Ollama LLM calls and require:
- Ollama running at http://localhost:11434
- Documents indexed in ChromaDB (run: python scripts/rag_cli.py index)
- qwen2.5:7b model available
"""
import pytest
import sys
import os
import requests

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from rag_service import RAGService
from test_cases import TEST_CASES


@pytest.fixture(scope="module")
def check_ollama():
    """Check if Ollama is running before running tests."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            pytest.skip("Ollama is not running at http://localhost:11434")
    except requests.exceptions.RequestException:
        pytest.skip("Ollama is not running at http://localhost:11434")


@pytest.fixture(scope="module")
def rag_service(check_ollama):
    """
    Initialize RAG service with real ChromaDB and Ollama.

    This fixture is module-scoped to avoid reinitializing for each test.
    """
    service = RAGService()

    # Verify collection has documents
    info = service.collection_info()
    if info['count'] == 0:
        pytest.fail(
            "ChromaDB collection is empty. Please run: python scripts/rag_cli.py index"
        )

    print(f"\n[OK] RAG service initialized with {info['count']} indexed chunks")
    return service


def check_answer_contains(answer: str, expected_keywords: list) -> bool:
    """
    Check if answer contains at least one of the expected keywords.

    Args:
        answer: Generated answer text
        expected_keywords: List of expected keywords/phrases

    Returns:
        True if any keyword is found in answer
    """
    answer_lower = answer.lower()
    for keyword in expected_keywords:
        if keyword.lower() in answer_lower:
            return True
    return False


def check_sources(sources: list, expected_source_patterns: list) -> bool:
    """
    Check if retrieved sources match expected patterns.

    Args:
        sources: List of source metadata dicts
        expected_source_patterns: List of expected source name patterns

    Returns:
        True if any source matches any pattern
    """
    for source in sources:
        source_name = source.get('source', '')
        bank_name = source.get('bank', '')
        combined = f"{bank_name} {source_name}".lower()

        for pattern in expected_source_patterns:
            if pattern.lower() in combined:
                return True
    return False


class TestFactualExtraction:
    """Test basic factual extraction from documents."""

    @pytest.mark.parametrize("test_case", [
        tc for tc in TEST_CASES if tc['category'] == 'factual_extraction'
    ], ids=lambda tc: tc['question'][:50])
    def test_factual_question(self, rag_service, test_case):
        """Test factual extraction with LLM generation."""
        question = test_case['question']
        expected_keywords = test_case['expected_answer_contains']
        expected_sources = test_case['expected_sources']

        # Query the RAG system
        result = rag_service.answer_question(question, n_results=5, use_llm=True)

        # Check we got sources
        assert result['num_sources'] > 0, "No sources retrieved"

        # Check answer is not empty
        assert result['answer'], "No answer generated"
        assert len(result['answer']) > 20, "Answer too short"

        # Check answer doesn't contain error messages
        assert "Error:" not in result['answer'], f"LLM error: {result['answer']}"

        # Check sources match expected
        sources_match = check_sources(result['sources'], expected_sources)
        if not sources_match:
            print(f"\nWarning: Expected sources {expected_sources}")
            print(f"Got sources: {[s.get('bank', '') + ' ' + s.get('source', '') for s in result['sources']]}")

        # Check answer contains expected keywords (flexible matching)
        keywords_found = check_answer_contains(result['answer'], expected_keywords)
        if not keywords_found:
            print(f"\nAnswer: {result['answer']}")
            print(f"Expected one of: {expected_keywords}")
            pytest.fail(f"Answer doesn't contain any expected keywords")


class TestRetrievalOnly:
    """Test retrieval quality without LLM generation (faster tests)."""

    @pytest.mark.parametrize("test_case", TEST_CASES, ids=lambda tc: tc['question'][:50])
    def test_retrieval_quality(self, rag_service, test_case):
        """Test that retrieval finds relevant documents."""
        question = test_case['question']
        expected_sources = test_case['expected_sources']

        # Retrieve without LLM
        result = rag_service.answer_question(question, n_results=5, use_llm=False)

        # Check we got sources
        assert result['num_sources'] > 0, "No sources retrieved"

        # Check sources match expected
        sources_match = check_sources(result['sources'], expected_sources)
        assert sources_match, (
            f"Retrieved sources don't match expected patterns.\n"
            f"Expected: {expected_sources}\n"
            f"Got: {[s.get('bank', '') + ' ' + s.get('source', '') for s in result['sources']]}"
        )


class TestBilingualConsistency:
    """Test bilingual consistency checking."""

    def test_bilingual_pair_employee_count(self, rag_service):
        """Test that French and Dutch queries return consistent employee numbers."""
        # Get bilingual test pair (Test 7 FR and Test 9 NL)
        test_fr = next(tc for tc in TEST_CASES if tc.get('question') and 'Combien' in tc['question'] and 'employés' in tc['question'])
        test_nl = next(tc for tc in TEST_CASES if tc.get('pair_with') == 7)

        # Query in both languages
        result_fr = rag_service.answer_question(test_fr['question'], n_results=3, use_llm=False)
        result_nl = rag_service.answer_question(test_nl['question'], n_results=3, use_llm=False)

        # Both should retrieve from BNP Paribas Fortis
        assert result_fr['num_sources'] > 0
        assert result_nl['num_sources'] > 0

        # Check both retrieve from same bank
        banks_fr = {s['bank'] for s in result_fr['sources']}
        banks_nl = {s['bank'] for s in result_nl['sources']}

        assert 'BNP Paribas Fortis' in banks_fr
        assert 'BNP Paribas Fortis' in banks_nl


class TestCrossLanguageRetrieval:
    """Test cross-language retrieval capabilities."""

    def test_english_query_multilingual_retrieval(self, rag_service):
        """Test that English query can retrieve from FR/NL documents."""
        test_case = next(tc for tc in TEST_CASES if tc['category'] == 'cross_language')

        result = rag_service.answer_question(test_case['question'], n_results=5, use_llm=False)

        # Check we got sources
        assert result['num_sources'] > 0, "No sources retrieved"

        # Check we retrieved from multiple languages (ideally)
        languages = {s['language'] for s in result['sources']}
        print(f"\nRetrieved languages: {languages}")

        # At minimum, should retrieve from KBC documents
        sources_match = check_sources(result['sources'], test_case['expected_sources'])
        assert sources_match, f"Should retrieve from KBC documents"


# Utility test to verify setup
class TestSetup:
    """Verify test environment is properly configured."""

    def test_ollama_accessible(self):
        """Verify Ollama is accessible."""
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        assert response.status_code == 200

        models = response.json().get('models', [])
        model_names = [m['name'] for m in models]
        print(f"\nAvailable models: {model_names}")

        # Check if qwen2.5:7b is available
        has_qwen = any('qwen2.5' in name for name in model_names)
        assert has_qwen, "qwen2.5:7b model not found in Ollama"

    def test_chromadb_has_data(self, rag_service):
        """Verify ChromaDB has indexed documents."""
        info = rag_service.collection_info()
        assert info['count'] > 1000, f"Too few documents indexed: {info['count']}"
        print(f"\n[OK] ChromaDB has {info['count']} indexed chunks")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"])
