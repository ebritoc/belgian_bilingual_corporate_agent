"""
Test cases for RAG integration testing.

USER ACTION REQUIRED: Please extract factual Q&A pairs from the BNP Paribas Fortis
and KBC Group reports and fill in this template.

Guidelines for creating good test cases:
- Answer should be a concrete fact (number, date, yes/no)
- Should appear clearly in the documents
- Verifiable by manual inspection
- Not too broad ("Tell me about the company")
- Not too obscure (buried in appendix)
"""

# Template - USER SHOULD FILL THIS IN
TEST_CASES = [
    # Example 1: Simple factual extraction (English)
    {
        "question": "What is KBC Group's Common Equity Tier 1 ratio?",
        "expected_answer_contains": ["15.8%", "CET1", "capital"],  # Flexible matching
        "expected_sources": ["KBC_Group_EN.pdf"],  # Which docs should be retrieved
        "language": "en",
        "category": "factual_extraction",
        "notes": "Should retrieve from English KBC report"
    },

    # Example 2: Factual extraction (French)
    {
        "question": "Quel est le résultat net de BNP Paribas Fortis?",
        "expected_answer_contains": ["résultat net", "bénéfice", "million"],
        "expected_sources": ["BNP_Paribas_Fortis_FR.pdf"],
        "language": "fr",
        "category": "factual_extraction",
        "notes": "Should retrieve from French BNP report"
    },

    # Example 3: Factual extraction (Dutch)
    {
        "question": "Wat is de nettowinst van KBC Groep?",
        "expected_answer_contains": ["nettowinst", "miljoen"],
        "expected_sources": ["KBC_Group_NL.pdf"],
        "language": "nl",
        "category": "factual_extraction",
        "notes": "Should retrieve from Dutch KBC report"
    },

    # Example 4: Bilingual consistency (French vs Dutch)
    {
        "question": "Quel est le ratio de fonds propres de BNP Paribas Fortis?",
        "expected_answer_contains": ["ratio", "capital", "%"],
        "expected_sources": ["BNP_Paribas_Fortis_FR.pdf"],
        "language": "fr",
        "category": "bilingual_consistency",
        "pair_with": 5,  # Links to test case #5
        "notes": "French version of bilingual pair"
    },

    {
        "question": "Wat is de kapitaalratio van BNP Paribas Fortis?",
        "expected_answer_contains": ["ratio", "kapitaal", "%"],
        "expected_sources": ["BNP_Paribas_Fortis_NL.pdf"],
        "language": "nl",
        "category": "bilingual_consistency",
        "pair_with": 4,  # Links to test case #4
        "notes": "Dutch version of bilingual pair - should have same numbers as French"
    },

    # Example 6: Cross-language retrieval (English query, multilingual docs)
    {
        "question": "What were KBC's main risks in 2023?",
        "expected_answer_contains": ["risk", "credit", "market"],
        "expected_sources": ["KBC_Group_EN.pdf", "KBC_Group_FR.pdf", "KBC_Group_NL.pdf"],
        "language": "en",
        "category": "cross_language",
        "notes": "English query should retrieve from multiple language versions"
    },

    # ADD MORE TEST CASES HERE:
    # - 2-3 more factual extraction tests
    # - 1-2 more bilingual consistency pairs
    # - 1 more cross-language test

]


# Helper function to validate test cases
def validate_test_cases():
    """Validate that test cases are well-formed."""
    categories = {"factual_extraction", "bilingual_consistency", "cross_language"}
    languages = {"en", "fr", "nl"}

    for i, test in enumerate(TEST_CASES):
        # Check required fields
        assert "question" in test, f"Test {i}: missing 'question'"
        assert "expected_answer_contains" in test, f"Test {i}: missing 'expected_answer_contains'"
        assert "expected_sources" in test, f"Test {i}: missing 'expected_sources'"
        assert "language" in test, f"Test {i}: missing 'language'"
        assert "category" in test, f"Test {i}: missing 'category'"

        # Check field types
        assert isinstance(test["question"], str), f"Test {i}: question must be string"
        assert isinstance(test["expected_answer_contains"], list), f"Test {i}: expected_answer_contains must be list"
        assert isinstance(test["expected_sources"], list), f"Test {i}: expected_sources must be list"

        # Check valid values
        assert test["language"] in languages, f"Test {i}: language must be en/fr/nl"
        assert test["category"] in categories, f"Test {i}: invalid category"

        # Check bilingual pairs
        if "pair_with" in test:
            pair_idx = test["pair_with"]
            assert 0 <= pair_idx < len(TEST_CASES), f"Test {i}: invalid pair_with index"
            assert test["category"] == "bilingual_consistency", f"Test {i}: pair_with only for bilingual tests"

    print(f"✓ All {len(TEST_CASES)} test cases are valid")


if __name__ == "__main__":
    validate_test_cases()
    print("\nTest case summary:")
    print(f"  Total: {len(TEST_CASES)}")
    print(f"  Factual extraction: {sum(1 for t in TEST_CASES if t['category'] == 'factual_extraction')}")
    print(f"  Bilingual consistency: {sum(1 for t in TEST_CASES if t['category'] == 'bilingual_consistency')}")
    print(f"  Cross-language: {sum(1 for t in TEST_CASES if t['category'] == 'cross_language')}")
