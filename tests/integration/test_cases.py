"""
Test cases for RAG integration testing.

Extracted from BNP Paribas Fortis and KBC Group 2024 annual reports.
"""

TEST_CASES = [
    # === CAPITAL RATIOS ===

    # Test 1: KBC Group CET1 ratio (English)
    {
        "question": "What is KBC Group's Common Equity Tier 1 ratio?",
        "expected_answer_contains": ["13.9", "13,9", "capital"],
        "expected_sources": ["KBC_Group"],
        "language": "en",
        "category": "factual_extraction",
        "notes": "KBC Group transitional CET1 ratio was 13.9% end of 2024"
    },

    # Test 2: KBC Bank CET1 ratio (English)
    {
        "question": "What is KBC Bank's common equity ratio?",
        "expected_answer_contains": ["13.2", "13,2", "capital"],
        "expected_sources": ["KBC_Group"],
        "language": "en",
        "category": "factual_extraction",
        "notes": "KBC Bank transitional CET1 ratio was 13.2% end of 2024"
    },

    # Test 3: KBC capital ratio (French)
    {
        "question": "Quel est le ratio common equity de KBC Groupe?",
        "expected_answer_contains": ["13,9", "13.9", "capital"],
        "expected_sources": ["KBC_Group_FR"],
        "language": "fr",
        "category": "factual_extraction",
        "notes": "Should retrieve from French KBC report"
    },

    # Test 4: KBC capital ratio (Dutch)
    {
        "question": "Wat is de common equity ratio van KBC Groep?",
        "expected_answer_contains": ["13,9", "13.9", "kapitaal"],
        "expected_sources": ["KBC_Group_NL"],
        "language": "nl",
        "category": "factual_extraction",
        "notes": "Should retrieve from Dutch KBC report"
    },

    # === NET PROFIT ===

    # Test 5: KBC net result (English)
    {
        "question": "What is KBC Group's net result for 2024?",
        "expected_answer_contains": ["3,415", "3.415", "3 415", "million"],
        "expected_sources": ["KBC_Group"],
        "language": "en",
        "category": "factual_extraction",
        "notes": "KBC Group consolidated net result: EUR 3,415 million"
    },

    # Test 6: BNP Paribas Fortis net profit (Dutch)
    {
        "question": "Wat is de winst van BNP Paribas Fortis voor 2024?",
        "expected_answer_contains": ["2.436", "2,436", "2 436", "miljoen"],
        "expected_sources": ["BNP_Paribas_Fortis_NL"],
        "language": "nl",
        "category": "factual_extraction",
        "notes": "BNP Paribas Fortis: EUR 2,436.9 million profit"
    },

    # === NUMBER OF EMPLOYEES ===

    # Test 7: BNP Paribas Fortis employees (French)
    {
        "question": "Combien d'employés travaillent chez BNP Paribas Fortis?",
        "expected_answer_contains": ["35.000", "35 000", "35000"],
        "expected_sources": ["BNP_Paribas_Fortis_FR"],
        "language": "fr",
        "category": "factual_extraction",
        "notes": "BNP Paribas Fortis: ~35,000 employees (Dec 2024)"
    },

    # Test 8: KBC employees (English)
    {
        "question": "How many employees does KBC Group have?",
        "expected_answer_contains": ["38,074", "38 074", "38074"],
        "expected_sources": ["KBC_Group_EN"],
        "language": "en",
        "category": "factual_extraction",
        "notes": "KBC Group: 38,074 FTE employees (2024 average)"
    },

    # === BILINGUAL CONSISTENCY ===

    # Test 9 & 10: BNP Paribas Fortis employees (bilingual pair)
    {
        "question": "Hoeveel werknemers heeft BNP Paribas Fortis?",
        "expected_answer_contains": ["35.000", "35 000"],
        "expected_sources": ["BNP_Paribas_Fortis_NL"],
        "language": "nl",
        "category": "bilingual_consistency",
        "pair_with": 7,  # Paired with French version (Test 7)
        "notes": "Dutch version - should return same employee count as French"
    },

    # === CROSS-LANGUAGE RETRIEVAL ===

    # Test 11: English query on multilingual documents
    {
        "question": "What is the capital ratio of KBC?",
        "expected_answer_contains": ["13.9", "13,9", "ratio"],
        "expected_sources": ["KBC_Group"],  # Should retrieve from any language
        "language": "en",
        "category": "cross_language",
        "notes": "English query should retrieve from FR/NL/EN documents"
    },
]


# Helper function to validate test cases
def validate_test_cases():
    """Validate that test cases are well-formed."""
    categories = {"factual_extraction", "bilingual_consistency", "cross_language"}
    languages = {"en", "fr", "nl"}

    errors = []

    for i, test in enumerate(TEST_CASES, 1):
        # Check required fields
        if "question" not in test:
            errors.append(f"Test {i}: missing 'question'")
        if "expected_answer_contains" not in test:
            errors.append(f"Test {i}: missing 'expected_answer_contains'")
        if "expected_sources" not in test:
            errors.append(f"Test {i}: missing 'expected_sources'")
        if "language" not in test:
            errors.append(f"Test {i}: missing 'language'")
        if "category" not in test:
            errors.append(f"Test {i}: missing 'category'")

        # Check field types
        if "question" in test and not isinstance(test["question"], str):
            errors.append(f"Test {i}: question must be string")
        if "expected_answer_contains" in test and not isinstance(test["expected_answer_contains"], list):
            errors.append(f"Test {i}: expected_answer_contains must be list")
        if "expected_sources" in test and not isinstance(test["expected_sources"], list):
            errors.append(f"Test {i}: expected_sources must be list")

        # Check valid values
        if "language" in test and test["language"] not in languages:
            errors.append(f"Test {i}: language must be en/fr/nl")
        if "category" in test and test["category"] not in categories:
            errors.append(f"Test {i}: invalid category")

        # Check bilingual pairs
        if "pair_with" in test:
            pair_idx = test["pair_with"]
            if pair_idx < 1 or pair_idx > len(TEST_CASES):
                errors.append(f"Test {i}: invalid pair_with index")
            if test["category"] != "bilingual_consistency":
                errors.append(f"Test {i}: pair_with only for bilingual tests")

    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False

    print(f"[OK] All {len(TEST_CASES)} test cases are valid")
    return True


def print_summary():
    """Print summary of test cases."""
    print("\nTest case summary:")
    print(f"  Total: {len(TEST_CASES)}")

    by_category = {}
    for test in TEST_CASES:
        cat = test['category']
        by_category[cat] = by_category.get(cat, 0) + 1

    for cat, count in sorted(by_category.items()):
        print(f"  {cat.replace('_', ' ').title()}: {count}")

    by_language = {}
    for test in TEST_CASES:
        lang = test['language']
        by_language[lang] = by_language.get(lang, 0) + 1

    print("\n  By language:")
    for lang, count in sorted(by_language.items()):
        print(f"    {lang.upper()}: {count}")


if __name__ == "__main__":
    if validate_test_cases():
        print_summary()
