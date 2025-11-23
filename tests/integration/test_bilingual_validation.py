from src.core.query_processor import QueryProcessor

def test_bilingual_validation():
    processor = QueryProcessor()
    results = processor.process("Compare digital investment")
    for result in results:
        assert "validation" in result
