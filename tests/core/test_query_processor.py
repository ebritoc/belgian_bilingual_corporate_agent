from src.core.query_processor import QueryProcessor

def test_process():
    processor = QueryProcessor()
    results = processor.process("digital investment")
    assert isinstance(results, list)
    assert "company" in results[0]
