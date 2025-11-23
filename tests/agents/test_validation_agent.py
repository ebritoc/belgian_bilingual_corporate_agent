from src.agents.validation_agent import ValidationAgent

def test_validate_match():
    agent = ValidationAgent()
    result = agent.validate("same", "same")
    assert result["status"] == "match"

def test_validate_discrepancy():
    agent = ValidationAgent()
    result = agent.validate("a", "b")
    assert result["status"] == "discrepancy"
