from src.agents.dutch_agent import DutchAgent

def test_extract():
    agent = DutchAgent()
    result = agent.extract("Rapport in het Nederlands", "investering")
    assert "NL data" in result["extracted"]
