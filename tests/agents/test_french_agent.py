from src.agents.french_agent import FrenchAgent

def test_extract():
    agent = FrenchAgent()
    result = agent.extract("Rapport en français", "investissement")
    assert "FR data" in result["extracted"]
