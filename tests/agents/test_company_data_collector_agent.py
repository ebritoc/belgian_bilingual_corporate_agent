from src.agents.company_data_collector_agent import CompanyDataCollectorAgent

def test_fetch_reports():
    agent = CompanyDataCollectorAgent(["BNP Paribas Fortis"])
    reports = agent.fetch_reports()
    assert "BNP Paribas Fortis" in reports
    assert "fr" in reports["BNP Paribas Fortis"]
    assert "nl" in reports["BNP Paribas Fortis"]
