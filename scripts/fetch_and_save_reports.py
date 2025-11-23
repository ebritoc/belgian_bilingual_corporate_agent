import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from agents.company_data_collector_agent import CompanyDataCollectorAgent

if __name__ == "__main__":
    companies = [
        "BNP Paribas Fortis",
        "KBC Group"
    ]
    agent = CompanyDataCollectorAgent(companies)
    results = agent.fetch_and_save_pdfs(save_dir="data/fetched_reports")
    for result in results:
        status = "OK" if result["success"] else f"FAILED: {result['error']}"
        print(f"{result['company']} ({result['lang']}): {status}")
    print("PDF reports fetched and saved to data/fetched_reports.")
