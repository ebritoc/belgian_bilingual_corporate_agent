from agents.router_agent import RouterAgent
from agents.company_data_collector_agent import CompanyDataCollectorAgent
from agents.french_agent import FrenchAgent
from agents.dutch_agent import DutchAgent
from agents.validation_agent import ValidationAgent

class QueryProcessor:
    """
    Orchestrates the multi-agent workflow for a user query.
    """
    def __init__(self):
        self.router = RouterAgent()
        self.fr_agent = FrenchAgent()
        self.nl_agent = DutchAgent()
        self.validator = ValidationAgent()

    def process(self, query):
        companies = self.router.route(query)
        collector = CompanyDataCollectorAgent(companies)
        reports = collector.fetch_reports()
        results = []
        for company in companies:
            fr_report = reports[company]['fr']
            nl_report = reports[company]['nl']
            fr_data = self.fr_agent.extract(fr_report, query)
            nl_data = self.nl_agent.extract(nl_report, query)
            validation = self.validator.validate(fr_data, nl_data)
            results.append({
                "company": company,
                "fr_data": fr_data,
                "nl_data": nl_data,
                "validation": validation
            })
        return results
