import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm

#TODO: Let an agent find the latest reports automatically
belgian_companies = {
    "BNP Paribas Fortis": {
        "FR": "https://www.bnpparibasfortis.com/docs/default-source/pdf-(fr)/rapports-financiers-2024-2/rapport_annuel_2024_bnp_paribas_fortis_fr.pdf",
        "NL": "https://www.bnpparibasfortis.com/docs/default-source/pdf-(nl)/financi%C3%ABle-verslagen-2024-2/jaarverslag_2024_bnp_paribas_fortis_nl.pdf",
        "EN": "https://www.bnpparibasfortis.be/rsc/documents/investors/annual_report_2024_bnp_paribas_fortis_en.pdf"
    },
    "KBC Group": {
        "FR": "https://www.kbc.com/content/dam/kbccom/doc/investor-relations/Results/jvs-2024/jvs-2024-gr-fr.pdf",
        "NL": "https://www.kbc.com/content/dam/kbccom/doc/investor-relations/Results/jvs-2024/jvs-2024-grp-nl.pdf",
        "EN": "https://www.kbc.com/content/dam/kbccom/doc/investor-relations/Results/jvs-2024/jvs-2024-grp-en.pdf"
    },
}

class CompanyDataCollectorAgent:
    """
    Collects corporate report data for selected Belgian companies.
    """

    def __init__(self, companies):
        self.companies = companies

    def fetch_report_content(self, url):
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            return response.text
        except Exception as e:
            return f"Error fetching {url}: {e}"

    def fetch_reports(self):
        """
        Fetches reports for the specified companies from the web in parallel.
        Returns a dict: {company_name: {'fr': fr_report, 'nl': nl_report}}
        """
        reports = {}

        def fetch_for_company(company):
            company_info = belgian_companies.get(company, {})
            fr_url = company_info.get("french")
            nl_url = company_info.get("dutch")
            return company, {
                'fr': self.fetch_report_content(fr_url) if fr_url else None,
                'nl': self.fetch_report_content(nl_url) if nl_url else None
            }

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fetch_for_company, company): company for company in self.companies}
            iterator = tqdm(as_completed(futures), total=len(futures), desc="Fetching reports")
            for future in iterator:
                company, result = future.result()
                reports[company] = result
        return reports


    def download_pdf(self, url, filepath, retries=3, timeout=60):
        for attempt in range(retries):
            try:
                response = requests.get(url, stream=True, timeout=timeout)
                response.raise_for_status()
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return True, None
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(5)  # wait before retrying
                else:
                    return False, str(e)

    def fetch_and_save_reports(self, save_dir="../data/fetched_reports"):
        """
        Fetches reports and saves them as HTML files in the specified directory.
        Each file is named as <company>_<lang>.html
        """
        os.makedirs(save_dir, exist_ok=True)
        reports = self.fetch_reports()
        for company, langs in reports.items():
            for lang, content in langs.items():
                safe_company = company.replace(" ", "_").replace("/", "-")
                filename = f"{safe_company}_{lang}.html"
                filepath = os.path.join(save_dir, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content if content else "")
        return True

    def fetch_and_save_pdfs(self, save_dir="data/fetched_reports"):
        os.makedirs(save_dir, exist_ok=True)
        tasks = []
        for company in self.companies:
            company_info = belgian_companies.get(company, {})
            for lang in ("FR", "NL", "EN"):
                url = company_info.get(lang)
                if url:
                    safe_company = company.replace(" ", "_").replace("/", "-")
                    filename = f"{safe_company}_{lang}.pdf"
                    filepath = os.path.join(save_dir, filename)
                    print(f"Downloading {url} to {filepath}")
                    tasks.append((url, filepath, company, lang))
        results = []
        print("Tasks to download:", tasks)
        for url, filepath, company, lang in tqdm(tasks, desc="Downloading PDFs"):
            success, error = self.download_pdf(url, filepath)
            results.append({
                "company": company,
                "lang": lang,
                "url": url,
                "filepath": filepath,
                "success": success,
                "error": error
            })
        return results
