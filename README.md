# Belgian Bilingual Corporate Agent

**Experimental free-time project (work in progress)**  
Multi-agent system analyzing Belgian corporate reports with built-in cross-lingual validation—exploiting Belgium's bilingual publishing requirements to ensure extraction accuracy across French and Dutch.

---

## Architecture Overview

**Bilingual Validation Architecture:**

1. **User Query:**  
   Example: "What is BNP Paribas Fortis's digital investment for 2023?"

2. **Router Agent:**  
   Determines which companies and analyses are relevant.

3. **Company Data Collector Agent:**  
   Collects French and Dutch reports for selected companies.

4. **French Agent & Dutch Agent:**  
   Extracts relevant information from each language version.

5. **Validation Agent:**  
   Compares extracted data across languages for consistency:
   - Checks if numbers, KPIs, and citations match.
   - Flags discrepancies for review.
   - Assigns confidence scores.

6. **Output:**  
   Presents validated results with confidence levels and references.

---

## Example Scenarios

- **Validated Financial Extraction:**  
  Confirms figures match across FR/NL reports (e.g., digital investment amounts).

- **Discrepancy Detection:**  
  Flags mismatches (e.g., employee counts differ between FR and NL versions).

- **Bilingual Query Handling:**  
  Handles queries in either language, validates against both report versions, and responds in the user's language.

---

## Project Structure

```
src/
  agents/
    router_agent.py
    french_agent.py
    dutch_agent.py
    validation_agent.py
    company_data_collector_agent.py
  core/
    query_processor.py
    report_extractor.py
  data/
    sample_reports/
  utils/
    language_detect.py
    logger.py
  main.py
tests/
  agents/
  core/
  integration/
requirements.txt
README.md
```

---

## Getting Started

1. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

2. **Run the main program:**
   ```
   python src/main.py
   ```

3. **Run tests:**
   ```
   pytest
   ```

---

## Notes

- The system is modular: add new agents or data sources as needed.
- Designed for extensibility and robust cross-lingual validation.

---

## Retrieval-Augmented Generation (RAG) Pipeline

This project includes a multilingual RAG system for question answering over banking PDFs using only free/open-source tools:
- **Ollama** (local LLM, e.g., Qwen 2.5 7B)
- **ChromaDB** (local vector database)
- **PyPDF2** (PDF parsing)

### Setup

1. **Install extra dependencies:**
   ```sh
   pip install chromadb pypdf tqdm
   ```
2. **Install Ollama and a model:**
   - [Ollama install instructions](https://ollama.ai)
   - Pull a multilingual model (e.g., Qwen 2.5 7B):
     ```sh
     ollama pull qwen2.5:7b
     ollama serve
     ```

### Indexing PDFs

1. Download PDFs using the data collector agent (see above).
2. Index all PDFs for retrieval:
   ```sh
   python scripts/rag_cli.py index
   ```
   This will process all PDFs in `data/fetched_reports` and build a local vector DB.

### Querying the RAG System

Ask questions in any supported language:
```sh
python scripts/rag_cli.py query "What is the capital ratio for KBC Group?"
```

- The system will detect the language, retrieve relevant passages, and generate an answer using the local LLM.
- Answers are always sourced/cited with bank name and page number.

---

## File Locations
- RAG class: `src/rag/free_banking_rag.py`
- CLI script: `scripts/rag_cli.py`
- Downloaded PDFs: `data/fetched_reports/`
- Vector DB: `banking_db/`

---

## Example Usage

```sh
# Index all PDFs (run after downloading reports)
python scripts/rag_cli.py index

# Ask a question
python scripts/rag_cli.py query "Quels sont les principaux indicateurs financiers de BNP Paribas Fortis?"
```

---

## Troubleshooting
- Ensure Ollama is running and the model is pulled.
- Ensure all dependencies are installed.
- Place your PDFs in `data/fetched_reports` with the expected filenames (see `src/rag/free_banking_rag.py`).

---

**Contact:**  
Project maintained by Eduardo Brito.
