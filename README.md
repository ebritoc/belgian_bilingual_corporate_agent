# Belgian Bilingual Corporate Agent

**(Very) Experimental Free-time Project (large parts esentially vibe-coded)**  
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
    rag_agent.py                      # RAG public facade
    rag_orchestrator.py               # Orchestrator: coordination & bilingual logic
    pdf_ingestion_agent.py            # PDF parsing, chunking, quality filtering
    retrieval_agent.py                # Embeddings, indexing, hybrid search
    generation_agent.py               # Prompt building & LLM generation
    router_agent.py                   # Query routing
    french_agent.py                   # French extraction
    dutch_agent.py                    # Dutch extraction
    validation_agent.py               # Cross-language validation
    company_data_collector_agent.py   # Data collection
  core/
    query_processor.py
    report_extractor.py
  data/
    fetched_reports/                  # Downloaded PDFs
    sample_reports/                   # Synthetic bilingual test data
  rag/
    free_banking_rag.py               # Deprecated; shim to RAGAgent
  utils/
    language_detect.py
    logger.py
  main.py

scripts/
  rag_cli.py                          # Command-line interface for RAG
  fetch_and_save_reports.py           # PDF download utility

tests/
  agents/
    test_rag_agent.py
    test_dutch_agent.py
    test_french_agent.py
    test_validation_agent.py
    test_company_data_collector_agent.py
  core/
    test_query_processor.py
  integration/
    test_bilingual_validation.py
  conftest.py                         # Pytest configuration (adds repo root to sys.path)

banking_db_v2/                        # ChromaDB persistent store (created at runtime)
requirements.txt                      # Dependencies
pytest.ini                            # Pytest settings
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

This project includes a modular, multilingual RAG system for question answering over banking PDFs using only free/open-source tools:
- **Ollama** (local LLM, e.g., Qwen 2.5 7B)
- **ChromaDB** (local persistent vector database)
- **sentence-transformers** (multilingual embeddings)
- **PyPDF** (PDF parsing)

### Architecture

The RAG pipeline is composed of modular agents:

- **PDFIngestionAgent** (`src/agents/pdf_ingestion_agent.py`):  
  Parses PDFs, cleans text, performs semantic chunking, and applies quality filters.

- **RetrievalAgent** (`src/agents/retrieval_agent.py`):  
  Manages embeddings, indexes chunks in ChromaDB, performs hybrid search (vector + keyword), and supports multi-language retrieval.

- **GenerationAgent** (`src/agents/generation_agent.py`):  
  Builds prompts and generates answers using the local Ollama LLM.

- **RAGOrchestrator** (`src/agents/rag_orchestrator.py`):  
  Coordinates ingestion, retrieval, and generation; implements bilingual query expansion and cross-language consistency checks.

- **RAGAgent** (`src/agents/rag_agent.py`):  
  Public facade with a simple API: `index_documents()`, `answer_question()`, `collection_info()`.

### Setup

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Install and start Ollama:**
   - [Download Ollama](https://ollama.ai)
   - Pull a multilingual model:
     ```sh
     ollama pull qwen2.5:7b
     ollama serve
     ```
     (On Windows, start Ollama from Start Menu; it runs at `http://localhost:11434`)

3. **Verify Ollama is running:**
   ```sh
   curl http://localhost:11434/api/tags
   ```

### Indexing PDFs

1. Download or place PDFs in `data/fetched_reports/`.
2. Index all PDFs:
   ```sh
   python scripts/rag_cli.py index
   ```
   This processes PDFs, chunks them, generates embeddings, and persists to `banking_db_v2/`.

### Querying the RAG System

Ask questions in French, Dutch, or English:

```sh
python scripts/rag_cli.py query "What is the capital ratio for KBC Group?"
python scripts/rag_cli.py query "Quels sont les principaux risques pour Ageas?"
python scripts/rag_cli.py query "Wat is het winstpercentage van Colruyt?"
```

**Features:**
- Language auto-detection and optimal retrieval.
- Answers are sourced and cited with document name and page number.
- Configurable Ollama endpoint, model, and timeout (see CLI flags below).

### CLI Flags

```sh
python scripts/rag_cli.py query <question> [OPTIONS]
```

- `--ollama-url URL` (default: `http://localhost:11434`)  
  Custom Ollama endpoint.

- `--model MODEL` (default: `qwen2.5:7b`)  
  Override the LLM model.

- `--timeout SECONDS` (default: `30`)  
  Timeout for LLM requests (increase for slow hardware).

- `--no-llm`  
  Disable LLM and return only retrieved passages (useful for testing or when LLM is unavailable).

- `--n-results N` (default: `3`)  
  Number of passages to retrieve per language.

- `--bilingual-check`  
  Enable bilingual consistency checking. If a question is asked in one language, the system queries in both languages (FR ↔ NL, or EN → both) and flags numeric or content discrepancies.

### Examples

```sh
# Simple query with default settings
python scripts/rag_cli.py query "What is Proximus's revenue?"

# Disable LLM, return only retrieved passages
python scripts/rag_cli.py query "Revenue of Ageas" --no-llm

# Enable bilingual consistency checking
python scripts/rag_cli.py query "Quel est le bénéfice net?" --bilingual-check

# Custom Ollama endpoint and longer timeout
python scripts/rag_cli.py query "..."  --ollama-url http://192.168.1.100:11434 --timeout 60
```

### File Locations

- RAG facade: `src/agents/rag_agent.py`
- Agent modules: `src/agents/{ingestion,retrieval,generation}_agent.py`, `src/agents/rag_orchestrator.py`
- CLI script: `scripts/rag_cli.py`
- Downloaded PDFs: `data/fetched_reports/`
- Vector DB (persistent): `banking_db_v2/` (ChromaDB with SQLite backend)

### Troubleshooting

- **Ollama not reachable:**  
  Ensure Ollama is running (`ollama serve` on the terminal or start the app on Windows).  
  Verify the endpoint: `curl http://localhost:11434/api/tags`

- **Timeout errors:**  
  Increase `--timeout` if your hardware is slow or the model is not yet downloaded.  
  Or disable LLM with `--no-llm` to test retrieval without waiting for generation.

- **No results retrieved:**  
  Check that PDFs are in `data/fetched_reports/` and have been indexed: `python scripts/rag_cli.py index`

- **Import errors in tests:**  
  Tests use `tests/conftest.py` to add the repo root to `sys.path`.  
  Run pytest from the repo root: `pytest -q`

---

### Testing

Run the test suite:
```sh
pytest -q
```

The suite includes:
- Unit tests for RAG agents (`tests/agents/test_rag_agent.py`).
- Core module tests (`tests/core/test_query_processor.py`).
- Integration tests for bilingual validation (`tests/integration/test_bilingual_validation.py`).

Synthetic bilingual test data is in `src/data/sample_reports/synthetic_bilingual.json`.

---

**Contact:**  
Project maintained by Eduardo Brito.
