# Belgian Bilingual Corporate Agent

A simplified RAG (Retrieval-Augmented Generation) system for analyzing Belgian banking reports with bilingual validation capabilities.

## What Makes This Different

Belgian banks publish annual reports in both French and Dutch (and sometimes English). This system:
1. **Retrieves information** from multiple language versions simultaneously
2. **Cross-validates numeric data** for consistency across languages
3. **Flags discrepancies** when French and Dutch reports contain different figures
4. **Uses multilingual embeddings** to enable cross-language queries

## Features

- 📄 **PDF Ingestion**: Automatically chunks and indexes banking reports
- 🔍 **Semantic Search**: Find relevant passages using multilingual vector embeddings
- 🤖 **Local LLM Integration**: Generate answers using Ollama (privacy-first)
- 🌍 **Bilingual Validation**: Cross-reference information across language versions
- ✅ **Tested**: Integration tests with real Q&A pairs from actual reports

## Quick Start

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai/) running locally
- At least 8GB RAM (for qwen2.5:7b model)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd belgian_bilingual_corporate_agent

# Install dependencies
pip install -r requirements.txt

# Start Ollama (in separate terminal)
ollama pull qwen2.5:7b
ollama serve
```

### Usage

```bash
# 1. Index documents (one-time setup)
python scripts/rag_cli.py index

# 2. Ask questions
python scripts/rag_cli.py query "What is KBC's capital ratio?"

# 3. Enable bilingual validation
python scripts/rag_cli.py query "Quel est le ratio de capital?" --bilingual-check

# 4. Retrieve without LLM (faster, for testing)
python scripts/rag_cli.py query "What is the net profit?" --no-llm

# 5. Check collection info
python scripts/rag_cli.py info
```

## Architecture

### Simple and Transparent

```
src/
├── rag_service.py              # Main RAG implementation (~360 lines)
├── bilingual_validator.py      # Cross-language validation (~230 lines)
├── utils/
│   └── pdf_parser.py           # PDF chunking (~110 lines)
└── agents/
    └── company_data_collector_agent.py  # PDF downloading utility
```

**Total core code**: ~700 lines (down from 1,500+ in previous version)

### Components

1. **RAGService**: Single-file RAG implementation
   - Document indexing using ChromaDB
   - Vector search with SentenceTransformer embeddings
   - LLM generation via Ollama
   - Optional bilingual validation

2. **BilingualValidator**: Language handling
   - Language detection using `langdetect`
   - Domain-specific term translation
   - Numeric consistency checking

3. **PDF Parser**: Simple text extraction
   - PyPDF-based extraction
   - Sentence-boundary chunking
   - Metadata preservation

### Technology Stack

- **Vector Store**: [ChromaDB](https://www.trychroma.com/) - Persistent vector database
- **Embeddings**: [SentenceTransformers](https://www.sbert.net/) - `paraphrase-multilingual-mpnet-base-v2`
- **LLM**: [Ollama](https://ollama.ai/) - Local model inference (qwen2.5:7b)
- **PDF Parsing**: [pypdf](https://pypdf.readthedocs.io/) - Pure Python PDF reader

## Testing

### Run Integration Tests

```bash
# All tests (requires Ollama running)
pytest tests/integration/test_end_to_end.py -v

# Only retrieval tests (faster, no LLM)
pytest tests/integration/test_end_to_end.py::TestRetrievalOnly -v

# Setup verification
pytest tests/integration/test_end_to_end.py::TestSetup -v
```

### Test Results

Based on KBC Group and BNP Paribas Fortis 2024 annual reports:

- ✅ **18/21 tests passing (86%)**
- ✅ All retrieval tests passing
- ✅ LLM generation working
- ✅ Bilingual consistency validation working
- ✅ Source attribution accurate

## Examples

### Basic Query

```bash
$ python scripts/rag_cli.py query "How many employees does KBC have?"
```

```
Question: How many employees does KBC have?
======================================================================

Retrieving 5 relevant passages...
[OK] Retrieved 5 passages

Generating answer with LLM...

======================================================================
ANSWER:
======================================================================
According to the provided context from KBC Group's 2024 annual report,
KBC Group had an average of 38,074 full-time equivalent employees in 2024.
This figure includes employees across KBC Bank, KBC Insurance, and other
group entities.

Source: KBC Group, Page 288, EN
======================================================================
Retrieved 5 source passages

Sources:
  1. KBC Group (Page 288, EN)
  2. KBC Group (Page 289, EN)
  ...
```

### Bilingual Query with Validation

```bash
$ python scripts/rag_cli.py query "Quel est le bénéfice net?" --bilingual-check
```

```
Question: Quel est le bénéfice net?
======================================================================

Detected language: FR
Expanded to 3 language variants

Performing multi-language retrieval...
Consistency check: ok (confidence: 0.80)
[OK] Retrieved 5 passages

Generating answer with LLM...

======================================================================
ANSWER:
======================================================================
Le bénéfice net de KBC Groupe pour 2024 s'élève à 3.415 millions d'euros...
======================================================================
Retrieved 5 source passages
Detected language: FR
Consistency: ok (confidence: 0.80) - Found 2 common numeric values across languages

Sources:
  1. KBC Group (Page 347, FR)
  2. KBC Group (Page 344, NL)
  ...
```

## Configuration

All configuration is optional. Defaults work out of the box.

```python
from rag_service import RAGService

# Custom configuration
config = {
    'chroma_path': './my_custom_db',
    'embedding_model': 'paraphrase-multilingual-mpnet-base-v2',
    'ollama_url': 'http://localhost:11434',
    'model_name': 'qwen2.5:7b',
    'ollama_timeout': 180
}

rag = RAGService(config)
```

## Project Structure

```
belgian_bilingual_corporate_agent/
├── src/
│   ├── rag_service.py              # Main RAG service
│   ├── bilingual_validator.py      # Bilingual validation
│   └── utils/
│       └── pdf_parser.py           # PDF chunking
│
├── scripts/
│   ├── rag_cli.py                  # Command-line interface
│   └── fetch_and_save_reports.py  # Download PDFs
│
├── tests/
│   └── integration/
│       ├── test_end_to_end.py      # Integration tests
│       ├── test_cases.py           # Real Q&A pairs
│       └── test_cases_template.py  # Template for adding tests
│
├── data/
│   └── fetched_reports/            # PDF storage
│
├── banking_db_v2/                  # ChromaDB persistent store
├── requirements.txt
└── README.md
```

## Data Sources

Currently supports:
- **BNP Paribas Fortis** (French & Dutch versions)
- **KBC Group** (French, Dutch & English versions)

Reports are 2024 annual reports downloaded via `scripts/fetch_and_save_reports.py`.

## Development

### Adding New Test Cases

Edit `tests/integration/test_cases.py`:

```python
TEST_CASES = [
    {
        "question": "Your question here?",
        "expected_answer_contains": ["keyword1", "keyword2"],
        "expected_sources": ["Bank_Name"],
        "language": "en",  # or "fr", "nl"
        "category": "factual_extraction"
    },
    # ... more test cases
]
```

### Extending to Other Banks

1. Add PDFs to `data/fetched_reports/` with format: `BankName_Language.pdf`
2. Run indexing: `python scripts/rag_cli.py index`
3. Add term mappings to `bilingual_validator.py` if needed

## Performance

- **Indexing**: ~10,000 chunks in ~2 minutes
- **Retrieval**: ~1 second per query (vector search)
- **Generation**: ~10-30 seconds (depends on context length and model)
- **Memory**: ~2GB for embeddings, ~4GB for LLM

## Limitations

- **Language Detection**: Works best with >20 words
- **Term Translation**: Limited to hardcoded financial terms
- **Numeric Consistency**: Simple regex-based, may miss formatted differences
- **Context Window**: Limited by Ollama model capacity (~4K tokens)

## Future Improvements

- [ ] Add more banks (ING, Belfius, Argenta)
- [ ] Implement proper BM25 hybrid search
- [ ] Support for multi-year comparisons
- [ ] Web UI for easier interaction
- [ ] Export to structured data (JSON, CSV)

## Contributing

This project was created as a demonstration of aggressive simplification in RAG systems. The codebase was reduced from ~1,500 lines with complex agent abstractions to ~700 lines with clear, maintainable code.

**Philosophy**:
- Prefer standard libraries over custom implementations
- Keep abstractions minimal and purposeful
- Every line of code should have a clear reason to exist

## License

[Add your license here]

## Acknowledgments

- Built on [ChromaDB](https://www.trychroma.com/), [Ollama](https://ollama.ai/), and [SentenceTransformers](https://www.sbert.net/)
- Inspired by the need for transparent, maintainable RAG systems
- Test data from public annual reports of Belgian banks

---

**Questions or Issues?** Open an issue on GitHub or contact [your contact info]
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
python scripts/rag_cli.py query "Quels sont les principaux risques pour BNP Paribas Fortis?"
python scripts/rag_cli.py query "Wat is het winstpercentage van KBC?"
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
python scripts/rag_cli.py query "What is KBC's revenue?"

# Disable LLM, return only retrieved passages
python scripts/rag_cli.py query "Revenue of BNP Paribas Fortis" --no-llm

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
