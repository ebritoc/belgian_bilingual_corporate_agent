# Belgian Bilingual Corporate Agent

A RAG (Retrieval-Augmented Generation) system for analyzing Belgian banking reports with bilingual validation capabilities.

## What Makes This Different

Belgian banks publish annual reports in both French and Dutch (and sometimes English). This system:
1. **Retrieves information** from multiple language versions simultaneously
2. **Cross-validates numeric data** for consistency across languages
3. **Flags discrepancies** when French and Dutch reports contain different figures
4. **Uses multilingual embeddings** to enable cross-language queries

## Features

- **Web UI**: User-friendly Gradio chat interface
- **PDF Ingestion**: Automatically chunks and indexes banking reports
- **Semantic Search**: Find relevant passages using multilingual vector embeddings
- **Local LLM Integration**: Generate answers using Ollama (privacy-first)
- **Bilingual Validation**: Cross-reference information across language versions
- **Tested**: Integration tests with real Q&A pairs from actual reports

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

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate  # Windows
# or: source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Start Ollama (in separate terminal)
ollama pull qwen2.5:7b
ollama serve
```

### Running the Web UI

```bash
python app.py
```

This will start a Gradio web interface. Open your browser to the URL shown (usually http://localhost:7860).

**Performance Note**: The first query may take 30-60 seconds as the embedding model loads. Subsequent queries with LLM generation typically take 10-30 seconds depending on your hardware. For faster testing, use the "Retrieval Only" option which skips LLM generation.

### Command-Line Usage

```bash
# 1. Index documents (one-time setup, if not already done)
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

## Web UI Features

The Gradio interface provides:

- **Chat Interface**: Conversational Q&A with the banking reports
- **Settings Panel**:
  - Enable/disable bilingual validation
  - Adjust number of sources retrieved (1-10)
  - Retrieval-only mode (skip LLM for faster results)
- **Response Metadata**: Shows detected language and consistency status when bilingual mode is enabled
- **Retrieved Sources**: Expandable cards showing source documents with bank name, page number, and language

## Architecture

### Simple and Transparent

```
belgian_bilingual_corporate_agent/
├── app.py                          # Gradio web UI
├── src/
│   ├── rag_service.py              # Main RAG implementation (~360 lines)
│   ├── bilingual_validator.py      # Cross-language validation (~230 lines)
│   └── utils/
│       └── pdf_parser.py           # PDF chunking (~110 lines)
├── scripts/
│   ├── rag_cli.py                  # Command-line interface
│   └── fetch_and_save_reports.py   # Download PDFs
├── tests/
│   ├── eval_benchmark.json         # Bilingual evaluation benchmark (20 facts)
│   └── integration/
│       ├── test_end_to_end.py      # Integration tests
│       └── test_cases.py           # Real Q&A pairs
├── data/
│   └── fetched_reports/            # PDF storage
├── banking_db_v2/                  # ChromaDB persistent store
└── requirements.txt
```

**Total core code**: ~700 lines

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

- **Web UI**: [Gradio](https://gradio.app/) - Modern web interface
- **Vector Store**: [ChromaDB](https://www.trychroma.com/) - Persistent vector database
- **Embeddings**: [SentenceTransformers](https://www.sbert.net/) - `paraphrase-multilingual-mpnet-base-v2`
- **LLM**: [Ollama](https://ollama.ai/) - Local model inference (qwen2.5:7b)
- **PDF Parsing**: [pypdf](https://pypdf.readthedocs.io/) - Pure Python PDF reader

## Performance Expectations

| Operation | Time | Notes |
|-----------|------|-------|
| First query (cold start) | 30-60s | Embedding model loading |
| Retrieval only | 1-2s | Vector search |
| LLM generation | 10-30s | Depends on hardware and context length |
| Indexing | ~2 min | For ~10,000 chunks |

**Hardware Requirements**:
- ~2GB RAM for embeddings
- ~4GB RAM for LLM (qwen2.5:7b)
- SSD recommended for faster model loading

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

- 18/21 tests passing (86%)
- All retrieval tests passing
- LLM generation working
- Bilingual consistency validation working
- Source attribution accurate

### Bilingual Evaluation Benchmark

The file `tests/eval_benchmark.json` contains 20 atomic facts extracted from the annual reports, each with a bilingual question pair (French + Dutch) and a verified ground-truth answer. This benchmark is designed for systematic evaluation of the RAG system's accuracy and cross-language retrieval quality.

**Coverage:**
- **12 KBC Group** and **8 BNP Paribas Fortis** facts
- **8 categories**: capital ratios, profitability, balance sheet, workforce, fee income, digital, dividends, liquidity
- Questions use **Belgian corporate language** nuances (e.g. FR: "exercice", "effectifs", "fonds propres"; NL: "boekjaar", "personeelsbestand", "eigen vermogen")

**Each fact contains:**
- `question_fr` / `question_nl`: Natural language questions in French and Dutch
- `ground_truth`: The verified answer (specific number, percentage, or amount)
- `source_bank`, `source_page`, `source_language`: Exact provenance in the PDF
- `category` and `notes`: Classification and additional context

**Example:**
```json
{
  "id": 1,
  "question_fr": "Quel est le ratio Common Equity Tier 1 transitionnel du Groupe KBC fin 2024 ?",
  "question_nl": "Wat is de transitionele Common Equity Tier 1-ratio van KBC Groep eind 2024?",
  "ground_truth": "13,9%",
  "source_bank": "KBC Group",
  "source_page": 101,
  "category": "capital_ratio"
}
```

This benchmark can be used to measure retrieval recall, answer accuracy, and bilingual consistency across model or configuration changes.

## Data Sources

Currently supports:
- **BNP Paribas Fortis** (French & Dutch versions)
- **KBC Group** (French, Dutch & English versions)

Reports are 2024 annual reports downloaded via `scripts/fetch_and_save_reports.py`.

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

## CLI Flags

```bash
python scripts/rag_cli.py query <question> [OPTIONS]
```

- `--ollama-url URL` - Custom Ollama endpoint (default: `http://localhost:11434`)
- `--model MODEL` - Override the LLM model (default: `qwen2.5:7b`)
- `--timeout SECONDS` - Timeout for LLM requests (default: `30`)
- `--no-llm` - Disable LLM, return only retrieved passages
- `--n-results N` - Number of passages to retrieve (default: `5`)
- `--bilingual-check` - Enable bilingual consistency checking

## Troubleshooting

**Ollama not reachable:**
- Ensure Ollama is running (`ollama serve` on terminal or start the app on Windows)
- Verify the endpoint: `curl http://localhost:11434/api/tags`

**Slow responses:**
- First query takes longer due to model loading
- Use `--no-llm` or "Retrieval Only" mode for faster testing
- Increase `--timeout` if your hardware is slower

**No results retrieved:**
- Check that PDFs are in `data/fetched_reports/`
- Run indexing: `python scripts/rag_cli.py index`
- Verify with: `python scripts/rag_cli.py info`

**Port already in use (Gradio):**
- The app will automatically find an available port
- Or manually specify: `GRADIO_SERVER_PORT=7861 python app.py`

## TODO

- [ ] Add visualization of retrieved chunks in the Gradio app (show highlighted text passages)
- [ ] Add more banks (ING, Belfius, Argenta)
- [ ] Implement proper BM25 hybrid search
- [ ] Support for multi-year comparisons
- [ ] Export to structured data (JSON, CSV)
- [ ] Add streaming responses for better UX

## Contributing

This project demonstrates aggressive simplification in RAG systems. The codebase was reduced from ~1,500 lines with complex agent abstractions to ~700 lines with clear, maintainable code.

**Philosophy**:
- Prefer standard libraries over custom implementations
- Keep abstractions minimal and purposeful
- Every line of code should have a clear reason to exist

## License

[Add your license here]

## Acknowledgments

- Built on [Gradio](https://gradio.app/), [ChromaDB](https://www.trychroma.com/), [Ollama](https://ollama.ai/), and [SentenceTransformers](https://www.sbert.net/)
- Inspired by the need for transparent, maintainable RAG systems
- Test data from public annual reports of Belgian banks

---

**Maintainer**: Eduardo Brito
