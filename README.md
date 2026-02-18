# CrossGuard: Belgian Bilingual Corporate Agent

A RAG system for analyzing Belgian banking reports with **crosslingual consistency guardrails**. It retrieves information from French, Dutch, and English annual reports, cross-validates across languages, and flags discrepancies using token-level explanations.

## What Makes This Different

Belgian banks publish annual reports in both French and Dutch (and sometimes English). This system:

1. **Retrieves information** from multiple language versions simultaneously
2. **Swappable retrieval backends** -- ChromaDB (dense) or ColBERT (late interaction) via a shared Retriever protocol
3. **Cross-validates numeric data** for consistency across languages using MaxSimE token alignment
4. **Flags discrepancies** when French and Dutch reports contain different figures
5. **Explains results** with token-level heatmaps showing which terms matched across languages

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) running locally (for LLM generation; optional for retrieval-only mode)
- At least 8GB RAM

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd belgian_bilingual_corporate_agent

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Pull the LLM model (optional -- only needed for answer generation)
ollama pull qwen2.5:7b
```

### Index the Corpus

Before querying, you need to index the PDF reports. Place your PDFs in `data/fetched_reports/` using the naming convention `BankName_Language.pdf` (e.g. `KBC_Group_FR.pdf`).

```bash
# Index with ChromaDB (default, faster)
python scripts/index_corpus.py

# Index with ColBERT (late interaction, better ranking)
python scripts/index_corpus.py --retriever colbert
```

You can also change the default retriever in `config.yaml`:

```yaml
retriever:
  type: "chroma"   # or "colbert"
```

### Launch the Web UI

```bash
python app.py
```

Open `http://localhost:7860` in your browser. The interface provides:

- **Chat**: Ask questions about the banking reports in French, Dutch, or English
- **Settings**: Toggle bilingual validation, adjust result count, enable retrieval-only mode
- **Sources**: Expandable cards showing each retrieved passage with bank, page, and language
- **Token Alignment**: Consistency visualization when bilingual mode is enabled

### Command-Line Usage

```bash
# Ask a question (with LLM answer generation)
python scripts/rag_cli.py query "What is KBC's capital ratio?"

# Retrieval only (no LLM, faster)
python scripts/rag_cli.py query "What is the net profit?" --no-llm

# Bilingual consistency check
python scripts/rag_cli.py query "Quel est le ratio de capital?" --bilingual-check

# Re-index documents
python scripts/rag_cli.py index

# Check collection info
python scripts/rag_cli.py info
```

## Testing

### Integration Tests

```bash
# All tests (requires Ollama running)
pytest tests/integration/test_end_to_end.py -v

# Retrieval-only tests (faster, no LLM needed)
pytest tests/integration/test_end_to_end.py::TestRetrievalOnly -v

# Setup verification
pytest tests/integration/test_end_to_end.py::TestSetup -v
```

### Consistency Tests

```bash
pytest tests/test_consistency.py -v
```

## Evaluation

The project includes a benchmark of 20 bilingual facts and scripts to measure retrieval performance.

### Run Evaluation

```bash
# Evaluate with ChromaDB (default retriever)
python scripts/run_evaluation.py --no-llm

# Evaluate with ColBERT
python scripts/run_evaluation.py --retriever colbert --no-llm

# Custom output directory
python scripts/run_evaluation.py --retriever colbert --no-llm --output-dir results_colbert

# Evaluate a subset of facts
python scripts/run_evaluation.py --facts 10 --no-llm
```

Results are saved to `results/results.json` and `results/summary.md` (or the directory specified by `--output-dir`).

### Validate the Benchmark

```bash
python scripts/validate_benchmark.py
```

This checks the structure, field completeness, and distribution of `tests/eval_benchmark.json`.

### Benchmark Details

`tests/eval_benchmark.json` contains 20 atomic facts from KBC Group (12) and BNP Paribas Fortis (8) annual reports. Each fact includes:

- `question_fr` / `question_nl`: Natural questions in French and Dutch
- `ground_truth`: Verified answer (number, percentage, or amount)
- `source_bank`, `source_page_fr`, `source_page_nl`: Exact provenance
- `category`: One of capital_ratio, profitability, balance_sheet, workforce, fee_income, digital, dividend, liquidity
- `difficulty`: easy, medium, hard, or adversarial

### Current Retrieval Results

| Metric | ChromaDB | ColBERT |
|--------|----------|---------|
| Recall@5 (FR) | 20.0% | 15.0% |
| Recall@5 (NL) | 15.0% | 10.0% |
| MRR (FR) | 0.108 | 0.125 |
| MRR (NL) | 0.058 | 0.075 |

ChromaDB retrieves more relevant documents overall; ColBERT ranks hits higher when it finds them. See `results/comparison_report.md` for the full analysis.

## Architecture

```
belgian_bilingual_corporate_agent/
├── app.py                          # Gradio web UI
├── config.yaml                     # Central configuration
├── src/
│   ├── retriever.py                # Retriever protocol + ChromaRetriever
│   ├── colbert_retriever.py        # ColBERT retriever (FAISS + MaxSim)
│   ├── rag_service.py              # Main RAG service
│   ├── bilingual_validator.py      # Cross-language validation
│   ├── consistency.py              # MaxSimE-based consistency scoring
│   ├── visualization.py            # Gradio visualization helpers
│   └── utils/
│       └── pdf_parser.py           # PDF chunking
├── scripts/
│   ├── rag_cli.py                  # Command-line interface
│   ├── index_corpus.py             # Corpus indexing (ChromaDB or ColBERT)
│   ├── run_evaluation.py           # Evaluation pipeline
│   └── validate_benchmark.py       # Benchmark structure checker
├── tests/
│   ├── eval_benchmark.json         # 20-fact bilingual benchmark
│   ├── test_consistency.py         # Consistency scorer tests
│   └── integration/
│       ├── test_end_to_end.py      # Integration tests
│       └── test_cases.py           # Q&A test pairs
├── data/
│   └── fetched_reports/            # PDF storage (BankName_Language.pdf)
├── banking_db_v2/                  # ChromaDB persistent store
├── colbert_index/                  # ColBERT/FAISS index
├── results/                        # Evaluation output
└── requirements.txt
```

### Components

1. **Retriever Protocol** (`src/retriever.py`): Abstract interface allowing backend swaps between ChromaDB and ColBERT without changing application code.

2. **ChromaRetriever**: Dense retrieval using `paraphrase-multilingual-mpnet-base-v2` embeddings stored in ChromaDB.

3. **ColBERTRetriever** (`src/colbert_retriever.py`): Late interaction retrieval using `answerdotai/answerai-colbert-small-v1`. Mean-pooled embeddings are indexed in FAISS for candidate retrieval, then re-ranked with full MaxSim scoring. Returns per-token embeddings for MaxSimE explanations.

4. **RAGService** (`src/rag_service.py`): Orchestrates retrieval, bilingual validation, and LLM answer generation via Ollama.

5. **ConsistencyScorer** (`src/consistency.py`): Compares FR/NL token alignments via MaxSimE to detect retrieval inconsistencies. Scoring formula: `0.6 * alignment_overlap + 0.4 * (1 - score_divergence)`.

6. **BilingualValidator** (`src/bilingual_validator.py`): Language detection, domain-specific term translation, numeric consistency checking, and token alignment consistency.

### Technology Stack

| Component | Tool |
|-----------|------|
| Web UI | [Gradio](https://gradio.app/) |
| Dense retrieval | [ChromaDB](https://www.trychroma.com/) + [SentenceTransformers](https://www.sbert.net/) |
| Late interaction | [ColBERT](https://github.com/stanford-futuredata/ColBERT) + [FAISS](https://github.com/facebookresearch/faiss) |
| Explainability | [MaxSimE](https://github.com/ebritoc/MaxSimE) |
| LLM | [Ollama](https://ollama.ai/) (qwen2.5:7b) |
| PDF parsing | [pypdf](https://pypdf.readthedocs.io/) |
| Config | YAML (`config.yaml`) |

## Configuration

All settings are in `config.yaml`. Defaults work out of the box.

```yaml
retriever:
  type: "chroma"                    # "chroma" or "colbert"
  chroma_path: "./banking_db_v2"
  embedding_model: "paraphrase-multilingual-mpnet-base-v2"
  colbert_model: "answerdotai/answerai-colbert-small-v1"

llm:
  ollama_url: "http://localhost:11434"
  model_name: "qwen2.5:7b"

bilingual:
  enabled: true
  consistency_threshold: 0.5

chunking:
  chunk_size: 1000
  chunk_overlap: 100
```

## Data Sources

Currently supports:
- **BNP Paribas Fortis** -- French and Dutch annual report (2024)
- **KBC Group** -- French, Dutch, and English annual report (2024)

Reports are stored in `data/fetched_reports/` as PDFs.

## Troubleshooting

**Ollama not reachable**: Ensure Ollama is running (`ollama serve`) and verify with `curl http://localhost:11434/api/tags`.

**No results retrieved**: Check that PDFs are in `data/fetched_reports/`, run `python scripts/index_corpus.py`, and verify with `python scripts/rag_cli.py info`.

**Slow first query**: The embedding model loads on first use. Use `--no-llm` or retrieval-only mode in the UI for faster testing.

**ColBERT indexing slow**: ColBERT encodes ~10k chunks in ~20 min on CPU. Use a GPU-enabled machine for faster indexing, or stick with ChromaDB for development.

## License

[Add your license here]

---

**Maintainer**: Eduardo Brito
