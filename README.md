# LLM-Extractify

**LLM-Extractify** is an end-to-end data ingestion and extraction pipeline that leverages large language models (LLMs) and vector search to transform unstructured web content into structured, queryable knowledge. With support for multiple LLM providers and Firecrawl integration, this project simplifies the process of scraping, chunking, embedding, and indexing data.

---

## 🚀 Features

- **Multi-Provider LLM Support:** OpenAI, Gemma, Mistral
- **Web Scraping:** Integrated with Firecrawl for dynamic and semantic extraction
- **Vector Storage:** Zilliz Milvus for efficient similarity search
- **Configurable Pipeline:** YAML-based prompt templates, environmental config
- **Streamlit UI:** Quick start interface for URL intake and retrieval testing

---

## 📦 Installation

1. **Install Poetry** (if not already installed):
   ```bash
   pip install poetry
   ```
2. **Clone the repo** and install dependencies:
   ```bash
   git clone https://github.com/your-org/llm-extractify.git
   cd llm-extractify
   poetry install
   ```
3. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate    # macOS/Linux
   # or
   .\.venv\\Scripts\\activate  # Windows PowerShell
   ```

---

## 🔑 Configuration

Create a `.env` file in the project root (use `.env.example` as a template) and populate the following keys:

```ini
OPENAI_API_KEY=
GEMMA_API_KEY=
MISTRAL_API_KEY=
FIRECRAWL_API_KEY=
ZILLIZ_AUTH_TOKEN=
ZILLIZ_CLOUD_URI=
```

> **Note (Windows):** If you encounter execution policy issues, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
> ```

---

## ⚙️ Usage

### 1. Streamlit UI

Launch the interactive frontend:
```bash
poetry run streamlit run frontend/streamlit_ui.py
```

### 2. CLI Scripts

- **Onboard URLs/files** (end-to-end processing):
  ```bash
  poetry run python scripts/onboard.py
  ```


## 🧪 Testing

Run unit and integration tests under the `tests/` folder:

```bash
# Single test
poetry run python tests/test_onboard.py

```

**Model Evaluations** (on `feat/model-evals` branch):
```bash
poetry run python tests/test_gpt_models.py
```

---

## 🌐 Sample URLs

Use these for quick testing or demos:

- https://foundation.wikimedia.org/wiki/Policy:Privacy_policy
- https://docs.llamaindex.ai/en/stable/examples/embeddings/huggingface/
- https://www.sas.com/en/events/sas-innovate/faq.html
- https://aiconference.com/faq/

---

## 🛠️ Troubleshooting

- **Missing API keys?** Ensure all keys are set in `.env`.
- **Zilliz cluster access:** Confirm `ZILLIZ_CLOUD_URI` and `ZILLIZ_AUTH_TOKEN` match your cloud cluster configuration.
- **Windows venv issues:** Use the PowerShell activation command above.


---

