# Minimal PDF Chatbot (RAG with LangChain, FAISS & Gemini)

A beginner-friendly, single-page PDF chatbot built with **Streamlit**, **LangChain**, **FAISS**, and **Google Gemini** (`gemini-2.5-flash` + `models/text-embedding-004`).

## Project Files

- `app.py` — Streamlit UI and complete RAG pipeline (PDF loading, chunking, FAISS indexing, retrieval, and Q&A).
- `requirements.txt` — Python dependencies.
- `.env.example` — Template for your `GOOGLE_API_KEY`.
- `README.md` — Setup and usage instructions.

## 3-Step Setup & Run

### 1. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure your Google API key

Copy `.env.example` to `.env` and add your Gemini API key:

```bash
cp .env.example .env
```

Edit `.env`:

```env
GOOGLE_API_KEY=your_actual_api_key_here
```

*(Alternatively, you can paste your API key directly into the Streamlit sidebar when running the app.)*

### 3. Run the chatbot

```bash
streamlit run app.py
```

Open the local URL shown in your terminal, upload a PDF in the sidebar, click **Process PDF**, and start asking questions about your document.
