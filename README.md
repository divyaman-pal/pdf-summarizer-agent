# 📄 PDF Summarization Agent

> Upload one or more PDFs and get AI-powered structured summaries instantly.  
> Built with **Streamlit + LangChain + OpenAI GPT**.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

---

## ✨ Features

- 📥 Drag & drop **multiple PDFs** simultaneously
- 🧠 **Map-reduce LLM pipeline** — every page is covered
- 📊 Extracts **text + tables** from each page
- 📋 Structured summaries with fixed sections
- 🌐 **Cross-document unified analysis** for multi-PDF sessions
- ⬇️ Download summaries as **Markdown + TXT + ZIP**
- 🔐 API key handled securely via Streamlit Secrets

---

## 🚀 Deploy to Streamlit Cloud (Step-by-Step)

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit — PDF Summarization Agent"
git remote add origin https://github.com/YOUR_USERNAME/pdf-summarizer.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Click **"New app"**
3. Select your GitHub repo
4. Set **Main file path** → `app.py`
5. Click **"Deploy"**

### Step 3 — Add OpenAI API Key (Secrets)

1. In Streamlit Cloud → your app → **"⋮" menu → Settings → Secrets**
2. Add:
   ```toml
   OPENAI_API_KEY = "sk-your-new-key-here"
   ```
3. Click **Save** → app auto-restarts

> ⚠️ **Never** paste your API key into GitHub code or a public chat.

---

## 🗂 Project Structure

```
pdf_summarizer_streamlit/
├── app.py                        # Streamlit UI (main entry point)
├── requirements.txt              # Python dependencies
├── .gitignore                    # Excludes secrets & cache
├── README.md
│
├── core/
│   ├── pdf_extractor.py          # pdfplumber + PyPDF2 extraction
│   ├── chunker.py                # Token-aware chunking (tiktoken)
│   └── summarizer.py             # LLM map-reduce pipeline
│
└── .streamlit/
    ├── config.toml               # Theme & server config
    └── secrets.toml.example      # Template (do NOT commit real secrets)
```

---

## 🧠 How It Works

```
PDF Upload(s)
     │
     ▼
┌──────────────────────────────┐
│ 1. Extract (pdfplumber)      │  text + tables per page
└──────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│ 2. Chunk (tiktoken)          │  token-safe page groups
└──────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│ 3a. Map — Chunk Summaries    │  LLM summarizes each chunk
└──────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│ 3b. Reduce — Doc Summary     │  merges chunks → structured doc summary
└──────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│ 3c. Unify (multi-PDF only)   │  cross-doc themes + comparative analysis
└──────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│ 4. Display + Download        │  tabs per doc, ZIP download
└──────────────────────────────┘
```

---

## 🔒 Security

- API keys are **never stored** — session-scoped only
- Use Streamlit Secrets for production deployments
- `.gitignore` excludes `secrets.toml` and `.env` by default
