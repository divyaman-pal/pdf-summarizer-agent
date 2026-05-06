"""
app.py — PDF Summarization Agent · Streamlit Web App
──────────────────────────────────────────────────────
Features:
  • Upload 1–N PDF files via drag-and-drop
  • Secure API key entry (sidebar, never stored)
  • Real-time progress tracking per document
  • Structured markdown summaries with expandable sections
  • Per-document + unified multi-doc summary tabs
  • One-click download of all summaries (.md / .txt)
  • Mobile-responsive layout
"""

import io
import os
import re
import zipfile
from datetime import datetime
from typing import List, Optional

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="PDF Summarization Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

from core.pdf_extractor import PDFExtractor, PDFDocument
from core.chunker import DocumentChunker
from core.summarizer import PDFSummarizer, DocumentSummary, MultiDocSummary


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #6C63FF 0%, #3ECFCF 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(108,99,255,0.3);
}
.hero h1 { color: #fff; font-size: 2.4rem; font-weight: 800; margin: 0; }
.hero p  { color: rgba(255,255,255,0.88); font-size: 1.05rem; margin: 0.5rem 0 0; }

/* ── Stat cards ── */
.stat-row { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
.stat-card {
    flex: 1; min-width: 120px;
    background: #1A1D27;
    border: 1px solid #2D3142;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.stat-card .val { font-size: 1.8rem; font-weight: 700; color: #6C63FF; }
.stat-card .lbl { font-size: 0.78rem; color: #888; margin-top: 0.2rem; }

/* ── File pill ── */
.file-pill {
    display: inline-block;
    background: #1A1D27;
    border: 1px solid #6C63FF55;
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.82rem;
    color: #CACADE;
    margin: 0.2rem;
}

/* ── Section badge ── */
.badge {
    display: inline-block;
    background: linear-gradient(90deg,#6C63FF,#3ECFCF);
    border-radius: 8px;
    padding: 0.15rem 0.7rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: #fff;
    margin-right: 0.5rem;
}

/* ── Summary card ── */
.summary-card {
    background: #1A1D27;
    border: 1px solid #2D3142;
    border-radius: 14px;
    padding: 1.5rem;
    margin: 0.8rem 0;
}

/* ── Progress label ── */
.prog-label { font-size: 0.88rem; color: #aaa; margin-bottom: 0.3rem; }

/* ── Warning box ── */
.warn-box {
    background: #2a1a1a;
    border-left: 4px solid #ff4b4b;
    border-radius: 8px;
    padding: 0.9rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
}

/* ── Success box ── */
.success-box {
    background: #162a1e;
    border-left: 4px solid #21c55d;
    border-radius: 8px;
    padding: 0.9rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #111320; }
[data-testid="stSidebar"] hr { border-color: #2D3142; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0 !important;
    font-weight: 600;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s;
}
.stDownloadButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def strip_md(text: str) -> str:
    """Minimal markdown → plain text."""
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
    text = re.sub(r"^\|.*\|$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_doc_markdown(ds: DocumentSummary) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    meta = "\n".join(f"- **{k}**: {v}" for k, v in (ds.metadata or {}).items()) or "_None_"
    return f"""# 📄 Summary: {ds.file_name}
> Generated: {ts} | Pages: {ds.total_pages} | Words: {ds.total_words:,}

## Document Metadata
{meta}

---

{ds.full_summary}
"""


def build_multi_markdown(ms: MultiDocSummary) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    doc_list = "\n".join(
        f"- **{ds.file_name}** — {ds.total_pages} pages, {ds.total_words:,} words"
        for ds in ms.document_summaries
    )
    doc_sections = "\n\n---\n\n".join(
        f"## 📄 {ds.file_name}\n\n{ds.full_summary}"
        for ds in ms.document_summaries
    )
    return f"""# 📚 Multi-Document Summary Report
> Generated: {ts} | Documents: {len(ms.document_summaries)}

## Documents Analyzed
{doc_list}

---

# 🌐 Unified Cross-Document Analysis

{ms.master_summary}

---

# 📄 Individual Document Summaries

{doc_sections}
"""


def make_zip(doc_summaries: List[DocumentSummary], multi: Optional[MultiDocSummary]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for ds in doc_summaries:
            stem = re.sub(r"[^\w\-]", "_", ds.file_name.replace(".pdf", ""))
            md = build_doc_markdown(ds)
            zf.writestr(f"{stem}.md", md)
            zf.writestr(f"{stem}.txt", strip_md(md))
        if multi and len(doc_summaries) > 1:
            md = build_multi_markdown(multi)
            zf.writestr("MULTI_DOC_SUMMARY.md", md)
            zf.writestr("MULTI_DOC_SUMMARY.txt", strip_md(md))
    buf.seek(0)
    return buf.read()


def get_api_key() -> str:
    """Resolve API key: Streamlit secrets → sidebar input."""
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return st.session_state.get("api_key", "")


def validate_key(key: str) -> bool:
    return bool(key and key.startswith("sk-") and len(key) > 20)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")

        # API key input
        has_secret = False
        try:
            has_secret = bool(st.secrets.get("OPENAI_API_KEY"))
        except Exception:
            pass

        if has_secret:
            st.markdown('<div class="success-box">🔐 API key loaded from Streamlit Secrets</div>',
                        unsafe_allow_html=True)
        else:
            key = st.text_input(
                "🔑 OpenAI API Key",
                type="password",
                placeholder="sk-...",
                help="Your key is never stored — used only for this session.",
                value=st.session_state.get("api_key", ""),
            )
            st.session_state["api_key"] = key
            if key and not validate_key(key):
                st.warning("Key format looks incorrect. Should start with `sk-`")
            elif key:
                st.success("✅ Key accepted")

        st.markdown("---")
        st.markdown("## 🧠 Model Settings")

        model = st.selectbox(
            "OpenAI Model",
            ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=0,
            help="gpt-4o gives the best summaries. gpt-3.5-turbo is faster & cheaper.",
        )
        st.session_state["model"] = model

        max_tokens = st.slider(
            "Max Tokens per Chunk",
            min_value=500, max_value=6000,
            value=3000, step=500,
            help="Larger chunks = more context per LLM call but slower.",
        )
        st.session_state["max_tokens"] = max_tokens

        st.markdown("---")
        st.markdown("## ℹ️ About")
        st.markdown("""
**PDF Summarization Agent**

Extracts and summarizes PDFs using:
- 📥 `pdfplumber` + `PyPDF2`
- ✂️ Token-aware chunking
- 🧠 OpenAI GPT (map-reduce)
- 📋 Structured markdown output

Supports **1 to N** PDFs simultaneously.
        """)
        st.markdown("---")
        st.caption("⚠️ Never share your API key in a public chat.")


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    render_sidebar()

    # ── Hero ──────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <h1>📄 PDF Summarization Agent</h1>
        <p>Upload one or more PDFs · Get AI-powered structured summaries instantly</p>
    </div>
    """, unsafe_allow_html=True)

    # ── File uploader ─────────────────────────────────────────
    st.markdown("### 📂 Upload PDF Files")
    uploaded_files = st.file_uploader(
        "Drag & drop or click to browse",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Upload one or more PDF files. Max 200 MB each.",
    )

    if not uploaded_files:
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem; color:#555; border:2px dashed #2D3142; border-radius:16px; margin-top:1rem;">
            <div style="font-size:3rem;">📄</div>
            <div style="font-size:1.1rem; margin-top:0.5rem;">Drop your PDFs here to get started</div>
            <div style="font-size:0.85rem; margin-top:0.3rem; color:#444;">
                Supports: reports · research papers · contracts · manuals · any PDF
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Show uploaded files
    st.markdown("**Uploaded files:**")
    pills = "".join(
        f'<span class="file-pill">📄 {f.name} ({f.size/1024:.0f} KB)</span>'
        for f in uploaded_files
    )
    st.markdown(pills, unsafe_allow_html=True)

    # ── Validate API key ──────────────────────────────────────
    api_key = get_api_key()
    if not validate_key(api_key):
        st.markdown("""
        <div class="warn-box">
        ⚠️ <strong>OpenAI API key required.</strong><br>
        Enter your key in the left sidebar to enable summarization.
        </div>
        """, unsafe_allow_html=True)
        return

    model = st.session_state.get("model", "gpt-4o")
    max_tokens = st.session_state.get("max_tokens", 3000)

    # ── Summarize button ──────────────────────────────────────
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run = st.button(
            f"🚀 Summarize {len(uploaded_files)} PDF{'s' if len(uploaded_files)>1 else ''}",
            use_container_width=True,
            type="primary",
        )

    if not run:
        return

    # ── Pipeline ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## ⚡ Processing")

    extractor = PDFExtractor()
    chunker   = DocumentChunker(max_chunk_tokens=max_tokens, model_name=model)

    try:
        summarizer = PDFSummarizer(api_key=api_key, model=model)
    except ValueError as e:
        st.error(str(e))
        return

    all_docs:     List[PDFDocument]     = []
    all_summaries: List[DocumentSummary] = []

    overall_bar = st.progress(0, text="Starting…")
    status_box  = st.empty()

    for file_idx, uploaded in enumerate(uploaded_files):
        fname = uploaded.name
        file_bytes = uploaded.read()

        # ── Extraction ──────────────────────────────────────
        status_box.markdown(f'<div class="prog-label">📥 Extracting <strong>{fname}</strong>…</div>',
                            unsafe_allow_html=True)
        overall_bar.progress(
            int((file_idx / len(uploaded_files)) * 30),
            text=f"Extracting {fname}…"
        )

        try:
            doc = extractor.extract_from_bytes(file_bytes, fname)
        except Exception as e:
            st.error(f"❌ Extraction failed for **{fname}**: {e}")
            continue

        all_docs.append(doc)

        # ── Chunking ────────────────────────────────────────
        status_box.markdown(f'<div class="prog-label">✂️ Chunking <strong>{fname}</strong>…</div>',
                            unsafe_allow_html=True)
        chunks = chunker.chunk_document(doc)

        if not chunks:
            st.warning(f"⚠️ No extractable text found in **{fname}**. Skipping.")
            continue

        # ── Chunk summaries ─────────────────────────────────
        chunk_bar = st.progress(0, text=f"Summarizing {fname}…")

        def on_chunk(done: int, total: int):
            pct = int((done / total) * 100)
            chunk_bar.progress(pct, text=f"🧠 Summarizing {fname}: chunk {done}/{total}")
            overall_pct = int(
                ((file_idx + done / total) / len(uploaded_files)) * 70 + 30
            )
            overall_bar.progress(min(overall_pct, 99), text=f"Processing {fname}…")

        status_box.markdown(
            f'<div class="prog-label">🧠 LLM summarizing <strong>{fname}</strong> '
            f'({len(chunks)} chunk{"s" if len(chunks)>1 else ""})…</div>',
            unsafe_allow_html=True
        )

        try:
            chunk_summaries = summarizer.summarize_chunks(chunks, on_chunk_done=on_chunk)
            doc_summary = summarizer.summarize_document(doc, chunk_summaries)
        except Exception as e:
            st.error(f"❌ Summarization failed for **{fname}**: {e}")
            chunk_bar.empty()
            continue

        all_summaries.append(doc_summary)
        chunk_bar.progress(100, text=f"✅ {fname} complete!")

    # ── Multi-doc unified summary ─────────────────────────────
    multi_summary: Optional[MultiDocSummary] = None
    if len(all_summaries) >= 1:
        if len(all_summaries) > 1:
            status_box.markdown(
                '<div class="prog-label">🌐 Generating unified cross-document analysis…</div>',
                unsafe_allow_html=True
            )
        try:
            multi_summary = summarizer.summarize_collection(all_summaries)
        except Exception as e:
            st.error(f"❌ Unified summary failed: {e}")

    overall_bar.progress(100, text="✅ All done!")
    status_box.empty()

    if not all_summaries:
        st.error("No summaries were generated. Check your PDFs and API key.")
        return

    # ── Results ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📋 Summary Results")

    # Stats row
    total_pages = sum(d.total_pages for d in all_docs)
    total_words = sum(d.total_words for d in all_docs)
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card"><div class="val">{len(all_docs)}</div><div class="lbl">PDFs Processed</div></div>
        <div class="stat-card"><div class="val">{total_pages}</div><div class="lbl">Total Pages</div></div>
        <div class="stat-card"><div class="val">{total_words:,}</div><div class="lbl">Words Analyzed</div></div>
        <div class="stat-card"><div class="val">{len(all_summaries)}</div><div class="lbl">Summaries Generated</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────
    tab_labels = [f"📄 {ds.file_name[:25]}" for ds in all_summaries]
    if multi_summary and len(all_summaries) > 1:
        tab_labels.append("🌐 Unified Report")

    tabs = st.tabs(tab_labels)

    # Individual document tabs
    for i, ds in enumerate(all_summaries):
        with tabs[i]:
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"### 📄 {ds.file_name}")
            with col_b:
                md_content = build_doc_markdown(ds)
                st.download_button(
                    "⬇️ Download .md",
                    data=md_content,
                    file_name=ds.file_name.replace(".pdf", "_summary.md"),
                    mime="text/markdown",
                    use_container_width=True,
                )

            # Stats
            st.markdown(f"""
            <div class="stat-row">
                <div class="stat-card"><div class="val">{ds.total_pages}</div><div class="lbl">Pages</div></div>
                <div class="stat-card"><div class="val">{ds.total_words:,}</div><div class="lbl">Words</div></div>
                <div class="stat-card"><div class="val">{len(ds.section_summaries)}</div><div class="lbl">Chunks</div></div>
            </div>
            """, unsafe_allow_html=True)

            # Metadata
            if ds.metadata:
                with st.expander("📋 Document Metadata", expanded=False):
                    for k, v in ds.metadata.items():
                        st.markdown(f"**{k.capitalize()}:** {v}")

            st.markdown("---")

            # Full summary
            st.markdown(ds.full_summary)

            # Chunk-level detail
            if len(ds.section_summaries) > 1:
                with st.expander(f"🔍 View Chunk-Level Detail ({len(ds.section_summaries)} chunks)", expanded=False):
                    for cs in ds.section_summaries:
                        st.markdown(f"**Chunk {cs.chunk_id} — Pages {cs.page_range}**")
                        st.markdown(cs.summary)
                        st.markdown("---")

    # Unified tab (multi-doc only)
    if multi_summary and len(all_summaries) > 1:
        with tabs[-1]:
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown("### 🌐 Cross-Document Unified Analysis")
            with col_b:
                md_content = build_multi_markdown(multi_summary)
                st.download_button(
                    "⬇️ Download .md",
                    data=md_content,
                    file_name="multi_doc_summary.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

            st.markdown(f"""
            <div class="stat-row">
                <div class="stat-card"><div class="val">{len(all_summaries)}</div><div class="lbl">Documents</div></div>
                <div class="stat-card"><div class="val">{total_pages}</div><div class="lbl">Total Pages</div></div>
                <div class="stat-card"><div class="val">{total_words:,}</div><div class="lbl">Total Words</div></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown(multi_summary.master_summary)

    # ── Download all as ZIP ───────────────────────────────────
    st.markdown("---")
    st.markdown("### 💾 Download All Summaries")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        zip_bytes = make_zip(all_summaries, multi_summary)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "⬇️ Download All as ZIP (.md + .txt for each PDF)",
            data=zip_bytes,
            file_name=f"pdf_summaries_{ts}.zip",
            mime="application/zip",
            use_container_width=True,
            type="primary",
        )

    st.balloons()


if __name__ == "__main__":
    main()
