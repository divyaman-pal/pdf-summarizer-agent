"""
app.py — PDF Summarization Agent · Streamlit Web App
──────────────────────────────────────────────────────
Features:
  • Upload 1–N PDF files via drag-and-drop
  • Secure API key entry (sidebar, never stored)
  • Real-time progress tracking per document
  • Rich, detailed, connected prose summaries
  • Per-document + unified multi-doc summary tabs
  • One-click download of all summaries (.md / .txt / .zip)
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
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #6C63FF 0%, #3ECFCF 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 1.8rem;
    box-shadow: 0 8px 32px rgba(108,99,255,0.25);
}
.hero h1 { color: #fff; font-size: 2.4rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
.hero p  { color: rgba(255,255,255,0.88); font-size: 1.05rem; margin: 0.5rem 0 0; }

/* Stat cards */
.stat-row { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
.stat-card {
    flex: 1; min-width: 110px;
    background: #1A1D27;
    border: 1px solid #2D3142;
    border-radius: 12px;
    padding: 1rem 0.8rem;
    text-align: center;
}
.stat-card .val { font-size: 1.75rem; font-weight: 700; color: #6C63FF; }
.stat-card .lbl { font-size: 0.76rem; color: #888; margin-top: 0.25rem; text-transform: uppercase; letter-spacing: 0.5px; }

/* File pill */
.file-pill {
    display: inline-block;
    background: #1A1D27;
    border: 1px solid #6C63FF55;
    border-radius: 20px;
    padding: 0.28rem 0.8rem;
    font-size: 0.82rem;
    color: #CACADE;
    margin: 0.2rem;
}

/* Summary prose area */
.prose-section {
    background: #13151f;
    border: 1px solid #242736;
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    margin: 0.6rem 0 1.2rem 0;
    line-height: 1.8;
    font-size: 0.97rem;
}

/* Warning / success */
.warn-box {
    background: #2a1a1a;
    border-left: 4px solid #ff4b4b;
    border-radius: 8px;
    padding: 0.9rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
}
.success-box {
    background: #162a1e;
    border-left: 4px solid #21c55d;
    border-radius: 8px;
    padding: 0.9rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
}
.info-box {
    background: #141c2e;
    border-left: 4px solid #6C63FF;
    border-radius: 8px;
    padding: 0.9rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #aab;
}

/* Progress label */
.prog-label { font-size: 0.88rem; color: #aaa; margin-bottom: 0.3rem; }

/* Sidebar */
[data-testid="stSidebar"] { background: #0f1119; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 0.4rem; }
.stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0 !important; font-weight: 600; }

/* Buttons */
.stButton > button, .stDownloadButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
}

/* Section divider */
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #6C63FF;
    margin: 1.5rem 0 0.4rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #2D3142;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def strip_md(text: str) -> str:
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

> Generated: {ts} | Pages: {ds.total_pages} | Words analysed: {ds.total_words:,}

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

## Documents Analysed
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

        # API key
        has_secret = False
        try:
            has_secret = bool(st.secrets.get("OPENAI_API_KEY"))
        except Exception:
            pass

        if has_secret:
            st.markdown(
                '<div class="success-box">🔐 API key loaded from Streamlit Secrets</div>',
                unsafe_allow_html=True,
            )
        else:
            key = st.text_input(
                "🔑 OpenAI API Key",
                type="password",
                placeholder="sk-...",
                help="Your key is used only for this session and never stored.",
                value=st.session_state.get("api_key", ""),
            )
            st.session_state["api_key"] = key
            if key and not validate_key(key):
                st.warning("Key should start with `sk-`")
            elif key:
                st.success("✅ Key accepted")

        st.markdown("---")
        st.markdown("## 🧠 Model")

        model = st.selectbox(
            "OpenAI Model",
            ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=0,
            help="gpt-4o produces the richest, most detailed summaries.",
        )
        st.session_state["model"] = model

        st.markdown("---")
        st.markdown("## ℹ️ How It Works")
        st.markdown("""
Your PDF is split into page-groups and each group is sent to the AI 
separately, then all results are merged into one detailed, connected summary.

**Pipeline:**
1. 📥 Extract text & tables
2. ✂️ Split into page chunks  
3. 🧠 AI summarises each chunk
4. 🔀 Merge into full document summary
5. 🌐 Cross-doc analysis (multi-PDF)
        """)
        st.markdown("---")
        st.caption("⚠️ Never share your API key in a public chat or message.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    render_sidebar()

    # Hero
    st.markdown("""
    <div class="hero">
        <h1>📄 PDF Summarization Agent</h1>
        <p>Upload one or more PDFs · Get deep, detailed, connected AI summaries</p>
    </div>
    """, unsafe_allow_html=True)

    # Upload
    st.markdown("### 📂 Upload PDF Files")
    uploaded_files = st.file_uploader(
        "label",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Upload one or more PDF files. Max 200 MB each.",
    )

    if not uploaded_files:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:#555;
                    border:2px dashed #2D3142;border-radius:16px;margin-top:1rem;">
            <div style="font-size:3rem;">📄</div>
            <div style="font-size:1.1rem;margin-top:0.5rem;color:#666;">
                Drop your PDFs here to get started
            </div>
            <div style="font-size:0.84rem;margin-top:0.4rem;color:#444;">
                Reports · Research papers · Contracts · Manuals · Any PDF
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # File pills
    pills = "".join(
        f'<span class="file-pill">📄 {f.name} &nbsp;({f.size/1024:.0f} KB)</span>'
        for f in uploaded_files
    )
    st.markdown(pills, unsafe_allow_html=True)

    # API key gate
    api_key = get_api_key()
    if not validate_key(api_key):
        st.markdown("""
        <div class="warn-box">
        ⚠️ <strong>OpenAI API key required.</strong>
        Enter your key in the left sidebar to enable summarization.
        </div>
        """, unsafe_allow_html=True)
        return

    model = st.session_state.get("model", "gpt-4o")

    # Summarize button
    st.markdown("---")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        run = st.button(
            f"🚀  Summarize {len(uploaded_files)} PDF{'s' if len(uploaded_files) > 1 else ''}",
            use_container_width=True,
            type="primary",
        )

    if not run:
        return

    # ── Pipeline ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## ⚡ Processing")

    extractor = PDFExtractor()
    chunker   = DocumentChunker(max_chunk_tokens=3000, model_name=model)

    try:
        summarizer = PDFSummarizer(api_key=api_key, model=model)
    except ValueError as e:
        st.error(str(e))
        return

    all_docs: List[PDFDocument] = []
    all_summaries: List[DocumentSummary] = []

    overall_bar = st.progress(0, text="Starting…")
    status_box  = st.empty()

    for file_idx, uploaded in enumerate(uploaded_files):
        fname      = uploaded.name
        file_bytes = uploaded.read()

        # Extract
        status_box.markdown(
            f'<div class="prog-label">📥 Extracting <strong>{fname}</strong>…</div>',
            unsafe_allow_html=True,
        )
        overall_bar.progress(
            int((file_idx / len(uploaded_files)) * 20),
            text=f"Extracting {fname}…",
        )

        try:
            doc = extractor.extract_from_bytes(file_bytes, fname)
        except Exception as e:
            st.error(f"❌ Extraction failed for **{fname}**: {e}")
            continue

        all_docs.append(doc)

        # Chunk
        status_box.markdown(
            f'<div class="prog-label">✂️ Splitting <strong>{fname}</strong> into chunks…</div>',
            unsafe_allow_html=True,
        )
        chunks = chunker.chunk_document(doc)

        if not chunks:
            st.warning(f"⚠️ No extractable text in **{fname}** — skipping.")
            continue

        # Summarize chunks
        chunk_bar = st.progress(0, text=f"Summarising {fname}…")

        def on_chunk(done: int, total: int, _fname=fname):
            pct = int((done / total) * 100)
            chunk_bar.progress(pct, text=f"🧠 {_fname}: chunk {done}/{total}")
            base = int(20 + (file_idx / len(uploaded_files)) * 75)
            bonus = int((done / total) * (75 / len(uploaded_files)))
            overall_bar.progress(min(base + bonus, 97), text=f"Summarising {_fname}…")

        status_box.markdown(
            f'<div class="prog-label">🧠 AI summarising <strong>{fname}</strong> '
            f'({len(chunks)} chunk{"s" if len(chunks) > 1 else ""})…</div>',
            unsafe_allow_html=True,
        )

        try:
            chunk_sums  = summarizer.summarize_chunks(chunks, on_chunk_done=on_chunk)
            doc_summary = summarizer.summarize_document(doc, chunk_sums)
        except Exception as e:
            st.error(f"❌ Summarisation failed for **{fname}**: {e}")
            chunk_bar.empty()
            continue

        all_summaries.append(doc_summary)
        chunk_bar.progress(100, text=f"✅ {fname} done!")

    # Unified (multi-doc)
    multi_summary: Optional[MultiDocSummary] = None
    if all_summaries:
        if len(all_summaries) > 1:
            status_box.markdown(
                '<div class="prog-label">🌐 Generating cross-document unified analysis…</div>',
                unsafe_allow_html=True,
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

    # ── Results ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📋 Summary Results")

    total_pages = sum(d.total_pages for d in all_docs)
    total_words = sum(d.total_words for d in all_docs)
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="val">{len(all_docs)}</div>
            <div class="lbl">PDFs Processed</div>
        </div>
        <div class="stat-card">
            <div class="val">{total_pages}</div>
            <div class="lbl">Total Pages</div>
        </div>
        <div class="stat-card">
            <div class="val">{total_words:,}</div>
            <div class="lbl">Words Analysed</div>
        </div>
        <div class="stat-card">
            <div class="val">{len(all_summaries)}</div>
            <div class="lbl">Summaries</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs
    tab_labels = [f"📄 {ds.file_name[:28]}" for ds in all_summaries]
    if multi_summary and len(all_summaries) > 1:
        tab_labels.append("🌐 Unified Report")
    tabs = st.tabs(tab_labels)

    # ── Per-document tabs ─────────────────────────────────────
    for i, ds in enumerate(all_summaries):
        with tabs[i]:
            # Header + download
            hc1, hc2 = st.columns([3, 1])
            with hc1:
                st.markdown(f"### 📄 {ds.file_name}")
            with hc2:
                st.download_button(
                    "⬇️ Download .md",
                    data=build_doc_markdown(ds),
                    file_name=ds.file_name.replace(".pdf", "_summary.md"),
                    mime="text/markdown",
                    use_container_width=True,
                )

            # Doc stats
            st.markdown(f"""
            <div class="stat-row">
                <div class="stat-card">
                    <div class="val">{ds.total_pages}</div><div class="lbl">Pages</div>
                </div>
                <div class="stat-card">
                    <div class="val">{ds.total_words:,}</div><div class="lbl">Words</div>
                </div>
                <div class="stat-card">
                    <div class="val">{len(ds.section_summaries)}</div><div class="lbl">Chunks</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Metadata
            if ds.metadata:
                with st.expander("📋 Document Metadata", expanded=False):
                    mc1, mc2 = st.columns(2)
                    items = list(ds.metadata.items())
                    for j, (k, v) in enumerate(items):
                        (mc1 if j % 2 == 0 else mc2).markdown(f"**{k.capitalize()}:** {v}")

            st.markdown("---")

            # ── Full summary — rendered as rich markdown ──────
            st.markdown(ds.full_summary)

            # ── Chunk-level detail (collapsible) ──────────────
            if len(ds.section_summaries) > 1:
                st.markdown("---")
                with st.expander(
                    f"🔍 View Raw Chunk Detail ({len(ds.section_summaries)} chunks processed)",
                    expanded=False,
                ):
                    st.markdown(
                        '<div class="info-box">These are the raw per-chunk summaries that were '
                        'merged to create the master summary above.</div>',
                        unsafe_allow_html=True,
                    )
                    for cs in ds.section_summaries:
                        st.markdown(f"**Chunk {cs.chunk_id} — Pages {cs.page_range}**")
                        st.markdown(cs.summary)
                        st.markdown("---")

    # ── Unified cross-doc tab ─────────────────────────────────
    if multi_summary and len(all_summaries) > 1:
        with tabs[-1]:
            hc1, hc2 = st.columns([3, 1])
            with hc1:
                st.markdown("### 🌐 Cross-Document Unified Analysis")
            with hc2:
                st.download_button(
                    "⬇️ Download .md",
                    data=build_multi_markdown(multi_summary),
                    file_name="unified_summary.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

            st.markdown(f"""
            <div class="stat-row">
                <div class="stat-card">
                    <div class="val">{len(all_summaries)}</div><div class="lbl">Documents</div>
                </div>
                <div class="stat-card">
                    <div class="val">{total_pages}</div><div class="lbl">Total Pages</div>
                </div>
                <div class="stat-card">
                    <div class="val">{total_words:,}</div><div class="lbl">Total Words</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown(multi_summary.master_summary)

    # ── Download all as ZIP ───────────────────────────────────
    st.markdown("---")
    st.markdown("### 💾 Download Everything")
    dc1, dc2, dc3 = st.columns([1, 2, 1])
    with dc2:
        zip_bytes = make_zip(all_summaries, multi_summary)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "⬇️  Download All Summaries as ZIP  (Markdown + Plain Text)",
            data=zip_bytes,
            file_name=f"pdf_summaries_{ts}.zip",
            mime="application/zip",
            use_container_width=True,
            type="primary",
        )

    st.balloons()


if __name__ == "__main__":
    main()
