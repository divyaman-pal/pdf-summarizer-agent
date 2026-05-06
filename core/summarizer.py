"""
core/summarizer.py — Map-reduce LLM summarization pipeline.
Map  : summarize each chunk independently
Reduce: merge chunk summaries → document summary
Unify : cross-document analysis (multi-PDF)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from core.chunker import TextChunk
from core.pdf_extractor import PDFDocument


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class ChunkSummary:
    chunk_id: int
    source_file: str
    page_range: str
    summary: str


@dataclass
class DocumentSummary:
    file_name: str
    total_pages: int
    total_words: int
    metadata: Dict
    section_summaries: List[ChunkSummary] = field(default_factory=list)
    full_summary: str = ""


@dataclass
class MultiDocSummary:
    document_summaries: List[DocumentSummary]
    master_summary: str = ""


# ── Prompt templates ──────────────────────────────────────────────────────────

_CHUNK_SYS = """You are an expert document analyst. Produce a thorough, precise summary of the PDF chunk below.
- Cover ALL topics, facts, numbers, statistics, names, and dates present.
- Use bullet points for lists of data.
- Do NOT add information not present in the text.
- Be concise but complete."""

_CHUNK_HMN = """Source: {source} | Pages: {pages} | Chunk #{cid}

TEXT:
──────────────────────────────────────
{text}
──────────────────────────────────────

Write a detailed summary of this chunk."""

_DOC_SYS = """You are a senior document summarization specialist. Synthesize chunk-level summaries into one authoritative master summary.

Use EXACTLY these markdown headers (include all, even if section is N/A):

## 📌 Document Overview
## 📚 Key Topics & Main Points
## 🔍 Critical Findings & Conclusions
## 📊 Data, Figures & Tables
## ✅ Action Items & Recommendations
## 🏷️ Important Entities (Names, Orgs, Dates, Locations)

Rules: comprehensive, professional, cite page numbers where relevant."""

_DOC_HMN = """Document : {name}
Pages    : {pages}
Words    : {words}
Metadata : {meta}

CHUNK SUMMARIES:
{chunks}

Produce the master summary now."""

_MULTI_SYS = """You are a chief research analyst synthesizing multiple document summaries.

Use EXACTLY these markdown headers:

## 🌐 Collection Overview
## 🔗 Cross-Document Themes
## ⚖️ Comparative Analysis
## 💡 Unified Key Findings
## 🎯 Unified Conclusions & Recommendations"""

_MULTI_HMN = """Total documents: {n}

{summaries}

Produce the unified cross-document analysis now."""


# ── Summarizer ────────────────────────────────────────────────────────────────

class PDFSummarizer:
    """LLM-powered summarization with optional streaming callback."""

    def __init__(self, api_key: str, model: str = "gpt-4o", temperature: float = 0.2):
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        self.llm = ChatOpenAI(model=model, temperature=temperature, api_key=api_key)

    # ── Public ────────────────────────────────────────────────

    def summarize_chunks(
        self,
        chunks: List[TextChunk],
        on_chunk_done: Optional[Callable[[int, int], None]] = None,
    ) -> List[ChunkSummary]:
        summaries = []
        for i, chunk in enumerate(chunks):
            prompt = _CHUNK_HMN.format(
                source=chunk.source_file, pages=chunk.page_range,
                cid=chunk.chunk_id, text=chunk.text,
            )
            resp = self._call(_CHUNK_SYS, prompt)
            summaries.append(ChunkSummary(chunk.chunk_id, chunk.source_file, chunk.page_range, resp))
            if on_chunk_done:
                on_chunk_done(i + 1, len(chunks))
        return summaries

    def summarize_document(self, doc: PDFDocument, chunk_summaries: List[ChunkSummary]) -> DocumentSummary:
        combined = "\n\n".join(
            f"--- Chunk {cs.chunk_id} | Pages {cs.page_range} ---\n{cs.summary}"
            for cs in chunk_summaries
        )
        prompt = _DOC_HMN.format(
            name=doc.file_name, pages=doc.total_pages,
            words=f"{doc.total_words:,}", meta=doc.metadata or "N/A",
            chunks=combined,
        )
        full = self._call(_DOC_SYS, prompt)
        return DocumentSummary(
            file_name=doc.file_name,
            total_pages=doc.total_pages,
            total_words=doc.total_words,
            metadata=doc.metadata,
            section_summaries=chunk_summaries,
            full_summary=full,
        )

    def summarize_collection(self, doc_summaries: List[DocumentSummary]) -> MultiDocSummary:
        if len(doc_summaries) == 1:
            return MultiDocSummary(doc_summaries, doc_summaries[0].full_summary)

        combined = "\n\n".join(
            f"═══ Document {i}: {ds.file_name} ({ds.total_pages} pages) ═══\n{ds.full_summary}"
            for i, ds in enumerate(doc_summaries, 1)
        )
        master = self._call(_MULTI_SYS, _MULTI_HMN.format(n=len(doc_summaries), summaries=combined))
        return MultiDocSummary(doc_summaries, master)

    # ── LLM call ─────────────────────────────────────────────

    def _call(self, system: str, human: str) -> str:
        resp = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        return resp.content.strip()
