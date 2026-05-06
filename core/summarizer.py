"""
core/summarizer.py — Map-reduce LLM summarization pipeline.
Map  : summarize each chunk independently (detailed, connective prose)
Reduce: merge chunk summaries → rich structured document summary
Unify : cross-document analysis (multi-PDF)
"""
from __future__ import annotations

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

_CHUNK_SYS = """\
You are a world-class document analyst specialising in producing rich, \
detailed, and deeply connected summaries. Your job is to read a chunk of \
text extracted from a PDF and write a comprehensive narrative summary of it.

STRICT RULES:
1. Write in flowing, connected prose — NOT bullet points, NOT fragmented lists.
2. Cover EVERY topic, argument, fact, number, statistic, name, date, and \
   concept present in the text. Miss nothing.
3. Preserve exact figures, percentages, monetary values, dates, and proper \
   nouns verbatim.
4. Connect ideas with clear transitions so the summary reads as a coherent \
   piece of writing, not a collection of isolated facts.
5. Write as if explaining the content to an intelligent reader who has NOT \
   read the original — give them the full picture.
6. Do NOT add any information, opinion, or inference that is not explicitly \
   present in the source text.
7. Length: write as much as needed to cover everything — do not truncate or \
   skip sections to save space."""

_CHUNK_HMN = """\
Source file : {source}
Pages       : {pages}
Chunk number: {cid}

━━━━━━━━━━━━━━━━━━━━━━━ SOURCE TEXT ━━━━━━━━━━━━━━━━━━━━━━━
{text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Write a detailed, flowing narrative summary of the source text above. \
Cover every piece of information. Use connective language to link ideas \
together naturally."""

# ─────────────────────────────────────────────────────────────────────────────
_DOC_SYS = """\
You are a senior analyst producing a definitive, richly detailed master \
summary of a document. You will receive a series of chunk-level summaries \
that together cover the entire document. Your task is to synthesise them \
into one authoritative, well-written, deeply informative report.

OUTPUT FORMAT — use these exact markdown headers, in this order. Under each \
header write detailed, connected prose paragraphs (NOT bullet lists):

## 📌 Document Overview
Write 2–4 paragraphs describing what this document is, its purpose, scope, \
audience, and overall structure. Mention the document type, authorship, date, \
and context where available.

## 📚 Key Topics & Main Points
Discuss every major topic covered in the document in depth. Organise by theme \
or section. Use sub-headings (###) if the document has clearly distinct \
sections. For each topic explain not just WHAT is covered but HOW it is argued \
or presented. Write in full paragraphs with smooth transitions between topics.

## 🔍 Critical Findings & Conclusions
Explain the most significant findings, results, conclusions, or decisions in \
the document. Why do they matter? What evidence or reasoning supports them? \
Write this as a connected analytical narrative, not a list.

## 📊 Data, Figures & Tables
Describe every quantitative finding, statistic, table, chart, or figure \
mentioned. Embed the numbers naturally in sentences. Explain what each \
figure means in context.

## ✅ Action Items & Recommendations
If the document contains recommendations, next steps, action items, or \
proposed decisions, describe them in full with their rationale. If none \
exist, write "This document does not contain explicit action items."

## 🏷️ Important Entities
Write a concise paragraph identifying all key people, organisations, \
locations, products, technologies, dates, and technical terms mentioned. \
Provide brief context for each.

GLOBAL RULES:
- Every section must be written in rich, flowing prose — no bullet points.
- Be comprehensive: a reader should understand the ENTIRE document from \
  this summary alone.
- Maintain a professional, authoritative tone throughout.
- Cite page numbers in parentheses (p. X) when referencing specific content.
- Minimum length: write as much detail as the content demands."""

_DOC_HMN = """\
Document : {name}
Pages    : {pages}
Words    : {words}
Metadata : {meta}

━━━━━━━━━━━━━━━━━━━━━ CHUNK SUMMARIES ━━━━━━━━━━━━━━━━━━━━━
{chunks}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Using the chunk summaries above, produce the definitive master summary \
of this document. Every section must be detailed, analytical, and written \
in connected prose. Cover the full document — leave nothing out."""

# ─────────────────────────────────────────────────────────────────────────────
_MULTI_SYS = """\
You are a chief research analyst producing a unified report across multiple \
documents. You will receive master summaries of each document. Your job is to \
synthesise them into a single, deeply analytical cross-document report written \
entirely in connected, flowing prose.

OUTPUT FORMAT — use these exact headers. Write rich prose paragraphs under \
each — no bullet points:

## 🌐 Collection Overview
Describe what this collection of documents is about as a whole. What is the \
shared subject, purpose, or context? How do the documents relate to each other?

## 🔗 Cross-Document Themes
Identify and discuss the major themes, concepts, or arguments that recur \
across multiple documents. Explain how each document contributes to or \
develops each theme.

## ⚖️ Comparative Analysis
Compare the documents in depth: where do they agree, where do they differ, \
how do they complement each other? Discuss differences in scope, methodology, \
conclusions, or perspective.

## 💡 Unified Key Findings
Synthesise the most important insights drawn from reading all documents \
together. What does the collection as a whole tell us that no single document \
does alone?

## 🎯 Unified Conclusions & Recommendations
Draw overarching conclusions and any recommendations that apply to the \
collection as a whole. Ground every statement in specific evidence from \
the documents.

GLOBAL RULES: rich connected prose only, no bullet points, comprehensive \
and analytical throughout."""

_MULTI_HMN = """\
Total documents: {n}

{summaries}

Produce the unified cross-document analysis now. Write in detailed, \
connected prose under each header."""


# ── Summarizer ────────────────────────────────────────────────────────────────

class PDFSummarizer:
    """LLM-powered summarization with optional per-chunk progress callback."""

    def __init__(self, api_key: str, model: str = "gpt-4o", temperature: float = 0.3):
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            max_tokens=4096,        # allow long, detailed responses
        )

    # ── Public ────────────────────────────────────────────────

    def summarize_chunks(
        self,
        chunks: List[TextChunk],
        on_chunk_done: Optional[Callable[[int, int], None]] = None,
    ) -> List[ChunkSummary]:
        summaries = []
        for i, chunk in enumerate(chunks):
            prompt = _CHUNK_HMN.format(
                source=chunk.source_file,
                pages=chunk.page_range,
                cid=chunk.chunk_id,
                text=chunk.text,
            )
            resp = self._call(_CHUNK_SYS, prompt)
            summaries.append(
                ChunkSummary(chunk.chunk_id, chunk.source_file, chunk.page_range, resp)
            )
            if on_chunk_done:
                on_chunk_done(i + 1, len(chunks))
        return summaries

    def summarize_document(
        self, doc: PDFDocument, chunk_summaries: List[ChunkSummary]
    ) -> DocumentSummary:
        combined = "\n\n".join(
            f"--- Chunk {cs.chunk_id} | Pages {cs.page_range} ---\n{cs.summary}"
            for cs in chunk_summaries
        )
        prompt = _DOC_HMN.format(
            name=doc.file_name,
            pages=doc.total_pages,
            words=f"{doc.total_words:,}",
            meta=doc.metadata or "N/A",
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

    def summarize_collection(
        self, doc_summaries: List[DocumentSummary]
    ) -> MultiDocSummary:
        if len(doc_summaries) == 1:
            return MultiDocSummary(doc_summaries, doc_summaries[0].full_summary)

        combined = "\n\n".join(
            f"═══ Document {i}: {ds.file_name} ({ds.total_pages} pages) ═══\n{ds.full_summary}"
            for i, ds in enumerate(doc_summaries, 1)
        )
        master = self._call(
            _MULTI_SYS,
            _MULTI_HMN.format(n=len(doc_summaries), summaries=combined),
        )
        return MultiDocSummary(doc_summaries, master)

    # ── Internal ──────────────────────────────────────────────

    def _call(self, system: str, human: str) -> str:
        resp = self.llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=human)]
        )
        return resp.content.strip()
