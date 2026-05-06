"""
core/chunker.py — Splits PDF pages into token-safe chunks for LLM processing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.pdf_extractor import PDFDocument, PageContent


@dataclass
class TextChunk:
    chunk_id: int
    text: str
    source_file: str
    page_range: str
    token_count: int = 0


class DocumentChunker:
    def __init__(self, max_chunk_tokens: int = 3000, model_name: str = "gpt-4o"):
        self.max_chunk_tokens = max_chunk_tokens
        try:
            self.encoder = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoder = tiktoken.get_encoding("cl100k_base")

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_tokens * 4,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_document(self, doc: PDFDocument) -> List[TextChunk]:
        chunks, chunk_id = [], 0
        current_pages: List[PageContent] = []
        current_tokens = 0

        for page in doc.pages:
            if not page.text.strip():
                continue
            page_tokens = self._count(page.text)

            if page_tokens > self.max_chunk_tokens:
                if current_pages:
                    chunks.append(self._build(chunk_id, current_pages, doc.file_name))
                    chunk_id += 1
                    current_pages, current_tokens = [], 0
                for sub in self.splitter.split_text(page.text):
                    chunks.append(self._build(chunk_id, [PageContent(page.page_number, sub)], doc.file_name))
                    chunk_id += 1
                continue

            if current_tokens + page_tokens > self.max_chunk_tokens and current_pages:
                chunks.append(self._build(chunk_id, current_pages, doc.file_name))
                chunk_id += 1
                current_pages, current_tokens = [], 0

            current_pages.append(page)
            current_tokens += page_tokens

        if current_pages:
            chunks.append(self._build(chunk_id, current_pages, doc.file_name))

        return chunks

    def _build(self, chunk_id: int, pages: List[PageContent], source: str) -> TextChunk:
        nums = [p.page_number for p in pages]
        page_range = str(nums[0]) if len(nums) == 1 else f"{nums[0]}-{nums[-1]}"
        text = "\n\n".join(f"[Page {p.page_number}]\n{p.text}" for p in pages)
        return TextChunk(chunk_id, text, source, page_range, self._count(text))

    def _count(self, text: str) -> int:
        try:
            return len(self.encoder.encode(text))
        except Exception:
            return len(text) // 4
