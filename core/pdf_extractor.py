"""
core/pdf_extractor.py — Extracts text, tables & metadata from PDFs.
Primary engine: pdfplumber | Fallback: PyPDF2
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import pdfplumber
import PyPDF2


@dataclass
class PageContent:
    page_number: int
    text: str
    tables: List[List[List[str]]] = field(default_factory=list)
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.text.split())


@dataclass
class PDFDocument:
    file_name: str
    total_pages: int
    metadata: Dict[str, Any]
    pages: List[PageContent]
    full_text: str = ""
    total_words: int = 0
    extraction_method: str = "unknown"
    extraction_errors: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.full_text = "\n\n".join(
            f"[PAGE {p.page_number}]\n{p.text}"
            for p in self.pages if p.text.strip()
        )
        self.total_words = sum(p.word_count for p in self.pages)


class PDFExtractor:
    """Extracts content from PDF bytes (file-like objects)."""

    def extract_from_bytes(self, file_bytes: bytes, file_name: str) -> PDFDocument:
        doc = self._try_pdfplumber(file_bytes, file_name)
        if not doc or not doc.full_text.strip():
            doc = self._try_pypdf2(file_bytes, file_name)
        return doc

    def _try_pdfplumber(self, file_bytes: bytes, file_name: str) -> Optional[PDFDocument]:
        try:
            pages: List[PageContent] = []
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                meta = self._clean_meta(pdf.metadata or {})
                for pg in pdf.pages:
                    text = pg.extract_text(x_tolerance=3, y_tolerance=3) or ""
                    tables = []
                    try:
                        for tbl in pg.extract_tables():
                            if tbl:
                                clean = [[c or "" for c in row] for row in tbl]
                                tables.append(clean)
                                text += "\n" + "\n".join(" | ".join(r) for r in clean)
                    except Exception:
                        pass
                    pages.append(PageContent(pg.page_number, text.strip(), tables))

            return PDFDocument(
                file_name=file_name,
                total_pages=len(pages),
                metadata=meta,
                pages=pages,
                extraction_method="pdfplumber",
            )
        except Exception:
            return None

    def _try_pypdf2(self, file_bytes: bytes, file_name: str) -> PDFDocument:
        pages, errors = [], []
        meta = {}
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            meta = self._clean_meta(dict(reader.metadata or {}))
            for i, pg in enumerate(reader.pages, 1):
                try:
                    text = pg.extract_text() or ""
                except Exception as e:
                    text = ""
                    errors.append(f"Page {i}: {e}")
                pages.append(PageContent(i, text.strip()))
        except Exception as e:
            raise RuntimeError(f"Both extractors failed for {file_name}: {e}")

        return PDFDocument(
            file_name=file_name,
            total_pages=len(pages),
            metadata=meta,
            pages=pages,
            extraction_method="PyPDF2",
            extraction_errors=errors,
        )

    @staticmethod
    def _clean_meta(raw: Dict) -> Dict:
        key_map = {
            "/Title": "title", "/Author": "author", "/Subject": "subject",
            "/Creator": "creator", "/Producer": "producer",
            "/CreationDate": "created", "/Keywords": "keywords",
        }
        return {
            friendly: str(raw[k]).strip()
            for k, friendly in key_map.items()
            if raw.get(k)
        }
