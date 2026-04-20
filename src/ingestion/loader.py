"""
Loads documents from disk (.txt, .md, .html, .pdf) and normalizes them to
plaintext Document objects.

Everything downstream (chunker, embedder, BM25) works on strings, so format
handling is isolated here. Adding a new format only requires one new method.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".pdf"}


@dataclass
class Document:
    content: str
    source: str
    doc_id: str
    title: str
    metadata: dict = field(default_factory=dict)


class DocumentLoader:

    def load_file(self, path: Path) -> Optional[Document]:
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.warning(f"Skipping unsupported file: {path}")
            return None

        try:
            content = self._extract_text(path)
            content = self._clean_whitespace(content)
            if not content.strip():
                logger.warning(f"Empty content: {path}")
                return None

            return Document(
                content=content,
                source=str(path),
                doc_id=self._doc_id(path),
                title=path.stem,
                metadata={"file_type": path.suffix.lower()},
            )
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return None

    def load_directory(self, directory: Path) -> list[Document]:
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        docs = []
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                doc = self.load_file(path)
                if doc:
                    docs.append(doc)

        logger.info(f"Loaded {len(docs)} documents from {directory}")
        return docs

    def _extract_text(self, path: Path) -> str:
        ext = path.suffix.lower()
        if ext in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="replace")
        elif ext in {".html", ".htm"}:
            return self._load_html(path)
        elif ext == ".pdf":
            return self._load_pdf(path)
        return ""

    def _load_html(self, path: Path) -> str:
        from bs4 import BeautifulSoup

        raw = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw, "lxml")

        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
            tag.decompose()

        return soup.get_text(separator="\n")

    def _load_pdf(self, path: Path) -> str:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[Page {i + 1}]\n{text}")

        if not pages:
            logger.warning(f"No text extracted from {path}. May be a scanned PDF.")

        return "\n\n".join(pages)

    @staticmethod
    def _clean_whitespace(text: str) -> str:
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    @staticmethod
    def _doc_id(path: Path) -> str:
        # MD5 of absolute path — stable across content edits, unique per file
        return hashlib.md5(str(path.resolve()).encode()).hexdigest()[:16]
