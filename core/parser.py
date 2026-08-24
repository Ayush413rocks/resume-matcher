"""
parser.py
Extracts plain text from resumes (PDF or .txt) and job descriptions
(plain text or a pasted URL body).
"""

from __future__ import annotations
import re
from pathlib import Path

import pdfplumber


def extract_text_from_pdf(file_path: str | Path) -> str:
    """Extract raw text from a PDF resume."""
    text_chunks = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_chunks.append(page_text)
    return "\n".join(text_chunks)


def extract_text_from_txt(file_path: str | Path) -> str:
    return Path(file_path).read_text(encoding="utf-8", errors="ignore")


def load_document(file_path: str | Path) -> str:
    """Dispatch based on file extension. Supports .pdf and .txt."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        raw = extract_text_from_pdf(path)
    elif suffix in (".txt", ".md"):
        raw = extract_text_from_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .pdf or .txt")
    return clean_text(raw)


def clean_text(text: str) -> str:
    """Normalize whitespace, drop empty lines, strip bullet artifacts."""
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()
