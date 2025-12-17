import os
import re
from typing import List

import pdfplumber
from docx import Document as DocxDocument

from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# Text extraction
# --------------------------------------------------

def extract_text_from_pdf(path: str) -> str:
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)


def extract_text_from_docx(path: str) -> str:
    doc = DocxDocument(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# --------------------------------------------------
# Text normalization
# --------------------------------------------------

def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\u00a0", " ")
    return text.strip()


# --------------------------------------------------
# Advanced chunking
# --------------------------------------------------

def chunk_text(text: str) -> List[str]:
    """
    Token-aware semantic chunking using LangChain splitters.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,          # optimal for Gemini embeddings
        chunk_overlap=150,
        separators=[
            "\n\n",              # paragraphs
            "\n",
            ". ",
            "? ",
            "! ",
            ", ",
            " "
        ]
    )

    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if len(c.strip()) > 50]


# --------------------------------------------------
# Unified ingest function
# --------------------------------------------------

def ingest_file(path: str) -> List[str]:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        text = extract_text_from_pdf(path)
    elif ext in {".docx", ".doc"}:
        text = extract_text_from_docx(path)
    elif ext in {".txt", ".md"}:
        text = extract_text_from_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    text = normalize_text(text)
    return chunk_text(text)
