"""
Builds the chatbot's knowledge base from the school website, PDF documents,
and hand-written text/markdown files in data/raw/.
Run this whenever source content changes: python ingest.py
"""

import json
import re
from pathlib import Path

import numpy as np
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"

PDF_PATHS = [RAW_DIR / "school_diary.pdf"]

TEXT_PATHS = list(RAW_DIR.glob("*.txt")) + list(RAW_DIR.glob("*.md"))

WEBSITE_URLS = [
    "https://www.lksec.org/",
    "https://www.lksec.org/about-lks",
    "https://www.lksec.org/vision-mission",
    "https://www.lksec.org/awards.php",
    "https://www.lksec.org/affiliation-and-accreditations",
    "https://www.lksec.org/location",
    "https://www.lksec.org/lksec-story",
    "https://www.lksec.org/milestones",
    "https://www.lksec.org/board-of-governors",
    "https://www.lksec.org/administration",
    "https://www.lksec.org/admission-criteria",
    "https://www.lksec.org/fee-structure",
    "https://www.lksec.org/infrastructure",
    "https://www.lksec.org/teaching-faculty.php",

]

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

OUTPUT_CHUNKS = DATA_DIR / "knowledge_base.json"
OUTPUT_EMBEDDINGS = DATA_DIR / "embeddings.npy"


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_website_text(url: str) -> str:
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return re.sub(r"\n{2,}", "\n\n", text).strip()


def chunk_text(text: str, source: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split on paragraph boundaries so we don't cut a fact in half mid-sentence."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, current = [], ""

    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current = f"{current}\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            current = (current[-overlap:] + "\n" + para).strip() if current else para

    if current:
        chunks.append(current)

    return [{"text": c, "source": source} for c in chunks if len(c) > 20]


def build_knowledge_base():
    all_chunks = []

    # PDFs
    for pdf_path in PDF_PATHS:
        if not pdf_path.exists():
            print(f"Skipping missing PDF: {pdf_path}")
            continue
        print(f"Extracting {pdf_path.name}...")
        text = extract_pdf_text(pdf_path)
        all_chunks.extend(chunk_text(text, source=pdf_path.name))

    # Hand-written text/markdown files (for content pypdf can't extract correctly)
    for text_path in TEXT_PATHS:
        print(f"Reading {text_path.name}...")
        text = text_path.read_text(encoding="utf-8")
        all_chunks.extend(chunk_text(text, source=text_path.name))

    # Website pages
    for url in WEBSITE_URLS:
        print(f"Fetching {url}...")
        try:
            text = extract_website_text(url)
            all_chunks.extend(chunk_text(text, source=url))
        except requests.RequestException as e:
            print(f"  Failed to fetch {url}: {e}")

    if not all_chunks:
        raise RuntimeError("No content extracted — check PDF_PATHS and WEBSITE_URLS.")

    print(f"Embedding {len(all_chunks)} chunks with {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_CHUNKS.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2))
    np.save(OUTPUT_EMBEDDINGS, embeddings)

    print(f"Saved {len(all_chunks)} chunks to {OUTPUT_CHUNKS}")
    print(f"Saved embeddings {embeddings.shape} to {OUTPUT_EMBEDDINGS}")


if __name__ == "__main__":
    build_knowledge_base()