"""
Builds the chatbot knowledge base from the school website, PDF documents,
and text/markdown files in data/raw/.

Run:
    python ingest.py

Run this whenever the school's source content changes.
"""

import json
import re
from pathlib import Path

import numpy as np
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"

PDF_PATHS = [
    RAW_DIR / "school_diary.pdf",
]

TEXT_PATHS = (
    list(RAW_DIR.glob("*.txt"))
    + list(RAW_DIR.glob("*.md"))
)

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

HEADERS = {
    "User-Agent": "LKSEC-AI-Chatbot/1.0"
}


# --------------------------------------------------
# Text extraction
# --------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraphs."""

    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF."""

    reader = PdfReader(str(path))

    pages = []

    for page in reader.pages:
        pages.append(page.extract_text() or "")

    return clean_text("\n\n".join(pages))


def extract_website_text(url: str) -> tuple[str, str]:
    """Extract page title and readable text from a website."""

    response = requests.get(
        url,
        timeout=15,
        headers=HEADERS
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    title = soup.title.get_text(strip=True) if soup.title else url

    for tag in soup([
        "script",
        "style",
        "nav",
        "footer",
        "noscript",
        "header"
    ]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    return title, clean_text(text)


# --------------------------------------------------
# Chunking
# --------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """Simple sentence splitter."""

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[dict]:

    paragraphs = [
        p.strip()
        for p in text.split("\n")
        if p.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:

        sentences = split_sentences(paragraph)

        for sentence in sentences:

            # Sentence fits into current chunk
            if len(current) + len(sentence) + 1 <= chunk_size:

                current = (
                    f"{current} {sentence}".strip()
                )

            else:

                if current:
                    chunks.append({
                        "text": current,
                        "source": source
                    })

                # Preserve a small amount of previous context
                overlap_text = current[-overlap:] if current else ""

                current = (
                    f"{overlap_text} {sentence}".strip()
                )

    if current:
        chunks.append({
            "text": current,
            "source": source
        })

    return [
        chunk
        for chunk in chunks
        if len(chunk["text"]) > 30
    ]


# --------------------------------------------------
# Knowledge base
# --------------------------------------------------

def build_knowledge_base():

    all_chunks = []

    # -----------------------------
    # PDFs
    # -----------------------------

    for pdf_path in PDF_PATHS:

        if not pdf_path.exists():
            print(f"Skipping missing PDF: {pdf_path}")
            continue

        print(f"Extracting {pdf_path.name}...")

        text = extract_pdf_text(pdf_path)

        all_chunks.extend(
            chunk_text(
                text,
                source=pdf_path.name
            )
        )

    # -----------------------------
    # Text / Markdown
    # -----------------------------

    for text_path in TEXT_PATHS:

        print(f"Reading {text_path.name}...")

        text = text_path.read_text(
            encoding="utf-8"
        )

        all_chunks.extend(
            chunk_text(
                text,
                source=text_path.name
            )
        )

    # -----------------------------
    # Website
    # -----------------------------

    for url in WEBSITE_URLS:

        print(f"Fetching {url}...")

        try:

            title, text = extract_website_text(url)

            source = f"{title} ({url})"

            all_chunks.extend(
                chunk_text(
                    text,
                    source=source
                )
            )

        except requests.RequestException as e:

            print(
                f"  Failed to fetch {url}: {e}"
            )

    # -----------------------------
    # Validate
    # -----------------------------

    if not all_chunks:
        raise RuntimeError(
            "No content extracted. "
            "Check PDF_PATHS and WEBSITE_URLS."
        )

    print(
        f"\nCreated {len(all_chunks)} chunks."
    )

    # -----------------------------
    # Embeddings
    # -----------------------------

    print(
        f"Embedding with {EMBEDDING_MODEL}..."
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=32
    )

    # -----------------------------
    # Save
    # -----------------------------

    DATA_DIR.mkdir(
        exist_ok=True
    )

    OUTPUT_CHUNKS.write_text(
        json.dumps(
            all_chunks,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    np.save(
        OUTPUT_EMBEDDINGS,
        embeddings
    )

    print(
        f"\nSaved {len(all_chunks)} chunks to:"
    )

    print(
        f"  {OUTPUT_CHUNKS}"
    )

    print(
        f"Saved embeddings:"
    )

    print(
        f"  {OUTPUT_EMBEDDINGS}"
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":
    build_knowledge_base()