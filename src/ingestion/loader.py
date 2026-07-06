"""
arXiv corpus downloader + PDF parser with reference/header stripping
and low-confidence extraction flagging. Produces metadata.jsonl.
"""

import json
import re
import time
import hashlib
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

import httpx
import fitz  # PyMuPDF

from src.utils.logger import setup_logger

logger = setup_logger("rags.ingestion.loader")

ARXIV_API_BASE = "https://export.arxiv.org/api/query"
PDF_DIR = Path("data/pdfs")
METADATA_PATH = Path("data/metadata.jsonl")

SEARCH_CONFIG = {
    "NLP": "cat:cs.CL",
    "CV": "cat:cs.CV",
    "RL": 'cat:cs.AI AND all:"reinforcement learning"',
    "Systems": "cat:cs.DC OR cat:cs.OS",
}

HEADER_PATTERNS = [
    re.compile(r"arXiv:\d{4}\.\d{4,}(v\d+)?", re.IGNORECASE),
    re.compile(r"under review|submitted to|published in|© \d{4}|copyright", re.IGNORECASE),
]

REFERENCE_HEADINGS = re.compile(
    r"^\s*(references|bibliography|works cited|literature cited)\s*$",
    re.IGNORECASE,
)


def sanitize_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", s)


def search_arxiv(query: str, max_results: int = 80) -> list[dict]:
    papers = []
    start = 0
    batch_size = 30

    while len(papers) < max_results:
        params = {
            "search_query": query,
            "start": start,
            "max_results": min(batch_size, max_results - len(papers)),
            "sortBy": "relevance",
        }
        resp = httpx.get(ARXIV_API_BASE, params=params, timeout=30)
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        entries = root.findall("atom:entry", ns)
        if not entries:
            break

        for entry in entries:
            arxiv_id_full = entry.find("atom:id", ns).text.strip()
            arxiv_id = arxiv_id_full.split("/")[-1]
            title = " ".join(entry.find("atom:title", ns).text.strip().split())
            published = entry.find("atom:published", ns).text.strip()

            papers.append(
                {
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "arxiv_url": f"https://arxiv.org/abs/{arxiv_id.replace('v', '')}",
                    "published": published,
                }
            )

        start += batch_size
        time.sleep(3)

    return papers[:max_results]


def download_pdf(arxiv_id: str, target_dir: Path) -> Path | None:
    pdf_path = target_dir / f"{arxiv_id}.pdf"
    if pdf_path.exists():
        logger.debug(f"Already downloaded: {arxiv_id}")
        return pdf_path

    url = f"https://arxiv.org/pdf/{arxiv_id}"
    for attempt in range(3):
        try:
            resp = httpx.get(url, timeout=60, follow_redirects=True)
            resp.raise_for_status()
            pdf_path.write_bytes(resp.content)
            logger.info(f"Downloaded: {arxiv_id}")
            time.sleep(2)
            return pdf_path
        except Exception as e:
            logger.warning(f"Attempt {attempt+1}/3 failed for {arxiv_id}: {e}")
            time.sleep(5)
    logger.error(f"Failed to download {arxiv_id} after 3 attempts")
    return None


def extract_textblocks_per_page(doc: fitz.Document) -> list[list[dict]]:
    """Returns list-of-pages, each page = list of text block dicts with bbox+font info."""
    pages = []
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        text_blocks = []
        for b in blocks:
            if b["type"] != 0:
                continue
            for line in b["lines"]:
                for span in line["spans"]:
                    text_blocks.append(
                        {
                            "text": span["text"],
                            "bbox": list(span["bbox"]),
                            "font_size": span["size"],
                            "font_name": span["font"],
                        }
                    )
        pages.append(text_blocks)
    return pages


def detect_boilerplate_lines(pages: list[list[dict]], threshold: float = 0.5) -> set[str]:
    """Lines whose normalized text appears on >=threshold fraction of pages are boilerplate."""
    counter: Counter = Counter()
    page_count = len(pages)
    for page_blocks in pages:
        seen_this_page: set[str] = set()
        for block in page_blocks:
            norm = block["text"].strip().lower()
            if len(norm) > 3:
                seen_this_page.add(norm)
        for norm in seen_this_page:
            counter[norm] += 1

    boilerplate = {text for text, count in counter.items() if count >= page_count * threshold}
    return boilerplate


def find_reference_start_page(pages: list[list[dict]]) -> int | None:
    """Return the first page index (0-based) where a reference heading appears in the last 20% of the paper."""
    total = len(pages)
    search_start = int(total * 0.75)
    for i in range(search_start, total):
        page_text = " ".join(block["text"] for block in pages[i])
        lines = page_text.split("\n")
        for line in lines:
            if REFERENCE_HEADINGS.match(line.strip()):
                return i
    return None


def extract_pdf(pdf_path: Path) -> dict:
    """
    Parse a single PDF. Returns:
      {paper_id, title, num_pages, pages: [{page_num, text, low_confidence, low_confidence_reason}]}
    """
    doc = fitz.open(str(pdf_path))
    metadata = doc.metadata
    title = metadata.get("title") or pdf_path.stem
    pages_blocks = extract_textblocks_per_page(doc)
    boilerplate_set = detect_boilerplate_lines(pages_blocks)
    ref_start = find_reference_start_page(pages_blocks)

    pages_out = []
    low_conf_count = 0

    for i, blocks in enumerate(pages_blocks):
        page_num = i + 1
        is_ref_page = ref_start is not None and page_num > ref_start

        cleaned_spans = []
        for block in blocks:
            norm = block["text"].strip().lower()
            if norm in boilerplate_set:
                continue
            if any(pat.search(block["text"]) for pat in HEADER_PATTERNS):
                continue
            cleaned_spans.append(block["text"])

        page_text = " ".join(cleaned_spans).strip()
        if is_ref_page:
            page_text = ""  # strip entirely; references aren't useful retrieval content

        char_count = len(page_text)
        reason = None
        low_conf = False

        if char_count < 100:
            low_conf = True
            reason = "short_text"
        elif not cleaned_spans:
            low_conf = True
            reason = "no_text_blocks"
        else:
            non_ascii_ratio = sum(1 for c in page_text if ord(c) > 127) / max(char_count, 1)
            if non_ascii_ratio > 0.8:
                low_conf = True
                reason = "high_non_ascii"

        if low_conf:
            low_conf_count += 1

        pages_out.append(
            {
                "page_num": page_num,
                "text": page_text,
                "char_count": char_count,
                "low_confidence": low_conf,
                "low_confidence_reason": reason,
            }
        )

    doc.close()

    paper_id = pdf_path.stem  # e.g. "2103.00001v2"

    return {
        "paper_id": paper_id,
        "title": title,
        "num_pages": len(pages_out),
        "pages": pages_out,
        "low_confidence_pages": [p["page_num"] for p in pages_out if p["low_confidence"]],
        "low_confidence_ratio": low_conf_count / max(len(pages_out), 1),
    }


def run_corpus_download(
    pdf_dir: str | Path | None = None,
    metadata_out: str | Path | None = None,
    max_per_topic: int = 80,
    min_pages: int = 15,
) -> Path:
    """
    Full corpus build: search arXiv, download PDFs, parse, write metadata.jsonl.
    Returns path to metadata.jsonl.
    """
    if pdf_dir is None:
        pdf_dir = PDF_DIR
    if metadata_out is None:
        metadata_out = METADATA_PATH
    pdf_dir = Path(pdf_dir)
    metadata_out = Path(metadata_out)

    pdf_dir.mkdir(parents=True, exist_ok=True)

    # Search and collect deduplicated paper entries
    seen_ids: set[str] = set()
    all_papers: list[dict] = []

    for topic, query in SEARCH_CONFIG.items():
        logger.info(f"Searching arXiv for {topic}: {query}")
        papers = search_arxiv(query, max_results=max_per_topic)
        new = [p for p in papers if p["arxiv_id"] not in seen_ids]
        for p in new:
            p["topic"] = topic
            seen_ids.add(p["arxiv_id"])
        all_papers.extend(new)
        logger.info(f"  {topic}: {len(papers)} found, {len(new)} new (total: {len(all_papers)})")

    logger.info(f"Total unique papers to download: {len(all_papers)}")

    # Download PDFs
    downloaded = []
    for i, paper in enumerate(all_papers):
        path = download_pdf(paper["arxiv_id"], pdf_dir)
        if path:
            downloaded.append(paper)
        if (i + 1) % 20 == 0:
            logger.info(f"Download progress: {i+1}/{len(all_papers)} ({len(downloaded)} succeeded)")

    logger.info(f"Downloaded {len(downloaded)} PDFs")

    # Parse and filter by page count
    entries = []
    total_pages = 0
    total_low_conf = 0
    skipped_short = 0

    for paper in downloaded:
        try:
            result = extract_pdf(pdf_dir / f"{paper['arxiv_id']}.pdf")
        except Exception as e:
            logger.warning(f"Extraction failed for {paper['arxiv_id']}: {e}")
            continue

        if result["num_pages"] < min_pages:
            skipped_short += 1
            continue

        entry = {
            "paper_id": paper["arxiv_id"],
            "title": paper["title"],
            "arxiv_url": paper["arxiv_url"],
            "topic": paper["topic"],
            "num_pages": result["num_pages"],
            "low_confidence_pages": result["low_confidence_pages"],
            "low_confidence_ratio": round(result["low_confidence_ratio"], 4),
            "pdf_path": str(pdf_dir / f"{paper['arxiv_id']}.pdf"),
            "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        entries.append(entry)
        total_pages += result["num_pages"]
        total_low_conf += len(result["low_confidence_pages"])

    # Write metadata.jsonl
    with open(metadata_out, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    low_conf_pct = (total_low_conf / total_pages * 100) if total_pages else 0
    logger.info(
        f"Corpus complete: {len(entries)} papers, {total_pages} pages, "
        f"{total_low_conf} low-conf pages ({low_conf_pct:.2f}%), "
        f"{skipped_short} skipped (<{min_pages} pages)"
    )

    return metadata_out