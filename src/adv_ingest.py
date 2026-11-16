"""Multimodal PDF ingestion for the Day 7 Advanced Challenge (100+ pages).

Extracts body text, structured Markdown tables (via pdfplumber), cropped figure/table
PNGs (via PyMuPDF), 2D row-grouped RapidOCR text, and optional VLM descriptions
(via local Ollama qwen3-vl:2b), then indexes everything into Chroma + BM25.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pdfplumber
import pymupdf
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

from src.config import (
    DAY6_REPO_DIR,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    RESULTS_DIR,
    ROOT,
)

PDF_PATH = Path(
    os.getenv(
        "ADV_PDF_PATH",
        str(DAY6_REPO_DIR / "data" / "OpenStax_Introduction_to_Sociology_2e.pdf"),
    )
)
SOURCE_ID = "openstax-introduction-to-sociology-2e"
SOURCE_URL = "https://assets.openstax.org/oscms-prodcms/media/documents/IntroductionToSociology2e-OP_tbTLqMj.pdf"
ATTRIBUTION = (
    "Introduction to Sociology 2e, Paul Peter Urone and Roger Hinrichs, "
    "OpenStax/Rice University, CC BY 4.0. Source: https://openstax.org/details/books/introduction-sociology-2e"
)

ADV_CHUNKS_PATH = ROOT / "data" / "adv_chunks.json"
ADV_MANIFEST_PATH = ROOT / "data" / "adv_manifest.json"
ADV_CROPS_DIR = RESULTS_DIR / "adv_crops"
ADV_CHROMA_DIR = RESULTS_DIR / "adv_chroma"
ADV_COLLECTION = "adv_soc_ch13_21"
VLM_MODEL = os.getenv("ADV_VLM_MODEL", "qwen3-vl:2b")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def make_chunk_id(page: int, region_type: str, idx: int, text: str) -> str:
    raw = f"{SOURCE_ID}|adv|{page}|{region_type}|{idx}|{text[:200]}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def detect_chapter(page_text: str, current_chapter: str) -> str:
    m = re.search(r"Chapter\s+(\d+)\s*\|", page_text)
    if m:
        return m.group(1)
    m2 = re.search(r"(?m)^\s*(\d{2})\.\d+\s+[A-Z]", page_text)
    if m2:
        return m2.group(1)
    return current_chapter


def should_skip_page(page_text: str) -> str | None:
    norm = normalize_text(page_text)
    if len(norm) < 250:
        return "too_short_or_blank"
    head = norm[:250].casefold()
    if any(
        marker in head
        for marker in (
            "section quiz",
            "short answer",
            "further research",
            "references\n",
            "key terms\n",
            "section summary\n",
        )
    ):
        return "end_of_chapter_review_or_references"
    # Skip pages whose main heading is References or Section Quiz even slightly lower.
    if re.search(r"(?m)^(?:References|Section Quiz|Short Answer|Further Research)\s*$", norm[:600]):
        return "end_of_chapter_review_or_references"
    return None


def bbox_overlap_ratio(inner: tuple[float, float, float, float], outer: tuple[float, float, float, float]) -> float:
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    dx = max(0.0, min(ix1, ox1) - max(ix0, ox0))
    dy = max(0.0, min(iy1, oy1) - max(iy0, oy0))
    area = max(1e-6, (ix1 - ix0) * (iy1 - iy0))
    return (dx * dy) / area


def table_to_markdown(rows: list[list[str | None]]) -> str:
    cleaned = [
        [re.sub(r"\s+", " ", (cell or "")).strip().replace("|", "\\|") for cell in row]
        for row in rows
    ]
    if not cleaned:
        return ""
    header = cleaned[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in cleaned[1:]:
        padded = row + [""] * max(0, len(header) - len(row))
        lines.append("| " + " | ".join(padded[: len(header)]) + " |")
    return "\n".join(lines)


def ocr_grouped_rows(ocr_engine: RapidOCR, img_path: Path) -> list[str]:
    res, _ = ocr_engine(str(img_path))
    if not res:
        return []
    items = []
    for box, txt, _score in res:
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        items.append(((min(ys) + max(ys)) / 2.0, min(xs), txt.strip()))
    items.sort(key=lambda t: (t[0], t[1]))
    rows: list[dict[str, Any]] = []
    for y_mid, x_left, txt in items:
        if rows and abs(y_mid - rows[-1]["y_mid"]) <= 18.0:
            rows[-1]["parts"].append((x_left, txt))
            rows[-1]["y_mid"] = (rows[-1]["y_mid"] + y_mid) / 2.0
        else:
            rows.append({"y_mid": y_mid, "parts": [(x_left, txt)]})
    return [" | ".join(t for _, t in sorted(r["parts"], key=lambda p: p[0])) for r in rows]


def find_nearest_caption(
    blocks: list[tuple[float, float, float, float, str, int, int]],
    target_bbox: tuple[float, float, float, float],
    prefix_regex: str,
) -> str:
    tx0, ty0, tx1, ty1 = target_bbox
    best_dist = 1e9
    best_text = ""
    for bx0, by0, bx1, by1, btxt, _bno, btype in blocks:
        if btype != 0:
            continue
        clean = re.sub(r"\s+", " ", btxt).strip()
        if not re.match(prefix_regex, clean):
            continue
        # Prefer blocks just below or just above the target bbox.
        if by0 >= ty1 - 15:
            dist = abs(by0 - ty1)
        elif by1 <= ty0 + 15:
            dist = abs(ty0 - by1) + 20.0
        else:
            dist = abs((by0 + by1) / 2.0 - (ty0 + ty1) / 2.0) + 50.0
        if dist < best_dist and dist < 240.0:
            best_dist = dist
            best_text = clean
    return best_text


def chunk_sentences(text: str, max_size: int = 850) -> list[str]:
    norm = normalize_text(text)
    if not norm:
        return []
    cuts = [m.start() for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z0-9“\"'])", norm)]
    bounds = [0, *cuts, len(norm)]
    units = [norm[a:b].strip() for a, b in zip(bounds, bounds[1:]) if norm[a:b].strip()]
    chunks: list[str] = []
    cur = ""
    for unit in units:
        if len(unit) > max_size:
            if cur:
                chunks.append(cur)
                cur = ""
            # Split extra-long single units cleanly on whitespace.
            words = unit.split(" ")
            buf = ""
            for w in words:
                cand = f"{buf} {w}".strip()
                if len(cand) <= max_size:
                    buf = cand
                else:
                    if buf:
                        chunks.append(buf)
                    buf = w
            if buf:
                chunks.append(buf)
        elif not cur:
            cur = unit
        elif len(cur) + 2 + len(unit) <= max_size:
            cur = f"{cur}\n\n{unit}"
        else:
            chunks.append(cur)
            cur = unit
    if cur:
        chunks.append(cur)
    return chunks


def extract_corpus(start_page: int = 281, end_page: int = 495) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

    ADV_CROPS_DIR.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(str(PDF_PATH))
    ocr_engine = RapidOCR()

    all_chunks: list[dict[str, Any]] = []
    included_pages: list[int] = []
    skipped_pages: list[dict[str, Any]] = []
    stats = {"text": 0, "table": 0, "figure": 0}
    current_chapter = "13"

    with pdfplumber.open(str(PDF_PATH)) as plumb:
        for pno in range(start_page, end_page + 1):
            fitz_page = doc[pno - 1]
            raw_text = fitz_page.get_text("text") or ""
            current_chapter = detect_chapter(raw_text, current_chapter)

            skip_reason = should_skip_page(raw_text)
            if skip_reason is not None:
                skipped_pages.append({"page": pno, "chapter": current_chapter, "reason": skip_reason})
                continue

            included_pages.append(pno)
            blocks = fitz_page.get_text("blocks")
            table_bboxes: list[tuple[float, float, float, float]] = []

            # 1. Extract structured Tables via pdfplumber + save table crop PNGs.
            plumb_page = plumb.pages[pno - 1]
            found_tables = plumb_page.find_tables()
            t_idx = 0
            for tbl in found_tables:
                rows = tbl.extract()
                if not rows or len(rows) < 2 or len(rows[0]) < 2:
                    continue
                bbox = tuple(float(v) for v in tbl.bbox)
                table_bboxes.append(bbox)
                md = table_to_markdown(rows)
                caption = find_nearest_caption(blocks, bbox, r"^Table\s+\d+\.\d+")
                if not caption:
                    # Fallback: check for Table X.Y anywhere in page text.
                    m_cap = re.search(r"Table\s+\d+\.\d+[^\n]{0,180}", raw_text)
                    caption = re.sub(r"\s+", " ", m_cap.group(0)).strip() if m_cap else f"Table on page {pno}"

                clip = pymupdf.Rect(
                    max(0, bbox[0] - 10),
                    max(0, bbox[1] - 32),
                    min(fitz_page.rect.width, bbox[2] + 10),
                    min(fitz_page.rect.height, bbox[3] + 10),
                )
                crop_rel = f"results/adv_crops/p{pno:03d}_table_{t_idx}.png"
                fitz_page.get_pixmap(clip=clip, dpi=180).save(str(ROOT / crop_rel))

                text = f"[TABLE | Page {pno} | Chapter {current_chapter}]\n{caption}\n\n{md}"
                cid = make_chunk_id(pno, "table", t_idx, text)
                all_chunks.append(
                    {
                        "id": cid,
                        "text": text,
                        "source_id": SOURCE_ID,
                        "source": SOURCE_URL,
                        "page": pno,
                        "metadata": {
                            "chapter": current_chapter,
                            "region_type": "table",
                            "chunk_index": t_idx,
                            "bbox": [round(x, 1) for x in bbox],
                            "crop_path": crop_rel,
                            "caption": caption,
                            "attribution": ATTRIBUTION,
                        },
                    }
                )
                stats["table"] += 1
                t_idx += 1

            # 2. Extract Figures (charts, diagrams, maps, photos) + crop PNGs + 2D RapidOCR.
            fig_idx = 0
            seen_xrefs: set[int] = set()
            for im in fitz_page.get_images(full=True):
                xref = im[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                rects = fitz_page.get_image_rects(xref)
                if not rects:
                    continue
                rect = rects[0]
                if rect.width < 80 or rect.height < 60:
                    continue
                bbox = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
                crop_rel = f"results/adv_crops/p{pno:03d}_fig_{fig_idx}.png"
                crop_abs = ROOT / crop_rel
                fitz_page.get_pixmap(clip=rect, dpi=200).save(str(crop_abs))
                ocr_rows = ocr_grouped_rows(ocr_engine, crop_abs)

                caption = find_nearest_caption(blocks, bbox, r"^Figure\s+\d+\.\d+")
                if not caption:
                    m_fig = re.search(r"Figure\s+\d+\.\d+[^\n]{0,200}", raw_text)
                    caption = re.sub(r"\s+", " ", m_fig.group(0)).strip() if m_fig else f"Figure on page {pno}"

                ocr_block = "\n".join(ocr_rows) if ocr_rows else "(no embedded text detected by OCR)"
                text = (
                    f"[FIGURE | Page {pno} | Chapter {current_chapter}]\n"
                    f"Caption: {caption}\n"
                    f"OCR extracted rows:\n{ocr_block}"
                )
                cid = make_chunk_id(pno, "figure", fig_idx, text)
                all_chunks.append(
                    {
                        "id": cid,
                        "text": text,
                        "source_id": SOURCE_ID,
                        "source": SOURCE_URL,
                        "page": pno,
                        "metadata": {
                            "chapter": current_chapter,
                            "region_type": "figure",
                            "chunk_index": fig_idx,
                            "bbox": [round(x, 1) for x in bbox],
                            "crop_path": crop_rel,
                            "caption": caption,
                            "ocr_rows": ocr_rows,
                            "vlm_description": "",
                            "attribution": ATTRIBUTION,
                        },
                    }
                )
                stats["figure"] += 1
                fig_idx += 1

            # 3. Extract clean Body Text blocks (excluding running header/footer & table areas).
            body_parts: list[str] = []
            for bx0, by0, bx1, by1, btxt, _bno, btype in sorted(blocks, key=lambda b: (b[1], b[0])):
                if btype != 0:
                    continue
                if by0 < 42 or by1 > 745:
                    continue
                b_bbox = (float(bx0), float(by0), float(bx1), float(by1))
                if any(bbox_overlap_ratio(b_bbox, tb) > 0.35 for tb in table_bboxes):
                    continue
                clean_b = normalize_text(btxt)
                if clean_b and not re.match(r"^\d+\s+Chapter\s+\d+", clean_b):
                    body_parts.append(clean_b)

            page_body = "\n\n".join(body_parts)
            for txt_idx, piece in enumerate(chunk_sentences(page_body, max_size=850)):
                cid = make_chunk_id(pno, "text", txt_idx, piece)
                all_chunks.append(
                    {
                        "id": cid,
                        "text": piece,
                        "source_id": SOURCE_ID,
                        "source": SOURCE_URL,
                        "page": pno,
                        "metadata": {
                            "chapter": current_chapter,
                            "region_type": "text",
                            "chunk_index": txt_idx,
                            "attribution": ATTRIBUTION,
                        },
                    }
                )
                stats["text"] += 1

    manifest = {
        "source_id": SOURCE_ID,
        "source_url": SOURCE_URL,
        "attribution": ATTRIBUTION,
        "physical_page_range": [start_page, end_page],
        "included_page_count": len(included_pages),
        "included_pages": included_pages,
        "skipped_page_count": len(skipped_pages),
        "skipped_pages": skipped_pages,
        "chunk_counts": stats,
        "total_chunks": len(all_chunks),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "chroma_collection": ADV_COLLECTION,
        "vlm_model": VLM_MODEL,
    }
    return all_chunks, manifest


def _encode_resized_png(img_path: Path, max_side: int = 896) -> str:
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")


def vlm_describe_image(img_path: Path, caption: str, model: str = VLM_MODEL) -> str:
    b64 = _encode_resized_png(img_path, max_side=896)
    prompt = (
        "Be direct and concise (<=100 words, minimal reasoning). State ONLY factual visual details:\n"
        "- Chart/graph/pyramid: chart type, title, X/Y axes & units, legend, key values and trends.\n"
        "- Diagram/map: labeled zones, legend colors, and what regions/elements each color or label represents.\n"
        "- Photo/illustration: visible objects, people, signs/text, setting, and actions.\n"
        f"PDF Caption: {caption}"
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [b64]}],
        "stream": False,
        "keep_alive": "15m",
        "options": {"temperature": 0.0, "num_ctx": 8192, "num_predict": 4096},
    }
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=240) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"VLM HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise ConnectionError(f"Cannot connect to Ollama VLM at {OLLAMA_BASE_URL}: {exc.reason}") from exc
    msg = data.get("message", {})
    return (msg.get("content") or "").strip()


def caption_figures_with_vlm(chunks: list[dict[str, Any]], *, model: str = VLM_MODEL) -> None:
    fig_chunks = [c for c in chunks if c["metadata"].get("region_type") == "figure"]
    total = len(fig_chunks)
    for i, c in enumerate(fig_chunks, start=1):
        if c["metadata"].get("vlm_description"):
            continue
        crop_abs = ROOT / c["metadata"]["crop_path"]
        t0 = time.perf_counter()
        try:
            desc = vlm_describe_image(crop_abs, c["metadata"].get("caption", ""), model=model)
        except Exception as exc:  # noqa: BLE001
            c["metadata"]["vlm_error"] = f"{type(exc).__name__}: {exc}"
            print(f"[VLM {i}/{total}] p.{c['page']} ERROR: {exc}", flush=True)
            continue
        c["metadata"]["vlm_description"] = desc
        ocr_rows = c["metadata"].get("ocr_rows") or []
        ocr_block = "\n".join(ocr_rows) if ocr_rows else "(none)"
        c["text"] = (
            f"[FIGURE | Page {c['page']} | Chapter {c['metadata']['chapter']}]\n"
            f"Caption: {c['metadata'].get('caption', '')}\n"
            f"VLM visual summary: {desc}\n"
            f"OCR extracted rows:\n{ocr_block}"
        )
        # Checkpoint every 5 figures so progress is preserved cleanly.
        if i % 5 == 0 or i == total:
            ADV_CHUNKS_PATH.write_text(
                json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        print(f"[VLM {i}/{total}] p.{c['page']} ({time.perf_counter()-t0:.1f}s, chars={len(desc)})", flush=True)


def build_chroma_index(chunks: list[dict[str, Any]]) -> float:
    import chromadb
    from fastembed import TextEmbedding

    t0 = time.perf_counter()
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    ADV_CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(ADV_CHROMA_DIR))
    try:
        client.delete_collection(ADV_COLLECTION)
    except Exception:
        pass
    col = client.get_or_create_collection(
        name=ADV_COLLECTION,
        metadata={"hnsw:space": "cosine", "dimensions": EMBEDDING_DIM},
    )
    embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
    vectors = [v.tolist() for v in embedder.embed([c["text"] for c in chunks], batch_size=32)]
    for offset in range(0, len(chunks), 250):
        batch = chunks[offset : offset + 250]
        col.upsert(
            ids=[c["id"] for c in batch],
            embeddings=vectors[offset : offset + 250],
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "document_id": c["source_id"],
                    "source": c["source"],
                    "page": int(c["page"]),
                    "chapter": str(c["metadata"]["chapter"]),
                    "region_type": str(c["metadata"]["region_type"]),
                    "crop_path": str(c["metadata"].get("crop_path", "")),
                    "caption": str(c["metadata"].get("caption", ""))[:400],
                }
                for c in batch
            ],
        )
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest 100+ page multimodal PDF for Day 7 Advanced.")
    parser.add_argument("--start-page", type=int, default=281)
    parser.add_argument("--end-page", type=int, default=495)
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--caption-vlm", action="store_true")
    parser.add_argument("--index", action="store_true")
    args = parser.parse_args()

    if args.skip_extract and ADV_CHUNKS_PATH.exists():
        chunks = json.loads(ADV_CHUNKS_PATH.read_text(encoding="utf-8"))
        manifest = json.loads(ADV_MANIFEST_PATH.read_text(encoding="utf-8"))
    else:
        chunks, manifest = extract_corpus(args.start_page, args.end_page)
        ADV_CHUNKS_PATH.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        ADV_MANIFEST_PATH.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"Extracted {manifest['included_page_count']} pages, "
            f"{manifest['total_chunks']} chunks ({manifest['chunk_counts']}), "
            f"skipped {manifest['skipped_page_count']} pages.",
            flush=True,
        )

    if args.caption_vlm:
        caption_figures_with_vlm(chunks, model=VLM_MODEL)
        ADV_CHUNKS_PATH.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    if args.index:
        elapsed = build_chroma_index(chunks)
        manifest["index_seconds"] = round(elapsed, 2)
        ADV_MANIFEST_PATH.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Indexed {len(chunks)} chunks into {ADV_COLLECTION} in {elapsed:.2f}s", flush=True)


if __name__ == "__main__":
    main()
