"""FastAPI wrapper for TonerHound's resolve() API.

Run with: uvicorn backend.server:app --reload --port 8000
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

import tonerhound

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="TonerHound API", version="0.3.1")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:3001",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def read_capped(file: UploadFile, max_bytes: int) -> bytes:
    """Stream-read an upload in 1 MB chunks, rejecting early if oversized.

    Unlike ``await file.read()``, this never holds more than *max_bytes* + 1 MB
    in memory — it raises 413 the instant the cap is crossed, before the full
    payload is buffered.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)  # 1 MB at a time
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "file_too_large",
                    "max_mb": max_bytes // (1024 * 1024),
                },
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _bbox_overlap(b1: list[float], b2_bbox: Any) -> float:
    """Calculate vertical/horizontal intersection area over line area."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2_bbox.x, b2_bbox.y, b2_bbox.width, b2_bbox.height
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    line_area = w2 * h2
    return inter_area / line_area if line_area > 0 else 0.0


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class FieldResult(BaseModel):
    field: str
    value: Any
    status: str
    page: int | None
    bbox: list[float] | None
    is_grounded: bool
    matched_text: str | None = None
    confidence: float = 0.0


class VerifyResponse(BaseModel):
    results: list[FieldResult]
    meta: dict[str, Any]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/api/verify", response_model=VerifyResponse)
@limiter.limit("10/minute")
async def verify(
    request: Request,
    file: UploadFile = File(...),
    extraction: str = Form(...),
) -> VerifyResponse:
    # Stream-read with size cap — rejects before OOM
    contents = await read_capped(file, MAX_FILE_BYTES)

    # Parse extraction JSON
    try:
        extraction_data = json.loads(extraction)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_json", "message": str(e)},
        )

    # Normalize input: accept both list and dict formats
    if isinstance(extraction_data, dict):
        fields = [
            {"field": k, "value": v} for k, v in extraction_data.items()
        ]
    elif isinstance(extraction_data, list):
        fields = extraction_data
    else:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_json", "message": "Expected object or array"},
        )

    # Write PDF to temp file and resolve
    start = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(contents)
        tmp.flush()

        try:
            doc_index = tonerhound.DocumentIndex.from_pdf(Path(tmp.name))
            raw_results = tonerhound.resolve(
                document=doc_index,
                extraction=fields,
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={"error": "internal", "message": str(e)},
            )

    duration_ms = int((time.perf_counter() - start) * 1000)

    # Ensure list
    if not isinstance(raw_results, list):
        raw_results = [raw_results]

    # Map to response
    STATUS_MAP = {
        "exact": "VERIFIED",
        "normalized": "VERIFIED",
        "fuzzy": "VERIFIED",
        "multi_region": "VERIFIED",
        "ambiguous": "MISMATCH",
        "derived": "MISMATCH",
        "not_found": "HALLUCINATION",
    }

    results = []
    grounded_results = []
    for res in raw_results:
        bbox_list = None
        if res.bbox is not None:
            bbox_list = [res.bbox.x, res.bbox.y, res.bbox.width, res.bbox.height]

        field_res = FieldResult(
            field=res.field,
            value=res.value,
            status=STATUS_MAP.get(res.status.value, "HALLUCINATION"),
            page=res.page,
            bbox=bbox_list,
            is_grounded=res.is_grounded,
            matched_text=res.matched_text,
            confidence=res.confidence,
        )
        results.append(field_res)
        if res.is_grounded and res.page is not None:
            grounded_results.append(res)

    # Compute document coverage & identify unmapped lines
    boxes_by_page: dict[int, list[tuple[list[float], str | None, str]]] = defaultdict(list)
    for g in grounded_results:
        val_str = str(g.value).lower().strip() if g.value else ""
        if g.regions:
            for reg in g.regions:
                boxes_by_page[g.page].append(([reg.x, reg.y, reg.width, reg.height], g.matched_text, val_str))
        elif g.bbox is not None:
            boxes_by_page[g.page].append(([g.bbox.x, g.bbox.y, g.bbox.width, g.bbox.height], g.matched_text, val_str))

    total_lines = 0
    mapped_lines = 0
    unmapped_lines = []

    for page in doc_index.pages:
        page_boxes = boxes_by_page.get(page.page_number, [])
        for line_idx, line in enumerate(page.lines):
            line_txt = line.text.strip()
            if not line_txt:
                continue
            total_lines += 1
            line_lower = line_txt.lower()
            is_mapped = False
            for gb, matched_txt, val_str in page_boxes:
                if (
                    _bbox_overlap(gb, line.bbox) > 0.25
                    or (matched_txt and line_lower in matched_txt.lower())
                    or (val_str and line_lower in val_str)
                ):
                    is_mapped = True
                    break

            if is_mapped:
                mapped_lines += 1
            else:
                if len(unmapped_lines) < 200:
                    unmapped_lines.append({
                        "id": f"unmapped_p{page.page_number}_{line_idx}",
                        "page": page.page_number,
                        "text": line_txt,
                        "bbox": [line.bbox.x, line.bbox.y, line.bbox.width, line.bbox.height],
                    })

    coverage_pct = round((mapped_lines / total_lines) * 100, 1) if total_lines > 0 else 100.0

    return VerifyResponse(
        results=results,
        meta={
            "fields_processed": len(results),
            "duration_ms": duration_ms,
            "coverage": {
                "total_lines": total_lines,
                "mapped_lines": mapped_lines,
                "coverage_percent": coverage_pct,
                "unmapped_lines": unmapped_lines,
            },
        },
    )


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
