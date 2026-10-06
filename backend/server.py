"""FastAPI wrapper for TonerHound's resolve() API.

Run with: uvicorn backend.server:app --reload --port 8000
"""

from __future__ import annotations

import json
import os
import tempfile
import time
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
            raw_results = tonerhound.resolve(
                document=Path(tmp.name),
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
    for res in raw_results:
        bbox_list = None
        if res.bbox is not None:
            bbox_list = [res.bbox.x, res.bbox.y, res.bbox.width, res.bbox.height]

        results.append(
            FieldResult(
                field=res.field,
                value=res.value,
                status=STATUS_MAP.get(res.status.value, "HALLUCINATION"),
                page=res.page,
                bbox=bbox_list,
                is_grounded=res.is_grounded,
                matched_text=res.matched_text,
                confidence=res.confidence,
            )
        )

    return VerifyResponse(
        results=results,
        meta={
            "fields_processed": len(results),
            "duration_ms": duration_ms,
        },
    )


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
