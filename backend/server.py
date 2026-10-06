"""FastAPI wrapper for TonerHound's resolve() API.

Run with: uvicorn backend.server:app --reload --port 8000
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import tonerhound

app = FastAPI(title="TonerHound API", version="0.3.1")

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

MAX_FILE_SIZE_MB = 25


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


@app.post("/api/verify", response_model=VerifyResponse)
async def verify(
    file: UploadFile = File(...),
    extraction: str = Form(...),
) -> VerifyResponse:
    import json

    # Validate file size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail={"error": "file_too_large", "max_mb": MAX_FILE_SIZE_MB},
        )

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
    results = []
    for res in raw_results:
        bbox_list = None
        if res.bbox is not None:
            bbox_list = [res.bbox.x, res.bbox.y, res.bbox.width, res.bbox.height]

        status_map = {
            "exact": "VERIFIED",
            "normalized": "VERIFIED",
            "fuzzy": "VERIFIED",
            "multi_region": "VERIFIED",
            "ambiguous": "MISMATCH",
            "derived": "MISMATCH",
            "not_found": "HALLUCINATION",
        }

        results.append(
            FieldResult(
                field=res.field,
                value=res.value,
                status=status_map.get(res.status.value, "HALLUCINATION"),
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
