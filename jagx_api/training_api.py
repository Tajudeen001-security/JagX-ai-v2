"""Admin training / LoRA flywheel endpoints."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from jagx_api import auth
from jagx_api.config import settings
from jagx_api import lora_flywheel

router = APIRouter(prefix="/training", tags=["training"])


class ExportRequest(BaseModel):
    admin_secret: str
    quality_only: bool = True
    limit: int = Field(default=5000, ge=1, le=50000)
    dest: str = "jagx_sft_export.jsonl"


@router.get("/stats")
def training_stats(x_api_key: Optional[str] = Header(None, alias="x-api-key")):
    key = (x_api_key or "").strip()
    if not auth.is_valid_key(key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return lora_flywheel.stats()


@router.post("/export")
def training_export(req: ExportRequest):
    if not settings.admin_secret or req.admin_secret != settings.admin_secret:
        raise HTTPException(status_code=403, detail="Invalid admin secret")
    return lora_flywheel.export_jsonl(req.dest, quality_only=req.quality_only, limit=req.limit)
