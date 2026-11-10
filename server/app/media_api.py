"""JagX v13 — the multimodal HTTP surface.

Kept in its own router so the core API stays readable. Mounted by ``main.py``
under the same ``/api`` prefix, so the paths are identical to the rest of the
API: ``/api/media/...``.

Every endpoint is authenticated and scoped to the calling user: an asset id
belonging to someone else returns 404, never 403, so ids cannot be probed.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from . import db, media, security
from .config import settings

logger = logging.getLogger("jagx.media_api")

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# auth — imported lazily to avoid a circular import with api.py
# ---------------------------------------------------------------------------
def current_user(authorization: str = Header(default="")) -> dict:
    """Self-contained auth, so this router never depends on api.py."""
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    token = authorization.split(" ", 1)[1].strip()
    payload = security.decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    user = db.get_user(payload.get("sub", ""))
    if not user:
        raise HTTPException(status_code=401, detail="User no longer exists.")
    return user


# ---------------------------------------------------------------------------
# request models
# ---------------------------------------------------------------------------
class GenerateBody(BaseModel):
    prompt: str
    width: int = 768
    height: int = 768
    steps: int = 24
    seed: Optional[int] = None
    style: str = ""
    negative: str = ""


class EditBody(BaseModel):
    operation: str
    prompt: str = ""
    box: Optional[list[float]] = None
    direction: str = "right"
    amount: int = 128
    factor: float = 2.0
    style: str = ""
    tolerance: int = 32
    degrees: float = 90.0
    width: int = 0
    height: int = 0
    seed: Optional[int] = None


class UnderstandBody(BaseModel):
    question: str = ""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _media_row(user_id: str, asset_id: str) -> dict:
    row = db.get_media(user_id, asset_id)
    if not row:
        raise HTTPException(status_code=404, detail="No such asset.")
    return row


def _record(
    user_id: str,
    result: dict,
    *,
    source: str,
    filename: str = "",
    parent_id: str = "",
    meta: dict | None = None,
) -> dict:
    aid, _path = media.save_bytes(user_id, result["data"], result.get("ext", ".png"))
    return db.add_media(
        user_id,
        aid,
        kind=result.get("kind", "image"),
        source=source,
        filename=filename or f"{aid}{result.get('ext', '.png')}",
        ext=result.get("ext", ".png"),
        mime="image/png",
        size_bytes=len(result["data"]),
        width=int(result.get("width") or 0),
        height=int(result.get("height") or 0),
        prompt=result.get("prompt", ""),
        operation=result.get("operation", ""),
        backend=result.get("backend", ""),
        parent_id=parent_id,
        meta={
            **(meta or {}),
            "seed": result.get("seed"),
            "duration_ms": result.get("duration_ms"),
        },
    )


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------
@router.get("/media/capabilities")
def media_capabilities(user: dict = Depends(current_user)) -> dict:
    """Which tier is live, and the honest limits of each capability."""
    return media.describe()


@router.get("/media")
def list_media(kind: str = "", limit: int = 200, user: dict = Depends(current_user)) -> dict:
    return {
        "media": db.list_media(user["id"], kind=kind, limit=max(1, min(limit, 500))),
        "stats": db.media_stats(user["id"]),
    }


@router.post("/media/generate")
def media_generate(body: GenerateBody, user: dict = Depends(current_user)) -> dict:
    if not body.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required.")
    try:
        result = media.generate_image(
            body.prompt,
            width=body.width,
            height=body.height,
            steps=body.steps,
            seed=body.seed,
            style=body.style,
            negative=body.negative,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("generation failed")
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc
    return _record(user["id"], result, source="generated")


@router.post("/media/upload")
async def media_upload(file: UploadFile = File(...), user: dict = Depends(current_user)) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(data) > settings.upload_max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {settings.upload_max_bytes // 1_000_000} MB).",
        )

    name = file.filename or "upload"
    ext = (name.rsplit(".", 1)[-1] if "." in name else "").lower()
    is_image = f".{ext}" in media.IMAGE_EXTS if ext else False

    aid, path = media.save_bytes(user["id"], data, f".{ext}" if ext else ".bin")
    width = height = 0
    kind = "image" if is_image else "document"
    if is_image:
        try:
            img = media._open_image(path)
            width, height = img.width, img.height
        except Exception:  # noqa: BLE001
            kind = "document"

    return db.add_media(
        user["id"],
        aid,
        kind=kind,
        source="upload",
        filename=name,
        ext=f".{ext}" if ext else ".bin",
        mime=file.content_type or "application/octet-stream",
        size_bytes=len(data),
        width=width,
        height=height,
        meta={"content_type": file.content_type or ""},
    )


@router.get("/media/{asset_id}")
def media_detail(asset_id: str, user: dict = Depends(current_user)) -> dict:
    return _media_row(user["id"], asset_id)


@router.get("/media/{asset_id}/raw")
def media_raw(asset_id: str, user: dict = Depends(current_user)) -> Response:
    row = _media_row(user["id"], asset_id)
    path = media.resolve_asset(user["id"], asset_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Asset file is missing from disk.")
    return Response(
        content=path.read_bytes(),
        media_type=row.get("mime") or "application/octet-stream",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.get("/media/{asset_id}/thumb")
def media_thumb(asset_id: str, size: int = 320, user: dict = Depends(current_user)) -> Response:
    _media_row(user["id"], asset_id)
    path = media.resolve_asset(user["id"], asset_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Asset file is missing from disk.")
    try:
        data = media.thumbnail(path, size=max(48, min(size, 1024)))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Cannot thumbnail this asset: {exc}") from exc
    return Response(
        content=data,
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.post("/media/{asset_id}/edit")
def media_edit(asset_id: str, body: EditBody, user: dict = Depends(current_user)) -> dict:
    _media_row(user["id"], asset_id)
    path = media.resolve_asset(user["id"], asset_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Asset file is missing from disk.")
    try:
        result = media.edit_image(
            path,
            body.operation,
            prompt=body.prompt,
            box=body.box,
            direction=body.direction,
            amount=body.amount,
            factor=body.factor,
            style=body.style,
            tolerance=body.tolerance,
            degrees=body.degrees,
            width=body.width or None,
            height=body.height or None,
            seed=body.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("edit failed")
        raise HTTPException(status_code=500, detail=f"Edit failed: {exc}") from exc
    return _record(user["id"], result, source="edited", parent_id=asset_id)


@router.post("/media/{asset_id}/understand")
def media_understand(
    asset_id: str, body: UnderstandBody, user: dict = Depends(current_user)
) -> dict:
    _media_row(user["id"], asset_id)
    path = media.resolve_asset(user["id"], asset_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Asset file is missing from disk.")
    try:
        return media.understand_image(path, body.question)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Understanding failed: {exc}") from exc


@router.post("/media/{asset_id}/read")
def media_read(asset_id: str, user: dict = Depends(current_user)) -> dict:
    _media_row(user["id"], asset_id)
    path = media.resolve_asset(user["id"], asset_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Asset file is missing from disk.")
    try:
        return media.read_document(path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Parsing failed: {exc}") from exc


@router.delete("/media/{asset_id}")
def media_delete(asset_id: str, user: dict = Depends(current_user)) -> dict:
    _media_row(user["id"], asset_id)
    path = media.resolve_asset(user["id"], asset_id)
    if path is not None:
        try:
            path.unlink()
        except OSError as exc:
            logger.warning("could not remove %s: %s", path, exc)
    db.delete_media(user["id"], asset_id)
    return {"ok": True, "deleted": asset_id}
