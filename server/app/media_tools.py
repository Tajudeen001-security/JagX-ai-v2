"""JagX v13 — multimodal tools for the agent loop.

These register into the same ``TOOLS`` registry as the core tools, so the agent
can call them autonomously mid-run exactly like ``web_search`` or ``run_code``.

Every tool here is local and free. Nothing in this file contacts a company's
image or vision API, and no API key is read anywhere.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from . import db, media
from .config import settings

logger = logging.getLogger("jagx.media_tools")


def _store(user_id: str, result: dict, *, source: str, filename: str = "", parent_id: str = "", meta: dict | None = None) -> dict:
    """Persist a produced image to disk + the media table."""
    aid, path = media.save_bytes(user_id, result["data"], result.get("ext", ".png"))
    row = db.add_media(
        user_id,
        aid,
        kind=result.get("kind", "image"),
        source=source,
        filename=filename or f"{aid}.png",
        ext=result.get("ext", ".png"),
        mime="image/png",
        size_bytes=len(result["data"]),
        width=int(result.get("width") or 0),
        height=int(result.get("height") or 0),
        prompt=result.get("prompt", ""),
        operation=result.get("operation", ""),
        backend=result.get("backend", ""),
        parent_id=parent_id,
        meta={**(meta or {}), "seed": result.get("seed"), "duration_ms": result.get("duration_ms")},
    )
    return {"asset": row, "path": path}


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------
def _generate_image(
    user_id: str,
    prompt: str,
    width: int = 768,
    height: int = 768,
    style: str = "",
    steps: int = 24,
    seed: int = -1,
) -> str:
    try:
        result = media.generate_image(
            prompt,
            width=width,
            height=height,
            steps=steps,
            style=style,
            seed=None if int(seed) < 0 else int(seed),
        )
    except Exception as exc:  # noqa: BLE001
        return f"Error: image generation failed: {type(exc).__name__}: {exc}"

    stored = _store(user_id, result, source="generated")
    row = stored["asset"]
    return json.dumps(
        {
            "ok": True,
            "asset_id": row["id"],
            "url": f"/api/media/{row['id']}/raw",
            "width": row["width"],
            "height": row["height"],
            "backend": row["backend"],
            "seed": result.get("seed"),
            "note": (
                "Image created and saved to the gallery. Reference it by asset_id in "
                "edit_image, or show it to the user with its url."
            ),
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# edit
# ---------------------------------------------------------------------------
def _edit_image(
    user_id: str,
    asset_id: str,
    operation: str,
    prompt: str = "",
    box: str = "",
    direction: str = "right",
    amount: int = 128,
    factor: float = 2.0,
    style: str = "",
    tolerance: int = 32,
    degrees: float = 90.0,
    width: int = 0,
    height: int = 0,
) -> str:
    src = media.resolve_asset(user_id, asset_id)
    if src is None:
        return f"Error: no image with asset_id '{asset_id}'. Use list_media to see what exists."

    parsed_box = None
    if box:
        try:
            parsed_box = [float(p) for p in box.replace(" ", "").split(",") if p]
        except ValueError:
            return "Error: box must be four comma-separated numbers, e.g. '10,10,200,200'."

    try:
        result = media.edit_image(
            src,
            operation,
            prompt=prompt,
            box=parsed_box,
            direction=direction,
            amount=amount,
            factor=factor,
            style=style,
            tolerance=tolerance,
            degrees=degrees,
            width=width or None,
            height=height or None,
        )
    except Exception as exc:  # noqa: BLE001
        return f"Error: {operation} failed: {type(exc).__name__}: {exc}"

    stored = _store(user_id, result, source="edited", parent_id=asset_id)
    row = stored["asset"]
    return json.dumps(
        {
            "ok": True,
            "asset_id": row["id"],
            "parent_id": asset_id,
            "url": f"/api/media/{row['id']}/raw",
            "operation": operation,
            "backend": row["backend"],
            "width": row["width"],
            "height": row["height"],
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# understand an image
# ---------------------------------------------------------------------------
def _understand_image(user_id: str, asset_id: str = "", question: str = "") -> str:
    if not asset_id:
        return "Error: asset_id is required. Use list_media to find one."
    path = media.resolve_asset(user_id, asset_id)
    if path is None:
        return f"Error: no image with asset_id '{asset_id}'."
    try:
        result = media.understand_image(path, question)
    except Exception as exc:  # noqa: BLE001
        return f"Error: image understanding failed: {type(exc).__name__}: {exc}"

    return json.dumps(
        {
            "ok": True,
            "asset_id": asset_id,
            "backend": result["backend"],
            "description": result["description"],
            "ocr_text": result["ocr_text"][:4000],
            "stats": result["stats"],
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# read a document
# ---------------------------------------------------------------------------
def _read_document(user_id: str, asset_id: str = "", max_chars: int = 12000) -> str:
    if not asset_id:
        return "Error: asset_id is required. Use list_media to find one."
    path = media.resolve_asset(user_id, asset_id)
    if path is None:
        return f"Error: no file with asset_id '{asset_id}'."
    try:
        result = media.read_document(path)
    except Exception as exc:  # noqa: BLE001
        return f"Error: document parsing failed: {type(exc).__name__}: {exc}"

    text = result["text"] or ""
    limit = max(500, min(int(max_chars or 12000), settings.doc_max_chars))
    truncated = len(text) > limit
    return json.dumps(
        {
            "ok": True,
            "asset_id": asset_id,
            "kind": result["kind"],
            "description": result["description"],
            "meta": result["meta"],
            "text": text[:limit],
            "truncated": truncated,
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# gallery
# ---------------------------------------------------------------------------
def _list_media(user_id: str, kind: str = "", limit: int = 30) -> str:
    rows = db.list_media(user_id, kind=kind, limit=max(1, min(int(limit or 30), 100)))
    if not rows:
        return json.dumps({"ok": True, "count": 0, "items": [], "note": "The gallery is empty."})
    items = [
        {
            "asset_id": r["id"],
            "kind": r["kind"],
            "source": r["source"],
            "filename": r["filename"],
            "width": r["width"],
            "height": r["height"],
            "prompt": r["prompt"][:160],
            "operation": r["operation"],
            "backend": r["backend"],
            "url": f"/api/media/{r['id']}/raw",
        }
        for r in rows
    ]
    return json.dumps({"ok": True, "count": len(items), "items": items}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------
def register(register_tool, tool_cls) -> None:
    """Called by tools.py to add the multimodal surface to the registry.

    ``register_tool`` is the registry's ``_register`` function and ``tool_cls``
    is its ``Tool`` dataclass — passed in so this module never has to import
    ``tools`` and create a circular import.
    """
    def add(**kwargs) -> None:
        register_tool(tool_cls(**kwargs))

    add(
        name="generate_image",
        description=(
            "Create an image from a text prompt, entirely on this machine (no API key). "
            "Use for illustrations, diagrams, mockups, icons, concept art or any visual the "
            "user asks for. Returns an asset_id you can pass to edit_image or understand_image."
        ),
        parameters={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "What to draw. Be specific and visual."},
                "width": {"type": "integer", "description": "Pixels wide (64-1024).", "default": 768},
                "height": {"type": "integer", "description": "Pixels tall (64-1024).", "default": 768},
                "style": {"type": "string", "description": "Optional style hint, e.g. 'flat vector', 'photorealistic', 'isometric'."},
                "steps": {"type": "integer", "description": "Diffusion steps when a real model is installed.", "default": 24},
                "seed": {"type": "integer", "description": "Seed for reproducibility; -1 for a prompt-derived seed.", "default": -1},
            },
            "required": ["prompt"],
        },
        handler=_generate_image,
        needs_user=True,
        category="media",
    )

    add(
        name="edit_image",
        description=(
            "Modify an existing image locally. Operations: inpaint (fill a box), outpaint "
            "(extend the canvas), remove-background, upscale, style (apply a named look), "
            "transform (rotate/flip/resize/crop/grayscale). Returns a new asset_id."
        ),
        parameters={
            "type": "object",
            "properties": {
                "asset_id": {"type": "string", "description": "The image to edit."},
                "operation": {
                    "type": "string",
                    "description": "inpaint | outpaint | remove-background | upscale | style | transform",
                },
                "prompt": {"type": "string", "description": "What should appear in the edited region (inpaint)."},
                "box": {"type": "string", "description": "Region for inpaint as 'x0,y0,x1,y1' (pixels or 0-1 fractions)."},
                "direction": {"type": "string", "description": "Outpaint direction: right, left, up, down or all.", "default": "right"},
                "amount": {"type": "integer", "description": "Outpaint pixels to add.", "default": 128},
                "factor": {"type": "number", "description": "Upscale factor (1.1-4.0).", "default": 2.0},
                "style": {
                    "type": "string",
                    "description": "Style name: grayscale, sepia, invert, poster, sketch, warm, cool, vivid, soft, noir, blur, sharpen.",
                },
                "tolerance": {"type": "integer", "description": "Background-removal colour tolerance (4-160).", "default": 32},
                "degrees": {"type": "number", "description": "Rotation degrees for transform.", "default": 90},
                "width": {"type": "integer", "description": "Target width for transform resize.", "default": 0},
                "height": {"type": "integer", "description": "Target height for transform resize.", "default": 0},
            },
            "required": ["asset_id", "operation"],
        },
        handler=_edit_image,
        needs_user=True,
        category="media",
    )

    add(
        name="understand_image",
        description=(
            "Look at an image and describe it, or answer a question about it. Also extracts any "
            "text in the image (OCR). Use whenever the user uploads or generates an image and "
            "you need to reason about its contents."
        ),
        parameters={
            "type": "object",
            "properties": {
                "asset_id": {"type": "string", "description": "The image to inspect."},
                "question": {"type": "string", "description": "Optional specific question about the image."},
            },
            "required": ["asset_id"],
        },
        handler=_understand_image,
        needs_user=True,
        category="media",
    )

    add(
        name="read_document",
        description=(
            "Read an uploaded file and return its text: PDF, DOCX, XLSX, CSV, TSV, TXT, MD, JSON, "
            "code files, and images (via OCR). Use before summarising, analysing or answering "
            "questions about a document the user gave you."
        ),
        parameters={
            "type": "object",
            "properties": {
                "asset_id": {"type": "string", "description": "The uploaded file to read."},
                "max_chars": {"type": "integer", "description": "Max characters to return.", "default": 12000},
            },
            "required": ["asset_id"],
        },
        handler=_read_document,
        needs_user=True,
        category="media",
    )

    add(
        name="list_media",
        description=(
            "List the images and files in this user's gallery, newest first. Use to find an "
            "asset_id before editing, reading or describing something."
        ),
        parameters={
            "type": "object",
            "properties": {
                "kind": {"type": "string", "description": "Filter: image or document. Empty for all."},
                "limit": {"type": "integer", "description": "How many to return (1-100).", "default": 30},
            },
        },
        handler=_list_media,
        needs_user=True,
        category="media",
    )
