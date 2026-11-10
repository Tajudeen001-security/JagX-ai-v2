"""JagX v13 — the multimodal engine.

Everything in this module runs **locally and for free**. There is no call to any
company's image API anywhere in this file, and no API key is required for any
code path.

Three capability groups:

    generate   text -> image      local diffusers (SD / SDXL / FLUX) or ComfyUI
    edit       image -> image     inpaint, outpaint, background removal,
                                  upscale, style transfer, transforms
    understand image/doc -> text  local VLM, OCR, and document parsing

**The honest design rule.** A diffusion model is a multi-gigabyte download and
needs a GPU to be pleasant. So every capability here has two tiers:

    tier 1  the real model, when the operator has installed it
            (``pip install diffusers torch`` + a local checkpoint, or ComfyUI)
    tier 2  a deterministic, dependency-light fallback that always works

The fallback is not a fake: it produces a real, valid PNG derived from the
prompt (procedural gradient art, seeded by a hash of the text) so the whole
product — agent loop, tool registry, API, UI, gallery — runs end to end on a
laptop with no GPU and no model. When a real model is present, the same code
path silently upgrades to it. ``capabilities()`` reports which tier is live so
the UI can tell the truth about it.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import math
import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .config import settings

logger = logging.getLogger("jagx.media")

# ---------------------------------------------------------------------------
# optional heavy dependencies — imported lazily, never required
# ---------------------------------------------------------------------------
_PIL = None
_NP = None


def _pil():
    global _PIL
    if _PIL is None:
        try:
            from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps  # noqa: F401

            _PIL = __import__("PIL", fromlist=["Image"])
        except ImportError:  # pragma: no cover - Pillow is a hard dep in practice
            _PIL = False
    return _PIL or None


def _np():
    global _NP
    if _NP is None:
        try:
            import numpy

            _NP = numpy
        except ImportError:
            _NP = False
    return _NP or None


def _has(module: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
DOC_EXTS = {".pdf", ".docx", ".txt", ".md", ".csv", ".tsv", ".json", ".xlsx", ".xls", ".log", ".py", ".js", ".ts", ".html", ".xml", ".yaml", ".yml"}


def media_root() -> Path:
    root = Path(settings.data_dir) / "media"
    root.mkdir(parents=True, exist_ok=True)
    return root


def user_dir(user_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", user_id or "anon")[:64]
    d = media_root() / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_asset_id() -> str:
    return f"med_{uuid.uuid4().hex[:16]}"


def asset_path(user_id: str, asset_id: str, ext: str = ".png") -> Path:
    return user_dir(user_id) / f"{asset_id}{ext}"


def resolve_asset(user_id: str, asset_id: str) -> Optional[Path]:
    """Find an asset on disk by id, whatever extension it was stored with."""
    d = user_dir(user_id)
    safe = re.sub(r"[^A-Za-z0-9_-]", "", asset_id or "")
    if not safe:
        return None
    for p in d.glob(f"{safe}.*"):
        if p.is_file():
            return p
    return None


def save_bytes(user_id: str, data: bytes, ext: str = ".png") -> tuple[str, Path]:
    aid = new_asset_id()
    p = asset_path(user_id, aid, ext)
    p.write_bytes(data)
    return aid, p


def save_upload(user_id: str, filename: str, data: bytes) -> tuple[str, Path, str]:
    ext = Path(filename or "").suffix.lower() or ".bin"
    aid, p = save_bytes(user_id, data, ext)
    return aid, p, ext


# ---------------------------------------------------------------------------
# capability probe
# ---------------------------------------------------------------------------
@dataclass
class Capabilities:
    image_generation: str = "fallback"
    image_editing: str = "fallback"
    vision: str = "fallback"
    ocr: bool = False
    documents: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "image_generation": self.image_generation,
            "image_editing": self.image_editing,
            "vision": self.vision,
            "ocr": self.ocr,
            "documents": self.documents,
            "notes": self.notes,
            "local_only": True,
            "requires_api_key": False,
        }


def _comfyui_reachable() -> bool:
    if not settings.comfyui_url:
        return False
    try:
        import httpx

        with httpx.Client(timeout=2.0) as c:
            r = c.get(f"{settings.comfyui_url.rstrip('/')}/system_stats")
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def _diffusion_available() -> bool:
    return _has("torch") and _has("diffusers")


def _vlm_available() -> bool:
    """A local vision-language model, served over an OpenAI-compatible endpoint."""
    if not settings.vision_model:
        return False
    try:
        import httpx

        base = settings.vision_base_url or settings.ollama_base_url
        with httpx.Client(timeout=2.0) as c:
            r = c.get(f"{base.rstrip('/').removesuffix('/v1')}/api/tags")
        if r.status_code != 200:
            return False
        names = [m.get("name", "") for m in (r.json().get("models") or [])]
        want = settings.vision_model.split(":")[0]
        return any(n.split(":")[0] == want for n in names)
    except Exception:  # noqa: BLE001
        return False


def capabilities() -> Capabilities:
    caps = Capabilities()

    if _comfyui_reachable():
        caps.image_generation = "comfyui"
        caps.image_editing = "comfyui"
        caps.notes.append(f"ComfyUI reachable at {settings.comfyui_url}.")
    elif _diffusion_available():
        caps.image_generation = "diffusers"
        caps.image_editing = "diffusers"
        caps.notes.append(
            f"diffusers + torch present; model '{settings.image_model}' loads on first use."
        )
    else:
        caps.notes.append(
            "No diffusion runtime installed — using the built-in procedural renderer. "
            "Install one with: pip install torch diffusers accelerate  "
            "(then set JAGX_IMAGE_MODEL to a local SD/SDXL/FLUX checkpoint)."
        )

    if _vlm_available():
        caps.vision = "vlm"
        caps.notes.append(f"Local vision model '{settings.vision_model}' detected.")
    else:
        caps.notes.append(
            "No local vision model detected — image understanding uses OCR + "
            "structural analysis. Add one with: ollama pull llava  "
            "(then set JAGX_VISION_MODEL=llava)."
        )

    caps.ocr = bool(shutil.which("tesseract"))
    if not caps.ocr:
        caps.notes.append("tesseract not found — OCR disabled (apt-get install tesseract-ocr).")

    docs = [".txt", ".md", ".csv", ".tsv", ".json", ".log", ".py", ".js", ".ts", ".html", ".xml", ".yaml", ".yml"]
    if _has("pypdf") or shutil.which("pdftotext"):
        docs.append(".pdf")
    if _has("docx"):
        docs.append(".docx")
    if _has("openpyxl"):
        docs.append(".xlsx")
    if caps.ocr:
        docs.append("images (OCR)")
    caps.documents = sorted(docs)
    return caps


# ---------------------------------------------------------------------------
# procedural renderer — the always-available tier
# ---------------------------------------------------------------------------
def _seed_from(text: str) -> int:
    return int(hashlib.sha256((text or "jagx").encode("utf-8")).hexdigest()[:12], 16)


def _palette(seed: int, n: int = 5) -> list[tuple[int, int, int]]:
    """A harmonious palette derived deterministically from the seed."""
    import colorsys

    base_hue = (seed % 360) / 360.0
    scheme = (seed >> 8) % 3
    hues = {
        0: [0.0, 0.08, 0.5, 0.58, 0.75],          # analogous
        1: [0.0, 0.33, 0.5, 0.66, 0.83],          # triadic-ish
        2: [0.0, 0.5, 0.25, 0.75, 0.12],          # split complement
    }[scheme]
    out = []
    for i, h in enumerate(hues[:n]):
        hue = (base_hue + h) % 1.0
        sat = 0.45 + ((seed >> (i * 3)) % 40) / 100.0
        val = 0.35 + ((seed >> (i * 5)) % 55) / 100.0
        r, g, b = colorsys.hsv_to_rgb(hue, min(sat, 0.95), min(val, 0.98))
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def _procedural_image(prompt: str, width: int, height: int, style: str = "") -> bytes:
    """Render a real PNG from the prompt text.

    This is deliberately *not* a placeholder box. It is a deterministic piece of
    generative art: the prompt seeds a palette, a flow field and a set of
    geometric motifs, so the same prompt always yields the same image and
    different prompts yield visibly different ones. It gives the agent a genuine
    image artifact to reason about, embed, edit and display when no diffusion
    model is installed.
    """
    PIL = _pil()
    if PIL is None:  # pragma: no cover
        raise RuntimeError("Pillow is required for image rendering.")

    from PIL import Image, ImageDraw, ImageFilter

    seed = _seed_from(f"{prompt}|{style}|{width}x{height}")
    pal = _palette(seed)
    img = Image.new("RGB", (width, height), pal[0])
    draw = ImageDraw.Draw(img, "RGBA")

    # 1) vertical gradient wash
    for y in range(height):
        t = y / max(1, height - 1)
        c0, c1 = pal[0], pal[1]
        col = (
            int(c0[0] + (c1[0] - c0[0]) * t),
            int(c0[1] + (c1[1] - c0[1]) * t),
            int(c0[2] + (c1[2] - c0[2]) * t),
        )
        draw.line([(0, y), (width, y)], fill=col)

    # 2) flow-field ribbons
    rnd = seed
    def nxt() -> float:
        nonlocal rnd
        rnd = (rnd * 1103515245 + 12345) & 0x7FFFFFFF
        return rnd / 0x7FFFFFFF

    ribbons = 14 + (seed % 10)
    for i in range(ribbons):
        col = pal[(i + 2) % len(pal)]
        alpha = 40 + int(nxt() * 90)
        pts = []
        x = nxt() * width
        y = nxt() * height
        angle = nxt() * math.tau
        for _ in range(60):
            pts.append((x, y))
            angle += (nxt() - 0.5) * 0.9
            x += math.cos(angle) * (width / 45)
            y += math.sin(angle) * (height / 45)
        if len(pts) > 1:
            draw.line(pts, fill=(*col, alpha), width=1 + int(nxt() * 5), joint="curve")

    # 3) geometric motifs
    for i in range(6 + (seed % 7)):
        col = pal[(i + 1) % len(pal)]
        cx, cy = nxt() * width, nxt() * height
        r = (0.04 + nxt() * 0.16) * min(width, height)
        alpha = 30 + int(nxt() * 70)
        kind = int(nxt() * 3)
        box = [cx - r, cy - r, cx + r, cy + r]
        if kind == 0:
            draw.ellipse(box, outline=(*col, alpha), width=2 + int(nxt() * 4))
        elif kind == 1:
            draw.rectangle(box, outline=(*col, alpha), width=2 + int(nxt() * 3))
        else:
            draw.polygon(
                [(cx, cy - r), (cx + r, cy + r * 0.7), (cx - r, cy + r * 0.7)],
                outline=(*col, alpha),
            )

    # 4) soft bloom + grain
    img = img.filter(ImageFilter.GaussianBlur(radius=max(0.4, min(width, height) / 400)))
    if _np() is not None:
        import numpy as np

        arr = np.asarray(img).astype("int16")
        noise = np.random.default_rng(seed % (2**32)).integers(-9, 10, arr.shape, dtype="int16")
        arr = np.clip(arr + noise, 0, 255).astype("uint8")
        img = Image.fromarray(arr, "RGB")

    # 5) vignette
    vig = Image.new("L", (width, height), 0)
    vd = ImageDraw.Draw(vig)
    vd.ellipse([-width * 0.25, -height * 0.25, width * 1.25, height * 1.25], fill=255)
    vig = vig.filter(ImageFilter.GaussianBlur(radius=min(width, height) / 8))
    black = Image.new("RGB", (width, height), (0, 0, 0))
    img = Image.composite(img, black, vig)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# tier-1 diffusion backends
# ---------------------------------------------------------------------------
_DIFFUSION_PIPE: Any = None


def _load_diffusion() -> Any:
    """Load a local diffusers pipeline once per process."""
    global _DIFFUSION_PIPE
    if _DIFFUSION_PIPE is not None:
        return _DIFFUSION_PIPE
    import torch
    from diffusers import AutoPipelineForText2Image

    model = settings.image_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    logger.info("loading diffusion model %s on %s", model, device)
    pipe = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=dtype)
    pipe = pipe.to(device)
    if device == "cpu":
        pipe.enable_attention_slicing()
    _DIFFUSION_PIPE = pipe
    return pipe


def _diffusers_generate(prompt: str, width: int, height: int, steps: int, seed: int) -> bytes:
    import torch

    pipe = _load_diffusion()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=settings.image_guidance,
        generator=gen,
    )
    img = out.images[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _comfyui_generate(prompt: str, width: int, height: int, steps: int, seed: int) -> bytes:
    """Minimal ComfyUI /prompt client using the default txt2img graph."""
    import httpx

    base = settings.comfyui_url.rstrip("/")
    graph = {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed, "steps": steps, "cfg": settings.image_guidance,
                "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
                "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0],
            },
        },
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": settings.comfyui_checkpoint}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": settings.image_negative, "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "jagx", "images": ["8", 0]}},
    }
    with httpx.Client(timeout=settings.image_timeout) as c:
        r = c.post(f"{base}/prompt", json={"prompt": graph})
        r.raise_for_status()
        pid = r.json()["prompt_id"]
        deadline = time.time() + settings.image_timeout
        while time.time() < deadline:
            h = c.get(f"{base}/history/{pid}")
            if h.status_code == 200 and pid in h.json():
                entry = h.json()[pid]
                for node in (entry.get("outputs") or {}).values():
                    for im in node.get("images") or []:
                        q = c.get(
                            f"{base}/view",
                            params={"filename": im["filename"], "subfolder": im.get("subfolder", ""), "type": im.get("type", "output")},
                        )
                        if q.status_code == 200:
                            return q.content
                raise RuntimeError("ComfyUI produced no image")
            time.sleep(0.5)
    raise RuntimeError("ComfyUI timed out")


# ---------------------------------------------------------------------------
# public: generate
# ---------------------------------------------------------------------------
def generate_image(
    prompt: str,
    *,
    width: int = 768,
    height: int = 768,
    steps: int = 24,
    seed: Optional[int] = None,
    style: str = "",
    negative: str = "",
) -> dict:
    """Text -> image. Returns a dict with the PNG bytes and provenance."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("A prompt is required.")
    width = max(64, min(int(width or 768), settings.image_max_side))
    height = max(64, min(int(height or 768), settings.image_max_side))
    steps = max(4, min(int(steps or 24), 80))
    seed = int(seed) if seed is not None else _seed_from(prompt) % (2**31)

    full_prompt = f"{prompt}, {style}" if style else prompt
    backend = "procedural"
    started = time.time()

    if _comfyui_reachable():
        try:
            data = _comfyui_generate(full_prompt, width, height, steps, seed)
            backend = "comfyui"
        except Exception as exc:  # noqa: BLE001
            logger.warning("comfyui generation failed, falling back: %s", exc)
            data = _procedural_image(prompt, width, height, style)
    elif _diffusion_available():
        try:
            data = _diffusers_generate(full_prompt, width, height, steps, seed)
            backend = "diffusers"
        except Exception as exc:  # noqa: BLE001
            logger.warning("diffusers generation failed, falling back: %s", exc)
            data = _procedural_image(prompt, width, height, style)
    else:
        data = _procedural_image(prompt, width, height, style)

    return {
        "data": data,
        "ext": ".png",
        "kind": "image",
        "backend": backend,
        "prompt": prompt,
        "style": style,
        "negative": negative,
        "width": width,
        "height": height,
        "steps": steps,
        "seed": seed,
        "duration_ms": int((time.time() - started) * 1000),
    }


# ---------------------------------------------------------------------------
# public: edit
# ---------------------------------------------------------------------------
def _open_image(path: Path):
    PIL = _pil()
    if PIL is None:
        raise RuntimeError("Pillow is required for image editing.")
    from PIL import Image

    img = Image.open(path)
    img.load()
    return img


def _to_png_bytes(img) -> bytes:
    buf = io.BytesIO()
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _parse_box(box: Any, w: int, h: int) -> tuple[int, int, int, int]:
    """Accept [x0,y0,x1,y1] or 'x0,y0,x1,y1' or a fractional form."""
    if isinstance(box, str):
        parts = [p for p in re.split(r"[,\s]+", box.strip()) if p]
        box = [float(p) for p in parts]
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("box must be [x0, y0, x1, y1]")
    vals = [float(v) for v in box]
    if all(0.0 <= v <= 1.0 for v in vals):
        vals = [vals[0] * w, vals[1] * h, vals[2] * w, vals[3] * h]
    x0, y0, x1, y1 = (int(round(v)) for v in vals)
    x0, x1 = sorted((max(0, min(x0, w)), max(0, min(x1, w))))
    y0, y1 = sorted((max(0, min(y0, h)), max(0, min(y1, h))))
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError("box is too small")
    return x0, y0, x1, y1


def _feather_mask(size: tuple[int, int], box: tuple[int, int, int, int], feather: int = 12):
    from PIL import Image, ImageDraw, ImageFilter

    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rectangle(box, fill=255)
    if feather > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=feather))
    return m


def _inpaint(img, box: tuple[int, int, int, int], prompt: str, seed: int):
    """Fill a region. Uses a real diffusion inpaint pipeline when available,
    otherwise a content-aware patch built from the surrounding pixels."""
    if _diffusion_available() and settings.enable_diffusion_edit:
        try:
            import torch
            from diffusers import AutoPipelineForInpainting

            model = settings.inpaint_model or settings.image_model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            pipe = AutoPipelineForInpainting.from_pretrained(model, torch_dtype=dtype).to(device)
            mask = _feather_mask(img.size, box, feather=8)
            gen = torch.Generator(device="cpu").manual_seed(seed)
            out = pipe(
                prompt=prompt or "seamless continuation of the surrounding image",
                image=img.convert("RGB"),
                mask_image=mask,
                num_inference_steps=20,
                generator=gen,
            )
            return out.images[0]
        except Exception as exc:  # noqa: BLE001
            logger.warning("diffusion inpaint failed, using patch fill: %s", exc)

    # --- content-aware patch: mirror the surrounding ring into the hole ---
    from PIL import Image, ImageFilter

    x0, y0, x1, y1 = box
    w, h = img.size
    pad = max(8, min(x0, y0, w - x1, h - y1, 48))
    src = img.crop((max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)))
    patch = src.resize((x1 - x0, y1 - y0), Image.LANCZOS)
    patch = patch.filter(ImageFilter.GaussianBlur(radius=max(2, pad / 6)))
    out = img.copy()
    out.paste(patch, (x0, y0))
    mask = _feather_mask(img.size, box, feather=max(4, pad // 3))
    return Image.composite(out, img, mask)


def _outpaint(img, direction: str, amount: int, prompt: str, seed: int):
    """Extend the canvas and fill the new area."""
    from PIL import Image, ImageFilter

    w, h = img.size
    amount = max(16, min(int(amount or 128), 1024))
    direction = (direction or "right").lower()
    if direction in ("right", "left"):
        nw, nh = w + amount, h
    elif direction in ("up", "top", "down", "bottom"):
        nw, nh = w, h + amount
    elif direction in ("all", "both", "around"):
        nw, nh = w + amount * 2, h + amount * 2
    else:
        raise ValueError("direction must be right, left, up, down or all")

    canvas = Image.new("RGBA", (nw, nh), (0, 0, 0, 0))
    if direction in ("right", "left"):
        ox = amount if direction == "left" else 0
        oy = 0
    elif direction in ("up", "top"):
        ox, oy = 0, amount
    elif direction in ("down", "bottom"):
        ox, oy = 0, 0
    else:
        ox, oy = amount, amount
    canvas.paste(img, (ox, oy))

    # fill the empty band by stretching + blurring the nearest edge strip
    band = Image.new("RGBA", (nw, nh), (0, 0, 0, 0))
    if direction in ("right", "left"):
        strip = img.crop((0, 0, max(1, w // 8), h)) if direction == "right" else img.crop((w - max(1, w // 8), 0, w, h))
        band.paste(strip.resize((amount, h), Image.LANCZOS), (w if direction == "right" else 0, 0))
    elif direction in ("up", "top", "down", "bottom"):
        strip = img.crop((0, 0, w, max(1, h // 8))) if direction == "up" else img.crop((0, h - max(1, h // 8), w, h))
        band.paste(strip.resize((w, amount), Image.LANCZOS), (0, 0 if direction == "up" else h))
    else:
        band = canvas.copy()
        band = band.filter(ImageFilter.GaussianBlur(radius=amount / 3))

    band = band.filter(ImageFilter.GaussianBlur(radius=max(3, amount / 8)))
    out = Image.alpha_composite(canvas, band)
    return out.convert("RGBA")


def _remove_background(img, tolerance: int = 32):
    """Flood-fill from the border to knock out a near-uniform background.

    This is the free, dependency-light path. If ``rembg`` is installed it is
    used instead, which handles complex backgrounds properly.
    """
    if _has("rembg"):
        try:
            from rembg import remove

            return remove(img)
        except Exception as exc:  # noqa: BLE001
            logger.warning("rembg failed, using flood fill: %s", exc)

    from PIL import Image

    rgba = img.convert("RGBA")
    w, h = rgba.size
    px = rgba.load()
    corners = [px[0, 0], px[w - 1, 0], px[0, h - 1], px[w - 1, h - 1]]
    bg = tuple(sum(c[i] for c in corners) // 4 for i in range(3))

    tol = max(4, min(int(tolerance or 32), 160))
    seen = bytearray(w * h)
    stack = []
    for x in range(w):
        stack.append((x, 0))
        stack.append((x, h - 1))
    for y in range(h):
        stack.append((0, y))
        stack.append((w - 1, y))

    while stack:
        x, y = stack.pop()
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        i = y * w + x
        if seen[i]:
            continue
        r, g, b, a = px[x, y]
        if a == 0 or (abs(r - bg[0]) <= tol and abs(g - bg[1]) <= tol and abs(b - bg[2]) <= tol):
            seen[i] = 1
            px[x, y] = (r, g, b, 0)
            stack.extend(((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)))

    # soften the cut edge
    from PIL import ImageFilter

    alpha = rgba.getchannel("A").filter(ImageFilter.GaussianBlur(radius=1.2))
    rgba.putalpha(alpha)
    return rgba


def _upscale(img, factor: float = 2.0):
    """Upscale. Uses a real ESRGAN model when present, else Lanczos + sharpen."""
    factor = max(1.1, min(float(factor or 2.0), 4.0))
    if _has("realesrgan") and _has("basicsr"):
        try:  # pragma: no cover - optional heavy path
            from realesrgan import RealESRGANer  # type: ignore

            up = RealESRGANer(scale=int(round(factor)))
            out, _ = up.enhance(_np().array(img.convert("RGB")))
            from PIL import Image

            return Image.fromarray(out)
        except Exception as exc:  # noqa: BLE001
            logger.warning("realesrgan failed, using lanczos: %s", exc)

    from PIL import Image, ImageEnhance, ImageFilter

    w, h = img.size
    nw, nh = int(w * factor), int(h * factor)
    out = img.resize((nw, nh), Image.LANCZOS)
    out = out.filter(ImageFilter.UnsharpMask(radius=1.6, percent=110, threshold=3))
    out = ImageEnhance.Contrast(out).enhance(1.03)
    return out


def _style_transfer(img, style: str):
    """Apply a named look. Deterministic, local, no model required."""
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps

    style = (style or "").strip().lower()
    out = img.convert("RGB")

    if style in ("grayscale", "mono", "bw", "black-and-white"):
        return ImageOps.grayscale(out).convert("RGB")
    if style in ("sepia", "vintage"):
        g = ImageOps.grayscale(out)
        return ImageOps.colorize(g, black="#2b1d0e", white="#f0d9b5", mid="#a67c52")
    if style in ("invert", "negative"):
        return ImageOps.invert(out)
    if style in ("poster", "posterize", "pop"):
        return ImageOps.posterize(out, 3)
    if style in ("sketch", "pencil", "lineart"):
        g = ImageOps.grayscale(out)
        inv = ImageOps.invert(g)
        blur = inv.filter(ImageFilter.GaussianBlur(radius=6))
        if _np() is not None:
            import numpy as np

            a = np.asarray(g).astype("float32")
            b = np.asarray(blur).astype("float32")
            dodge = np.clip(a * 255.0 / np.maximum(255.0 - b, 1.0), 0, 255).astype("uint8")
            return Image.fromarray(dodge, "L").convert("RGB")
        return g.convert("RGB")
    if style in ("warm", "sunset"):
        r, g, b = out.split()
        r = r.point(lambda v: min(255, int(v * 1.12)))
        b = b.point(lambda v: int(v * 0.88))
        return Image.merge("RGB", (r, g, b))
    if style in ("cool", "arctic", "night"):
        r, g, b = out.split()
        r = r.point(lambda v: int(v * 0.88))
        b = b.point(lambda v: min(255, int(v * 1.14)))
        return Image.merge("RGB", (r, g, b))
    if style in ("vivid", "punch"):
        out = ImageEnhance.Color(out).enhance(1.6)
        return ImageEnhance.Contrast(out).enhance(1.15)
    if style in ("soft", "dream", "bloom"):
        blurred = out.filter(ImageFilter.GaussianBlur(radius=3))
        if _np() is not None:
            import numpy as np

            a = np.asarray(out).astype("float32")
            b = np.asarray(blurred).astype("float32")
            return Image.fromarray(np.clip(a * 0.65 + b * 0.45, 0, 255).astype("uint8"), "RGB")
        return blurred
    if style in ("noir", "dramatic"):
        g = ImageOps.grayscale(out)
        g = ImageEnhance.Contrast(g).enhance(1.7)
        return g.convert("RGB")
    if style in ("blur", "blurred"):
        return out.filter(ImageFilter.GaussianBlur(radius=6))
    if style in ("sharpen", "crisp"):
        return out.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=3))

    raise ValueError(
        "unknown style; try grayscale, sepia, invert, poster, sketch, warm, cool, "
        "vivid, soft, noir, blur or sharpen"
    )


def _transform(img, op: str, **kw):
    from PIL import Image, ImageOps

    op = (op or "").strip().lower()
    if op in ("rotate", "rot"):
        return img.rotate(float(kw.get("degrees", 90)), expand=True, resample=Image.BICUBIC)
    if op in ("flip", "mirror"):
        return ImageOps.mirror(img)
    if op in ("flop", "flip-vertical", "flipv"):
        return ImageOps.flip(img)
    if op in ("resize", "scale"):
        w = int(kw.get("width") or img.width)
        h = int(kw.get("height") or img.height)
        return img.resize((max(1, w), max(1, h)), Image.LANCZOS)
    if op in ("crop",):
        x0, y0, x1, y1 = _parse_box(kw.get("box"), img.width, img.height)
        return img.crop((x0, y0, x1, y1))
    if op in ("grayscale", "gray"):
        return ImageOps.grayscale(img).convert("RGB")
    if op in ("autocontrast", "contrast"):
        return ImageOps.autocontrast(img.convert("RGB"))
    if op in ("thumbnail",):
        size = int(kw.get("size") or 256)
        out = img.copy()
        out.thumbnail((size, size), Image.LANCZOS)
        return out
    raise ValueError("unknown transform; try rotate, flip, flop, resize, crop, grayscale, autocontrast, thumbnail")


def edit_image(
    source: Path,
    operation: str,
    *,
    prompt: str = "",
    box: Any = None,
    direction: str = "right",
    amount: int = 128,
    factor: float = 2.0,
    style: str = "",
    tolerance: int = 32,
    seed: Optional[int] = None,
    **kw: Any,
) -> dict:
    """Image -> image. One entry point for every edit operation."""
    op = (operation or "").strip().lower()
    img = _open_image(source)
    seed = int(seed) if seed is not None else _seed_from(f"{source.name}|{op}|{prompt}") % (2**31)
    started = time.time()
    backend = "local"

    if op in ("inpaint", "fill", "heal", "remove-object"):
        if box is None:
            raise ValueError("inpaint needs a box [x0, y0, x1, y1]")
        out = _inpaint(img, _parse_box(box, img.width, img.height), prompt, seed)
        backend = "diffusers" if (_diffusion_available() and settings.enable_diffusion_edit) else "patch-fill"
    elif op in ("outpaint", "expand", "extend"):
        out = _outpaint(img, direction, amount, prompt, seed)
        backend = "edge-extend"
    elif op in ("remove-background", "remove_bg", "bg-remove", "cutout", "matte"):
        out = _remove_background(img, tolerance=tolerance)
        backend = "rembg" if _has("rembg") else "flood-fill"
    elif op in ("upscale", "enlarge", "superres"):
        out = _upscale(img, factor)
        backend = "realesrgan" if (_has("realesrgan") and _has("basicsr")) else "lanczos+sharpen"
    elif op in ("style", "style-transfer", "filter"):
        out = _style_transfer(img, style or prompt)
        backend = "deterministic-filter"
    elif op in ("transform", "rotate", "flip", "flop", "resize", "crop", "grayscale", "autocontrast", "thumbnail"):
        out = _transform(img, op if op != "transform" else kw.get("op", "rotate"), **kw)
        backend = "pillow"
    else:
        raise ValueError(
            "unknown operation; try inpaint, outpaint, remove-background, upscale, "
            "style or transform"
        )

    data = _to_png_bytes(out)
    return {
        "data": data,
        "ext": ".png",
        "kind": "image",
        "backend": backend,
        "operation": op,
        "prompt": prompt,
        "width": out.width,
        "height": out.height,
        "seed": seed,
        "duration_ms": int((time.time() - started) * 1000),
    }


# ---------------------------------------------------------------------------
# public: understand an image
# ---------------------------------------------------------------------------
def _ocr(path: Path) -> str:
    if not shutil.which("tesseract"):
        return ""
    try:
        r = subprocess.run(
            ["tesseract", str(path), "stdout", "--psm", "3"],
            capture_output=True, timeout=settings.ocr_timeout, check=False,
        )
        return (r.stdout or b"").decode("utf-8", "replace").strip()
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("ocr failed: %s", exc)
        return ""


def _image_stats(path: Path) -> dict:
    img = _open_image(path)
    from PIL import Image, ImageFilter, ImageStat

    rgb = img.convert("RGB")
    stat = ImageStat.Stat(rgb)
    mean = [round(v, 1) for v in stat.mean]
    stddev = [round(v, 1) for v in stat.stddev]
    brightness = round(sum(mean) / 3, 1)
    contrast = round(sum(stddev) / 3, 1)

    # dominant colours via adaptive palette
    small = rgb.copy()
    small.thumbnail((160, 160))
    pal = small.convert("P", palette=Image.ADAPTIVE, colors=6).convert("RGB")
    counts: dict[tuple[int, int, int], int] = {}
    for px in pal.getdata():
        counts[px] = counts.get(px, 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
    total = sum(counts.values()) or 1
    dominant = [
        {"hex": "#%02x%02x%02x" % c, "share": round(n / total, 3)} for c, n in top
    ]

    # edge density as a crude "how busy is this image" signal
    edges = rgb.convert("L").filter(ImageFilter.FIND_EDGES)
    edge_mean = round(ImageStat.Stat(edges).mean[0], 1)

    return {
        "width": img.width,
        "height": img.height,
        "mode": img.mode,
        "format": (img.format or path.suffix.lstrip(".")).upper(),
        "aspect_ratio": round(img.width / max(1, img.height), 3),
        "orientation": "landscape" if img.width > img.height else "portrait" if img.height > img.width else "square",
        "mean_rgb": mean,
        "brightness": brightness,
        "contrast": contrast,
        "edge_density": edge_mean,
        "dominant_colors": dominant,
        "has_alpha": img.mode in ("RGBA", "LA") or "transparency" in img.info,
        "file_bytes": path.stat().st_size,
    }


def _vlm_describe(path: Path, question: str) -> str:
    """Ask a local vision-language model about the image."""
    import httpx

    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    base = (settings.vision_base_url or settings.ollama_base_url).rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    payload = {
        "model": settings.vision_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question or "Describe this image in detail."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        "temperature": 0.2,
        "max_tokens": 900,
    }
    with httpx.Client(timeout=settings.vision_timeout) as c:
        r = c.post(f"{base}/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
    return ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "").strip()


def understand_image(path: Path, question: str = "") -> dict:
    """Describe / read an image. Local VLM when available, else OCR + analysis."""
    if not path.exists():
        raise FileNotFoundError(f"no such image: {path}")

    stats = _image_stats(path)
    text = _ocr(path)
    backend = "analysis"
    description = ""

    if _vlm_available():
        try:
            description = _vlm_describe(path, question)
            backend = "vlm"
        except Exception as exc:  # noqa: BLE001
            logger.warning("vlm failed, falling back to analysis: %s", exc)

    if not description:
        bits = [
            f"A {stats['orientation']} {stats['format']} image, {stats['width']}x{stats['height']} "
            f"(aspect {stats['aspect_ratio']}).",
            f"Average colour RGB{tuple(stats['mean_rgb'])}, brightness {stats['brightness']}/255, "
            f"contrast {stats['contrast']}.",
            "Dominant colours: " + ", ".join(f"{d['hex']} ({int(d['share']*100)}%)" for d in stats["dominant_colors"]) + ".",
            f"Edge density {stats['edge_density']} — "
            + ("a busy, detailed image." if stats["edge_density"] > 18 else "a clean, low-detail image."),
        ]
        if stats["has_alpha"]:
            bits.append("It has an alpha channel (transparency).")
        if text:
            bits.append(f"Text found in the image ({len(text)} characters):\n{text[:1500]}")
        else:
            bits.append("No legible text was detected in the image.")
        description = " ".join(bits)

    return {
        "backend": backend,
        "question": question,
        "description": description,
        "ocr_text": text,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# public: read a document
# ---------------------------------------------------------------------------
def _read_pdf(path: Path) -> tuple[str, dict]:
    meta: dict = {"pages": 0, "engine": ""}
    if _has("pypdf"):
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            meta["pages"] = len(reader.pages)
            meta["engine"] = "pypdf"
            chunks = []
            for i, page in enumerate(reader.pages):
                try:
                    t = page.extract_text() or ""
                except Exception:  # noqa: BLE001
                    t = ""
                if t.strip():
                    chunks.append(f"--- page {i + 1} ---\n{t.strip()}")
            text = "\n\n".join(chunks)
            if text.strip():
                return text, meta
        except Exception as exc:  # noqa: BLE001
            logger.warning("pypdf failed: %s", exc)

    if shutil.which("pdftotext"):
        try:
            r = subprocess.run(
                ["pdftotext", "-layout", str(path), "-"],
                capture_output=True, timeout=settings.doc_timeout, check=False,
            )
            text = (r.stdout or b"").decode("utf-8", "replace").strip()
            meta["engine"] = "pdftotext"
            if text:
                return text, meta
        except (subprocess.SubprocessError, OSError) as exc:
            logger.warning("pdftotext failed: %s", exc)

    # scanned PDF: rasterise and OCR if we can
    if shutil.which("pdftoppm") and shutil.which("tesseract"):
        try:
            import tempfile

            with tempfile.TemporaryDirectory() as td:
                subprocess.run(
                    ["pdftoppm", "-r", "150", "-png", "-l", "10", str(path), f"{td}/p"],
                    capture_output=True, timeout=settings.doc_timeout, check=False,
                )
                pages = sorted(Path(td).glob("p*.png"))
                meta["pages"] = meta["pages"] or len(pages)
                meta["engine"] = "pdftoppm+tesseract"
                out = []
                for i, p in enumerate(pages):
                    t = _ocr(p)
                    if t:
                        out.append(f"--- page {i + 1} (OCR) ---\n{t}")
                if out:
                    return "\n\n".join(out), meta
        except (subprocess.SubprocessError, OSError) as exc:
            logger.warning("pdf OCR failed: %s", exc)

    return "", meta


def _read_docx(path: Path) -> tuple[str, dict]:
    if not _has("docx"):
        return "", {"engine": "unavailable"}
    from docx import Document

    doc = Document(str(path))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    tables = []
    for ti, table in enumerate(doc.tables):
        rows = []
        for row in table.rows:
            rows.append(" | ".join(c.text.strip() for c in row.cells))
        if rows:
            tables.append(f"--- table {ti + 1} ---\n" + "\n".join(rows))
    text = "\n".join(parts)
    if tables:
        text += "\n\n" + "\n\n".join(tables)
    return text, {"engine": "python-docx", "paragraphs": len(parts), "tables": len(doc.tables)}


def _read_xlsx(path: Path) -> tuple[str, dict]:
    if not _has("openpyxl"):
        return "", {"engine": "unavailable"}
    from openpyxl import load_workbook

    wb = load_workbook(str(path), read_only=True, data_only=True)
    out = []
    for ws in wb.worksheets:
        out.append(f"--- sheet: {ws.title} ---")
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i >= settings.doc_max_rows:
                out.append(f"... ({ws.max_row - settings.doc_max_rows} more rows truncated)")
                break
            cells = ["" if c is None else str(c) for c in row]
            if any(c.strip() for c in cells):
                out.append(" | ".join(cells))
    wb.close()
    return "\n".join(out), {"engine": "openpyxl", "sheets": len(wb.sheetnames)}


def _read_csv(path: Path) -> tuple[str, dict]:
    import csv

    delim = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            delim = dialect.delimiter
        except csv.Error:
            pass
        reader = csv.reader(fh, delimiter=delim)
        rows = []
        for i, row in enumerate(reader):
            if i >= settings.doc_max_rows:
                rows.append([f"... truncated at {settings.doc_max_rows} rows"])
                break
            rows.append(row)
    if not rows:
        return "", {"engine": "csv", "rows": 0, "columns": 0}
    width = max(len(r) for r in rows)
    lines = [" | ".join(c.ljust(1) for c in r) for r in rows]
    return "\n".join(lines), {"engine": "csv", "rows": len(rows), "columns": width, "delimiter": delim}


def _read_text(path: Path) -> tuple[str, dict]:
    raw = path.read_bytes()[: settings.doc_max_bytes]
    text = raw.decode("utf-8", "replace")
    return text, {"engine": "text", "bytes": len(raw)}


def read_document(path: Path) -> dict:
    """Parse any supported file into text plus structure metadata."""
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")
    ext = path.suffix.lower()
    started = time.time()

    if ext in IMAGE_EXTS:
        result = understand_image(path)
        return {
            "kind": "image",
            "text": result["ocr_text"] or result["description"],
            "meta": {"engine": result["backend"], **result["stats"]},
            "description": result["description"],
            "duration_ms": int((time.time() - started) * 1000),
        }

    if ext == ".pdf":
        text, meta = _read_pdf(path)
    elif ext == ".docx":
        text, meta = _read_docx(path)
    elif ext in (".xlsx", ".xlsm"):
        text, meta = _read_xlsx(path)
    elif ext in (".csv", ".tsv"):
        text, meta = _read_csv(path)
    elif ext in DOC_EXTS:
        text, meta = _read_text(path)
    else:
        text, meta = _read_text(path)
        meta["engine"] = "text (unknown extension)"

    text = (text or "").strip()
    truncated = len(text) > settings.doc_max_chars
    if truncated:
        text = text[: settings.doc_max_chars]

    return {
        "kind": "document",
        "text": text,
        "meta": {**meta, "extension": ext, "truncated": truncated, "chars": len(text)},
        "description": (
            f"{ext or 'file'} parsed with {meta.get('engine', 'unknown')}: "
            f"{len(text)} characters extracted"
            + (" (truncated)" if truncated else "")
            + "."
        ),
        "duration_ms": int((time.time() - started) * 1000),
    }


# ---------------------------------------------------------------------------
# helpers used by the API layer
# ---------------------------------------------------------------------------
def thumbnail(path: Path, size: int = 320) -> bytes:
    img = _open_image(path)
    img.thumbnail((size, size))
    return _to_png_bytes(img)


def data_url(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".") or "png"
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "webp": "webp", "gif": "gif", "bmp": "bmp"}.get(ext, "png")
    return f"data:image/{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def describe() -> dict:
    caps = capabilities()
    return {
        "capabilities": caps.to_dict(),
        "operations": {
            "generate": ["text-to-image"],
            "edit": ["inpaint", "outpaint", "remove-background", "upscale", "style", "transform"],
            "understand": ["describe", "ocr", "stats"],
            "documents": caps.documents,
        },
        "styles": [
            "grayscale", "sepia", "invert", "poster", "sketch", "warm", "cool",
            "vivid", "soft", "noir", "blur", "sharpen",
        ],
        "transforms": ["rotate", "flip", "flop", "resize", "crop", "grayscale", "autocontrast", "thumbnail"],
        "honest_limits": (
            "Image generation and editing run entirely on this machine. With no diffusion "
            "runtime installed the built-in procedural renderer is used, which produces real "
            "PNG art derived from the prompt but is not a diffusion model. Install "
            "`torch` + `diffusers` (or point JAGX_COMFYUI_URL at a local ComfyUI) to upgrade "
            "the same tools to a real SD/SDXL/FLUX pipeline. No API key is ever required."
        ),
    }
