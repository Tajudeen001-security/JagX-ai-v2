"""JagX v13 — multimodal engine tests.

Everything here runs offline, with no GPU, no diffusion model and no API key.
That is the point: the free fallback tier is the baseline the product must
always satisfy, so these tests pin it down.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from app import db, media, media_tools, tools


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _png_bytes(width: int = 64, height: int = 64) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), (200, 40, 90)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def asset(user):
    """A generated image stored for the test user."""
    result = media.generate_image("a test pattern", width=96, height=96)
    aid, path = media.save_bytes(user["id"], result["data"], ".png")
    db.add_media(
        user["id"], aid, kind="image", source="generated",
        filename=f"{aid}.png", ext=".png", mime="image/png",
        size_bytes=len(result["data"]), width=96, height=96,
        prompt="a test pattern", backend=result["backend"],
    )
    return aid, path


# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------
def test_capabilities_reports_local_and_keyless():
    caps = media.capabilities().to_dict()
    assert caps["local_only"] is True
    assert caps["requires_api_key"] is False
    assert caps["image_generation"] in ("fallback", "diffusers", "comfyui")
    assert caps["vision"] in ("fallback", "vlm")
    assert isinstance(caps["documents"], list) and caps["documents"]


def test_describe_lists_operations_and_honest_limits():
    d = media.describe()
    assert "generate" in d["operations"]
    assert "inpaint" in d["operations"]["edit"]
    assert "remove-background" in d["operations"]["edit"]
    assert "upscale" in d["operations"]["edit"]
    assert "style" in d["operations"]["edit"]
    assert "transform" in d["operations"]["edit"]
    assert "honest_limits" in d and d["honest_limits"]


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
def test_generate_image_produces_a_real_png():
    result = media.generate_image("a lighthouse at dusk", width=128, height=96)
    data = result["data"]
    assert data[:8] == b"\x89PNG\r\n\x1a\n"  # real PNG magic
    assert len(data) > 1000
    assert result["width"] == 128 and result["height"] == 96
    assert result["backend"] in ("procedural", "diffusers", "comfyui")
    assert result["seed"] is not None


def test_generate_is_deterministic_for_the_same_prompt():
    a = media.generate_image("same prompt", width=64, height=64)
    b = media.generate_image("same prompt", width=64, height=64)
    assert a["data"] == b["data"]


def test_generate_differs_for_different_prompts():
    a = media.generate_image("a red desert", width=64, height=64)
    b = media.generate_image("a blue ocean", width=64, height=64)
    assert a["data"] != b["data"]


def test_generate_rejects_empty_prompt():
    with pytest.raises(ValueError):
        media.generate_image("   ")


def test_generate_clamps_oversized_dimensions():
    result = media.generate_image("clamp me", width=99999, height=99999)
    from app.config import settings

    assert result["width"] <= settings.image_max_side
    assert result["height"] <= settings.image_max_side


# ---------------------------------------------------------------------------
# editing
# ---------------------------------------------------------------------------
def test_inpaint_changes_the_image(asset):
    _, path = asset
    before = path.read_bytes()
    result = media.edit_image(path, "inpaint", box=[10, 10, 60, 60], prompt="a boat")
    assert result["data"] != before
    assert result["operation"] == "inpaint"
    assert result["backend"] in ("patch-fill", "diffusers")


def test_inpaint_requires_a_box(asset):
    _, path = asset
    with pytest.raises(ValueError):
        media.edit_image(path, "inpaint")


def test_outpaint_grows_the_canvas(asset):
    _, path = asset
    result = media.edit_image(path, "outpaint", direction="right", amount=32)
    assert result["width"] == 96 + 32
    assert result["height"] == 96


def test_outpaint_all_grows_both_axes(asset):
    _, path = asset
    result = media.edit_image(path, "outpaint", direction="all", amount=16)
    assert result["width"] == 96 + 32
    assert result["height"] == 96 + 32


def test_remove_background_adds_alpha(asset):
    _, path = asset
    result = media.edit_image(path, "remove-background")
    from PIL import Image

    img = Image.open(io.BytesIO(result["data"]))
    assert img.mode in ("RGBA", "LA")
    assert result["backend"] in ("flood-fill", "rembg")


def test_upscale_multiplies_dimensions(asset):
    _, path = asset
    result = media.edit_image(path, "upscale", factor=2.0)
    assert result["width"] == 192 and result["height"] == 192


def test_style_transfer_grayscale(asset):
    _, path = asset
    result = media.edit_image(path, "style", style="grayscale")
    from PIL import Image

    img = Image.open(io.BytesIO(result["data"])).convert("RGB")
    r, g, b = img.getpixel((5, 5))
    assert abs(r - g) <= 2 and abs(g - b) <= 2


def test_style_transfer_rejects_unknown_style(asset):
    _, path = asset
    with pytest.raises(ValueError):
        media.edit_image(path, "style", style="not-a-real-style")


def test_transform_rotate(asset):
    _, path = asset
    result = media.edit_image(path, "transform", degrees=90)
    assert result["width"] > 0 and result["height"] > 0


def test_unknown_operation_is_rejected(asset):
    _, path = asset
    with pytest.raises(ValueError):
        media.edit_image(path, "explode")


# ---------------------------------------------------------------------------
# understanding
# ---------------------------------------------------------------------------
def test_understand_image_returns_description_and_stats(asset):
    _, path = asset
    result = media.understand_image(path, "what is this?")
    assert result["description"]
    assert result["backend"] in ("analysis", "vlm")
    stats = result["stats"]
    assert stats["width"] == 96 and stats["height"] == 96
    assert stats["orientation"] == "square"
    assert len(stats["dominant_colors"]) > 0
    assert "brightness" in stats and "contrast" in stats


def test_understand_image_ocr_reads_text(tmp_path):
    from PIL import Image, ImageDraw, ImageFont

    # Render at a size tesseract can actually read: a real TrueType font at
    # ~48px, not PIL's tiny built-in bitmap font.
    font = None
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if Path(candidate).exists():
            font = ImageFont.truetype(candidate, 48)
            break

    img = Image.new("RGB", (760, 140), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((30, 40), "JAGX OCR 12345", fill=(0, 0, 0), font=font)
    p = tmp_path / "text.png"
    img.save(p)

    result = media.understand_image(p)
    if media.capabilities().ocr:
        text = result["ocr_text"].upper().replace(" ", "")
        assert "JAGX" in text or "12345" in text, f"OCR returned: {result['ocr_text']!r}"
    else:
        assert result["ocr_text"] == ""


def test_understand_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        media.understand_image(tmp_path / "nope.png")


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------
def test_read_txt(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("Hello JagX.\nSecond line.")
    result = media.read_document(p)
    assert "Hello JagX." in result["text"]
    assert result["meta"]["engine"] == "text"


def test_read_csv(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("name,score\nalpha,10\nbeta,20\n")
    result = media.read_document(p)
    assert "alpha" in result["text"] and "beta" in result["text"]
    assert result["meta"]["engine"] == "csv"
    assert result["meta"]["rows"] == 3


def test_read_json(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"a": 1, "b": [2, 3]}))
    result = media.read_document(p)
    assert '"a"' in result["text"]


def test_read_pdf(tmp_path):
    pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    p = tmp_path / "blank.pdf"
    with p.open("wb") as fh:
        writer.write(fh)

    result = media.read_document(p)
    assert result["kind"] == "document"
    assert result["meta"]["pages"] == 1


def test_read_docx(tmp_path):
    pytest.importorskip("docx")
    from docx import Document

    doc = Document()
    doc.add_paragraph("JagX docx paragraph.")
    p = tmp_path / "d.docx"
    doc.save(p)

    result = media.read_document(p)
    assert "JagX docx paragraph." in result["text"]
    assert result["meta"]["engine"] == "python-docx"


def test_read_xlsx(tmp_path):
    pytest.importorskip("openpyxl")
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.append(["name", "score"])
    ws.append(["alpha", 10])
    p = tmp_path / "d.xlsx"
    wb.save(p)

    result = media.read_document(p)
    assert "alpha" in result["text"]
    assert result["meta"]["engine"] == "openpyxl"


def test_read_document_on_an_image_uses_vision(asset):
    _, path = asset
    result = media.read_document(path)
    assert result["kind"] == "image"
    assert result["text"]


def test_read_missing_document_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        media.read_document(tmp_path / "gone.pdf")


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------
def test_save_and_resolve_asset(user):
    aid, path = media.save_bytes(user["id"], _png_bytes(), ".png")
    assert path.exists()
    assert media.resolve_asset(user["id"], aid) == path
    assert media.resolve_asset(user["id"], "med_does_not_exist") is None


def test_thumbnail_is_smaller(asset):
    _, path = asset
    thumb = media.thumbnail(path, size=32)
    from PIL import Image

    img = Image.open(io.BytesIO(thumb))
    assert max(img.size) <= 32


def test_data_url_is_embeddable(asset):
    _, path = asset
    url = media.data_url(path)
    assert url.startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# db layer
# ---------------------------------------------------------------------------
def test_media_crud(user):
    row = db.add_media(
        user["id"], "med_test_crud", kind="image", source="generated",
        filename="x.png", ext=".png", mime="image/png", size_bytes=123,
        width=10, height=20, prompt="p", backend="procedural",
    )
    assert row["id"] == "med_test_crud"
    assert row["width"] == 10

    got = db.get_media(user["id"], "med_test_crud")
    assert got and got["prompt"] == "p"
    assert isinstance(got["meta"], dict)

    assert any(m["id"] == "med_test_crud" for m in db.list_media(user["id"]))
    assert any(m["id"] == "med_test_crud" for m in db.list_media(user["id"], kind="image"))
    assert not any(m["id"] == "med_test_crud" for m in db.list_media(user["id"], kind="document"))

    stats = db.media_stats(user["id"])
    assert stats["total_count"] >= 1

    db.delete_media(user["id"], "med_test_crud")
    assert db.get_media(user["id"], "med_test_crud") is None


def test_media_is_scoped_per_user(user):
    other = db.create_user("other-media@example.com", "x")
    db.add_media(user["id"], "med_scoped", kind="image", filename="a.png")
    assert db.get_media(other["id"], "med_scoped") is None


# ---------------------------------------------------------------------------
# tool registry
# ---------------------------------------------------------------------------
def test_multimodal_tools_are_registered():
    for name in ("generate_image", "edit_image", "understand_image", "read_document", "list_media"):
        assert name in tools.TOOLS, f"{name} missing from the registry"
        assert tools.TOOLS[name].category == "media"


def test_tool_schemas_are_openai_shaped():
    by_name = {s["function"]["name"]: s["function"] for s in tools.schemas()}
    for name in ("generate_image", "edit_image", "understand_image", "read_document", "list_media"):
        spec = by_name[name]
        assert spec["description"]
        assert spec["parameters"]["type"] == "object"
        assert "properties" in spec["parameters"]


def test_tool_generate_then_edit_then_understand(user):
    out, ok = tools.execute("generate_image", {"prompt": "a fox", "width": 96, "height": 96}, user_id=user["id"])
    assert ok, out
    payload = json.loads(out)
    assert payload["ok"] is True
    aid = payload["asset_id"]

    out2, ok2 = tools.execute("edit_image", {"asset_id": aid, "operation": "style", "style": "sepia"}, user_id=user["id"])
    assert ok2, out2
    assert json.loads(out2)["parent_id"] == aid

    out3, ok3 = tools.execute("understand_image", {"asset_id": aid}, user_id=user["id"])
    assert ok3, out3
    assert json.loads(out3)["description"]

    out4, ok4 = tools.execute("list_media", {}, user_id=user["id"])
    assert ok4, out4
    assert json.loads(out4)["count"] >= 2


def test_tool_edit_unknown_asset_is_a_clean_error(user):
    out, ok = tools.execute("edit_image", {"asset_id": "med_nope", "operation": "upscale"}, user_id=user["id"])
    assert ok is True  # the tool ran; it reported a domain error
    assert "no image with asset_id" in out


def test_tool_read_document_roundtrip(user, tmp_path):
    p = tmp_path / "r.txt"
    p.write_text("roundtrip content")
    aid, _ = media.save_bytes(user["id"], p.read_bytes(), ".txt")
    db.add_media(user["id"], aid, kind="document", source="upload", filename="r.txt", ext=".txt")

    out, ok = tools.execute("read_document", {"asset_id": aid}, user_id=user["id"])
    assert ok, out
    assert "roundtrip content" in json.loads(out)["text"]


def test_tool_list_media_empty_gallery(user):
    out, ok = tools.execute("list_media", {}, user_id=user["id"])
    assert ok
    assert json.loads(out)["count"] == 0


# ---------------------------------------------------------------------------
# guard
# ---------------------------------------------------------------------------
def test_guard_allows_media_tools():
    from app import guard

    for name in ("generate_image", "edit_image", "understand_image", "read_document", "list_media"):
        verdict = guard.check(name, "local operation", tool=name)
        assert verdict.verdict == guard.ALLOW, f"{name} should be allowed, got {verdict.verdict}"


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.api import create_app

    return TestClient(create_app())


@pytest.fixture
def auth(client):
    import uuid

    email = f"media{uuid.uuid4().hex[:8]}@example.com"
    r = client.post("/api/auth/signup", json={"email": email, "password": "password123"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def test_api_media_capabilities(client, auth):
    r = client.get("/api/media/capabilities", headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["capabilities"]["requires_api_key"] is False


def test_api_generate_list_raw_thumb(client, auth):
    r = client.post("/api/media/generate", json={"prompt": "api test image", "width": 96, "height": 96}, headers=auth)
    assert r.status_code == 200, r.text
    asset = r.json()
    aid = asset["id"]
    assert asset["width"] == 96

    r2 = client.get("/api/media", headers=auth)
    assert r2.status_code == 200
    assert any(m["id"] == aid for m in r2.json()["media"])

    r3 = client.get(f"/api/media/{aid}/raw", headers=auth)
    assert r3.status_code == 200
    assert r3.content[:8] == b"\x89PNG\r\n\x1a\n"

    r4 = client.get(f"/api/media/{aid}/thumb", headers=auth)
    assert r4.status_code == 200
    assert r4.headers["content-type"] == "image/png"


def test_api_upload_then_read_document(client, auth):
    files = {"file": ("notes.txt", b"uploaded document body", "text/plain")}
    r = client.post("/api/media/upload", files=files, headers=auth)
    assert r.status_code == 200, r.text
    aid = r.json()["id"]
    assert r.json()["kind"] == "document"

    r2 = client.post(f"/api/media/{aid}/read", headers=auth)
    assert r2.status_code == 200
    assert "uploaded document body" in r2.json()["text"]


def test_api_upload_image_is_detected(client, auth):
    files = {"file": ("pic.png", _png_bytes(40, 30), "image/png")}
    r = client.post("/api/media/upload", files=files, headers=auth)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "image"
    assert body["width"] == 40 and body["height"] == 30


def test_api_edit_and_understand(client, auth):
    r = client.post("/api/media/generate", json={"prompt": "edit me", "width": 96, "height": 96}, headers=auth)
    aid = r.json()["id"]

    r2 = client.post(f"/api/media/{aid}/edit", json={"operation": "upscale", "factor": 2.0}, headers=auth)
    assert r2.status_code == 200, r2.text
    assert r2.json()["width"] == 192

    r3 = client.post(f"/api/media/{aid}/understand", json={"question": "what is this?"}, headers=auth)
    assert r3.status_code == 200
    assert r3.json()["description"]


def test_api_edit_bad_operation_is_400(client, auth):
    r = client.post("/api/media/generate", json={"prompt": "x", "width": 64, "height": 64}, headers=auth)
    aid = r.json()["id"]
    r2 = client.post(f"/api/media/{aid}/edit", json={"operation": "nonsense"}, headers=auth)
    assert r2.status_code == 400


def test_api_generate_requires_prompt(client, auth):
    r = client.post("/api/media/generate", json={"prompt": "  "}, headers=auth)
    assert r.status_code == 400


def test_api_media_requires_auth(client):
    assert client.get("/api/media").status_code == 401
    assert client.post("/api/media/generate", json={"prompt": "x"}).status_code == 401


def test_api_media_is_scoped_to_the_owner(client, auth):
    r = client.post("/api/media/generate", json={"prompt": "private", "width": 64, "height": 64}, headers=auth)
    aid = r.json()["id"]

    import uuid

    email = f"other{uuid.uuid4().hex[:8]}@example.com"
    r2 = client.post("/api/auth/signup", json={"email": email, "password": "password123"})
    other = {"Authorization": f"Bearer {r2.json()['token']}"}

    assert client.get(f"/api/media/{aid}", headers=other).status_code == 404
    assert client.get(f"/api/media/{aid}/raw", headers=other).status_code == 404


def test_api_delete_media(client, auth):
    r = client.post("/api/media/generate", json={"prompt": "delete me", "width": 64, "height": 64}, headers=auth)
    aid = r.json()["id"]
    assert client.delete(f"/api/media/{aid}", headers=auth).status_code == 200
    assert client.get(f"/api/media/{aid}", headers=auth).status_code == 404


def test_api_media_stats(client, auth):
    client.post("/api/media/generate", json={"prompt": "stats", "width": 64, "height": 64}, headers=auth)
    r = client.get("/api/media", headers=auth)
    assert r.json()["stats"]["total_count"] >= 1
