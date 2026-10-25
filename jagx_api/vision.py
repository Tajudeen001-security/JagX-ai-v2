"""
JagX AI — Vision input (describe / understand images).
Created by JagX & JRILICENSE
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from jagx_api.providers import chat_completion
from jagx_api.chat import sanitize_identity

logger = logging.getLogger("jagx-ai")


def _data_url(image_base64: str, mime: str = "image/jpeg") -> str:
    raw = (image_base64 or "").strip()
    if raw.startswith("data:"):
        return raw
    return f"data:{mime};base64,{raw}"


def describe_image(
    image_base64: str,
    prompt: str = "Describe this image in detail. Note text, UI elements, and anything useful for debugging or answering the user.",
    *,
    mime_type: str = "image/jpeg",
    max_tokens: int = 800,
) -> dict[str, Any]:
    if not image_base64 or len(image_base64) < 32:
        raise ValueError("image_base64 is required")
    if len(image_base64) > 5_500_000:
        raise ValueError("Image too large (max ~4MB encoded)")

    data_url = _data_url(image_base64, mime_type)
    user_content = [
        {"type": "text", "text": (prompt or "Describe this image.")[:2000]},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]
    messages = [
        {
            "role": "system",
            "content": (
                "You are JagX AI (JagX & JRILICENSE). Describe images accurately and helpfully. "
                "If the image is a UI screenshot or code, extract text and suggest concrete next steps."
            ),
        },
        {"role": "user", "content": user_content},
    ]

    text, provider = chat_completion(messages, max_tokens=max_tokens)
    if not text:
        return {
            "ok": False,
            "description": (
                "Vision is configured, but no multimodal provider accepted this image. "
                "Ensure Groq/OpenRouter/NVIDIA model supports vision, or attach a smaller JPEG/PNG."
            ),
            "provider": provider or "none",
        }

    return {
        "ok": True,
        "description": sanitize_identity(text.strip()),
        "provider": provider,
    }


def vision_as_context(image_base64: str, user_prompt: str, mime_type: str = "image/jpeg") -> Optional[str]:
    try:
        result = describe_image(
            image_base64,
            prompt=f"The user asked: {user_prompt[:500]}\n\nAnalyze the image in service of that question.",
            mime_type=mime_type,
        )
        if result.get("ok"):
            return "Image analysis:\n" + result["description"]
        return result.get("description")
    except Exception as exc:
        logger.warning("vision_as_context failed: %s", exc)
        return None
