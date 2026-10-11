"""Optional speech services: local Whisper transcription and keyless online TTS."""
from __future__ import annotations
import io
import os
from functools import lru_cache
from typing import Optional

@lru_cache(maxsize=1)
def _whisper_model():
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Speech recognition is not installed. Install requirements-speech.txt and ensure the host has enough memory and disk space.") from exc
    model_size = os.environ.get("JAGX_WHISPER_MODEL", "small")
    device = os.environ.get("JAGX_WHISPER_DEVICE", "cpu")
    compute_type = os.environ.get("JAGX_WHISPER_COMPUTE_TYPE", "int8")
    return WhisperModel(model_size, device=device, compute_type=compute_type)

def transcribe_audio(audio_bytes: bytes) -> dict:
    if not audio_bytes:
        raise ValueError("Audio file is empty.")
    if len(audio_bytes) > 20 * 1024 * 1024:
        raise ValueError("Audio file exceeds the 20 MB limit.")
    model = _whisper_model()
    segments, info = model.transcribe(io.BytesIO(audio_bytes), beam_size=3, vad_filter=True)
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return {"text": text, "language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None), "engine": "faster-whisper"}

async def synthesize_speech(text: str, voice: Optional[str] = None) -> bytes:
    if not (text or "").strip():
        raise ValueError("Text is required.")
    if len(text) > 5000:
        raise ValueError("Text exceeds the 5000-character limit.")
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError("Speech synthesis is not installed. Install requirements-speech.txt.") from exc
    selected_voice = voice or os.environ.get("JAGX_TTS_VOICE", "en-NG-AbeoNeural")
    output = io.BytesIO()
    try:
        communicate = edge_tts.Communicate(text.strip(), selected_voice)
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                output.write(chunk["data"])
    except Exception as exc:
        raise RuntimeError("The selected speech voice could not be reached. Check internet access and the voice name; speech synthesis uses a free external service and is not guaranteed to be available offline.") from exc
    if not output.getvalue():
        raise RuntimeError("The speech service returned no audio.")
    return output.getvalue()
