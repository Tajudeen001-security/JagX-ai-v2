"""Optional speech-to-text and text-to-speech helpers; install requirements-speech.txt."""
from __future__ import annotations
import io
import os

def transcribe(audio: bytes, filename: str = "audio.wav") -> dict:
    if not audio:
        raise ValueError("Audio is empty")
    if len(audio) > 20 * 1024 * 1024:
        raise ValueError("Audio exceeds 20 MB limit")
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Speech recognition is not installed; install requirements-speech.txt") from exc
    model_name = os.environ.get("JAGX_WHISPER_MODEL","tiny")
    model = WhisperModel(model_name, device=os.environ.get("JAGX_WHISPER_DEVICE","cpu"),
                         compute_type=os.environ.get("JAGX_WHISPER_COMPUTE","int8"))
    segments, info = model.transcribe(io.BytesIO(audio), beam_size=3, vad_filter=True)
    text = " ".join(s.text.strip() for s in segments).strip()
    return {"text":text,"language":getattr(info,"language",None),
            "language_probability":getattr(info,"language_probability",None),
            "model":model_name}

async def synthesize(text: str, voice: str | None = None) -> bytes:
    if not text or len(text) > 5000:
        raise ValueError("Text must contain 1–5000 characters")
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError("Speech synthesis is not installed; install requirements-speech.txt") from exc
    chosen = voice or os.environ.get("JAGX_TTS_VOICE","en-NG-AbeoNeural")
    communicate = edge_tts.Communicate(text, chosen)
    output = bytearray()
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            output.extend(chunk["data"])
    if not output:
        raise RuntimeError("Speech service returned no audio")
    return bytes(output)
