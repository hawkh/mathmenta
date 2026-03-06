"""
input_processing/asr.py
Convert audio to text using OpenAI's Whisper API.
Handles math-specific phrases and provides confidence scoring.
Falls back gracefully if OPENAI_API_KEY is not set.
"""
import os
import re
import json
import tempfile
from pathlib import Path
from typing import Tuple


# ── Math phrase normalisation ────────────────────────────────────────────────
MATH_REPLACEMENTS = [
    # Verbal → symbolic
    (r"\bsquare root of\b", "sqrt("),
    (r"\bcube root of\b", "cbrt("),
    (r"\braised to the power of\b", "^"),
    (r"\braised to\b", "^"),
    (r"\bto the power\b", "^"),
    (r"\bsquared\b", "^2"),
    (r"\bcubed\b", "^3"),
    (r"\bdivided by\b", "/"),
    (r"\btimes\b", "*"),
    (r"\bminus\b", "-"),
    (r"\bplus\b", "+"),
    (r"\bequals\b", "="),
    (r"\bgreater than or equal to\b", ">="),
    (r"\bless than or equal to\b", "<="),
    (r"\bgreater than\b", ">"),
    (r"\bless than\b", "<"),
    (r"\binfinity\b", "inf"),
    (r"\bpi\b", "π"),
    (r"\bsigma\b", "Σ"),
    (r"\bdelta\b", "Δ"),
    (r"\balpha\b", "α"),
    (r"\bbeta\b", "β"),
    (r"\bgamma\b", "γ"),
    (r"\blambda\b", "λ"),
    (r"\btheta\b", "θ"),
    # Numbers written out
    (r"\bone half\b", "1/2"),
    (r"\bone third\b", "1/3"),
    (r"\btwo thirds\b", "2/3"),
    (r"\bone fourth\b", "1/4"),
    (r"\bthree fourths\b", "3/4"),
]


def normalise_math_speech(text: str) -> str:
    """Apply math-specific text normalisation to ASR output."""
    result = text
    for pattern, replacement in MATH_REPLACEMENTS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


# ── Whisper transcription ────────────────────────────────────────────────────
def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe audio using OpenAI Whisper API.

    Returns:
        {
          "transcript": str,
          "raw_transcript": str,
          "confidence": float,
          "notes": str
        }
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        return {
            "transcript": "",
            "raw_transcript": "",
            "confidence": 0.0,
            "notes": "OPENAI_API_KEY not set. Audio transcription unavailable. "
                     "Please type your question instead.",
        }

    try:
        import openai
        client = openai.OpenAI(api_key=openai_key)

        with open(audio_path, "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="en",
                response_format="verbose_json",
            )

        raw_text = response.text.strip()
        normalised = normalise_math_speech(raw_text)

        # Estimate confidence from avg log-probability (if available)
        confidence = 0.85  # Whisper is generally reliable
        if hasattr(response, "segments") and response.segments:
            avg_logprob = sum(s.get("avg_logprob", -0.2) for s in response.segments) / len(response.segments)
            # Map logprob to 0-1 range (logprob is typically -0.5 to 0)
            confidence = min(1.0, max(0.0, 1.0 + avg_logprob * 2))

        return {
            "transcript": normalised,
            "raw_transcript": raw_text,
            "confidence": round(confidence, 3),
            "notes": "Normalised math phrases from speech." if normalised != raw_text else "",
        }

    except Exception as e:
        return {
            "transcript": "",
            "raw_transcript": "",
            "confidence": 0.0,
            "notes": f"Transcription error: {str(e)}",
        }


def transcribe_bytes(audio_bytes: bytes, suffix: str = ".wav") -> dict:
    """
    Transcribe from bytes (e.g. Streamlit uploaded file).
    """
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        result = transcribe_audio(tmp_path)
    finally:
        os.unlink(tmp_path)

    return result
