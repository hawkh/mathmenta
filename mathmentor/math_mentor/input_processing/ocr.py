"""
input_processing/ocr.py
Extract mathematical text from images using Claude's vision capability.
Returns the extracted text AND a confidence estimate (0-1).
"""
import base64
import json
import re
from pathlib import Path
from typing import Tuple

import anthropic


def _encode_image(image_path: str) -> Tuple[str, str]:
    """Base64-encode an image; return (base64_data, media_type)."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    media_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_map.get(suffix, "image/jpeg")
    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, media_type


def extract_text_from_image(image_path: str) -> dict:
    """
    Use Claude vision to extract mathematical text from an image.

    Returns:
        {
          "extracted_text": str,
          "confidence": float (0-1),
          "notes": str   # any OCR caveats
        }
    """
    client = anthropic.Anthropic()
    b64_data, media_type = _encode_image(image_path)

    prompt = """You are an expert OCR system for mathematical content.

Your task:
1. Extract ALL mathematical text from this image.
2. Preserve mathematical notation as closely as possible using standard text representations:
   - Use ^ for exponents (x^2, not x²)
   - Use sqrt() for square roots
   - Use fractions as (numerator)/(denominator)
   - Preserve all variable names, operators, and symbols
3. If the image contains a math problem/question, reproduce it verbatim.
4. After extraction, rate your confidence (0.0 to 1.0) based on:
   - Image clarity: 1.0 = crystal clear, 0.5 = somewhat blurry, 0.0 = unreadable
   - Completeness: did you capture everything visible?

Respond ONLY in this JSON format (no markdown, no extra text):
{
  "extracted_text": "<full extracted text here>",
  "confidence": <float between 0.0 and 1.0>,
  "notes": "<any issues: blur, partial visibility, ambiguous symbols, etc.>"
}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_data,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    raw = response.content[0].text.strip()

    # Strip any accidental markdown fences
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        result = json.loads(raw)
        result.setdefault("confidence", 0.5)
        result.setdefault("notes", "")
        result.setdefault("extracted_text", "")
    except json.JSONDecodeError:
        # Fallback: treat whole response as extracted text
        result = {
            "extracted_text": raw,
            "confidence": 0.5,
            "notes": "JSON parse failed; raw text returned.",
        }

    return result


def extract_from_bytes(image_bytes: bytes, suffix: str = ".png") -> dict:
    """
    Convenience wrapper: extract from bytes (e.g. Streamlit uploaded file).
    Writes to a temp file, runs OCR, cleans up.
    """
    import tempfile, os

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        result = extract_text_from_image(tmp_path)
    finally:
        os.unlink(tmp_path)

    return result
