"""
memory/memory_store.py
JSON-based memory layer.
- Persists solved problems, user feedback, and OCR/ASR corrections.
- Retrieves similar past problems at runtime using simple keyword + embedding matching.
"""
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

STORE_PATH = Path(os.getenv("MEMORY_STORE_PATH", Path(__file__).parent / "memory_store.json"))


# ── Persistence helpers ──────────────────────────────────────────────────────
def _load_store() -> List[Dict]:
    if STORE_PATH.exists():
        try:
            return json.loads(STORE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _save_store(records: List[Dict]) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Save a solved problem ────────────────────────────────────────────────────
def save_problem(
    *,
    input_mode: str,
    raw_input: str,
    parsed_problem: dict,
    retrieved_chunks: List[dict],
    solution: str,
    explanation: str,
    verifier_confidence: float,
    is_correct: bool,
    user_feedback: Optional[str] = None,
    human_correction: Optional[str] = None,
) -> str:
    """Save a solved problem to the memory store. Returns the record ID."""
    records = _load_store()
    record_id = str(uuid.uuid4())[:8]
    record = {
        "id": record_id,
        "timestamp": datetime.utcnow().isoformat(),
        "input_mode": input_mode,
        "raw_input": raw_input,
        "parsed_problem": parsed_problem,
        "retrieved_sources": [c.get("source", "") for c in retrieved_chunks],
        "solution": solution,
        "explanation": explanation,
        "verifier_confidence": verifier_confidence,
        "is_correct": is_correct,
        "user_feedback": user_feedback,
        "human_correction": human_correction,
    }
    records.append(record)
    _save_store(records)
    return record_id


# ── Retrieve similar problems ────────────────────────────────────────────────
def _keyword_overlap(text_a: str, text_b: str) -> float:
    """Simple Jaccard-like keyword overlap score."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def retrieve_similar(
    query: str,
    topic: Optional[str] = None,
    top_k: int = 3,
    only_correct: bool = True,
) -> List[Dict]:
    """
    Retrieve top-k most similar past problems.
    Optionally filter by topic and only return verified-correct problems.
    """
    records = _load_store()
    if not records:
        return []

    candidates = records
    if only_correct:
        candidates = [r for r in records if r.get("is_correct", False)]
    if topic:
        candidates = [
            r for r in candidates
            if r.get("parsed_problem", {}).get("topic", "").lower() == topic.lower()
        ] or candidates  # fall back to all if filtered list is empty

    scored = []
    for r in candidates:
        problem_text = r.get("parsed_problem", {}).get("problem_text", "")
        raw = r.get("raw_input", "")
        score = _keyword_overlap(query, problem_text + " " + raw)
        scored.append((score, r))

    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:top_k] if _ > 0.05]  # min 5% overlap


# ── OCR / ASR correction rules ───────────────────────────────────────────────
def get_correction_rules() -> List[Dict]:
    """
    Return saved OCR/ASR correction rules derived from past HITL edits.
    These can be applied to future similar inputs.
    """
    records = _load_store()
    rules = []
    for r in records:
        if r.get("human_correction") and r.get("raw_input"):
            rules.append({
                "original": r["raw_input"],
                "corrected": r["human_correction"],
                "input_mode": r.get("input_mode", "text"),
                "timestamp": r.get("timestamp", ""),
            })
    return rules


# ── Get all records (for UI display) ─────────────────────────────────────────
def get_all_records(limit: int = 50) -> List[Dict]:
    records = _load_store()
    return records[-limit:]


# ── Update feedback on an existing record ───────────────────────────────────
def update_feedback(record_id: str, feedback: str, is_correct: bool) -> bool:
    records = _load_store()
    for r in records:
        if r["id"] == record_id:
            r["user_feedback"] = feedback
            r["is_correct"] = is_correct
            _save_store(records)
            return True
    return False
