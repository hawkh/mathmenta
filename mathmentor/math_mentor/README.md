# 🧮 Math Mentor — Multimodal JEE Math AI Tutor

> A Reliable Multimodal Math Mentor built with RAG + Multi-Agent LangGraph + HITL + Memory

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-red)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-green)](https://langchain-ai.github.io/langgraph/)
[![Claude](https://img.shields.io/badge/Claude-Sonnet%204-orange)](https://anthropic.com)

---

## 📐 Architecture Overview

```
Multimodal Input (image/audio/text)
        │
        ▼
Extraction Layer (Claude Vision OCR / Whisper ASR)
        │
        ▼ [HITL if low confidence]
LangGraph Multi-Agent Pipeline:
  Parser → Router → Solver (RAG) → Verifier → Explainer → Memory
        │                │                │
        ▼ [HITL if       ▼ FAISS+         ▼ [HITL if
        ambiguous]   embeddings        confidence low]
        ▼
Streamlit UI (trace, RAG sources, explanation, feedback)
        │
        ▼
Memory Store (JSON) — reused for similar future problems
```

See `architecture.mermaid` for the full diagram.

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/yourname/math-mentor.git
cd math-mentor
```

### 2. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your API key(s):
```

```env
ANTHROPIC_API_KEY=sk-ant-...     # Required
OPENAI_API_KEY=sk-...            # Optional (for audio transcription)
```

### 5. Build the knowledge base index (auto-builds on first run, or manually):
```bash
python -c "from rag.vector_store import build_index; build_index(force_rebuild=True)"
```

### 6. Run the app
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## 🏗️ Project Structure

```
math_mentor/
├── app.py                        # 🖥️  Streamlit UI (main entrypoint)
├── requirements.txt
├── .env.example
├── architecture.mermaid
│
├── agents/
│   ├── graph.py                  # 🕸️  LangGraph state machine
│   └── nodes.py                  # 🤖  All 5+ agent node implementations
│
├── rag/
│   ├── vector_store.py           # 📚  FAISS build/load/retrieve
│   └── knowledge_base/
│       ├── algebra.txt
│       ├── probability.txt
│       ├── calculus.txt
│       ├── linear_algebra.txt
│       └── jee_strategies.txt
│
├── input_processing/
│   ├── ocr.py                    # 📷  Claude Vision OCR
│   └── asr.py                    # 🎙️  OpenAI Whisper ASR
│
├── memory/
│   └── memory_store.py           # 🧠  JSON-based persistent memory
│
└── utils/
    └── tools.py                  # 🖩  Safe Python calculator
```

---

## 🤖 Multi-Agent System

| Agent | Role | Key Output |
|-------|------|-----------|
| **Parser** | Cleans OCR/ASR noise → structured JSON | `{topic, variables, constraints, what_to_find}` |
| **Router** | Classifies problem, plans strategy, queries memory | `{strategy, key_concepts, rag_query, difficulty}` |
| **Solver** | RAG retrieval + Python calculator + solution | Solution steps + final answer |
| **Verifier** | Independent correctness check + edge cases | `{is_correct, confidence, issues_found}` |
| **Explainer** | Student-friendly step-by-step explanation | Markdown explanation with key insights |
| **Memory Node** | Persists result for future retrieval | Record ID saved to JSON |

---

## 📚 RAG Pipeline

- **Knowledge base:** 5 `.txt` files covering algebra, probability, calculus, linear algebra, and JEE strategies
- **Chunking:** 400-character chunks with 80-character overlap
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **Vector store:** FAISS `IndexFlatIP` (inner product ≈ cosine similarity after L2 normalisation)
- **Retrieval:** Top-5 chunks, displayed with source citations in UI

---

## 👤 Human-in-the-Loop (HITL) Triggers

| Trigger | Condition | Action |
|---------|-----------|--------|
| OCR review | Confidence < 0.75 | Show extraction with warning, allow editing |
| ASR review | Confidence < 0.75 | Show transcript with warning, allow editing |
| Parser HITL | `needs_clarification = true` or topic = unknown | Ask user to clarify problem |
| Verifier HITL | Confidence < 0.70 or `is_correct = false` | Show solution for human review/correction |
| User-triggered | User clicks "Incorrect" feedback button | Correction saved to memory |

---

## 🧠 Memory & Self-Learning

Memory is stored in `memory/memory_store.json` with:
```json
{
  "id": "abc12345",
  "timestamp": "2025-01-01T12:00:00",
  "input_mode": "text",
  "parsed_problem": {"topic": "probability", ...},
  "solution": "...",
  "final_answer": "7/12",
  "verifier_confidence": 0.92,
  "is_correct": true,
  "user_feedback": "correct"
}
```

At runtime:
- **Router Agent** retrieves similar past problems (keyword overlap scoring)
- **Solver Agent** uses past solution patterns as context
- HITL corrections saved as learning signals for future OCR/ASR problems

---

## 🌐 Deployment

### Streamlit Cloud (recommended)
1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → select `app.py`
4. Add secrets in Streamlit Cloud dashboard:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   OPENAI_API_KEY = "sk-..."     # optional
   ```
5. Deploy → get public URL

### HuggingFace Spaces
1. Create Space with Streamlit SDK
2. Push code
3. Add secrets in Space settings

### Render / Railway
```bash
# Procfile
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

---

## 🧪 Demo Scenarios

### 1. Text Input
> "A bag contains 5 red and 3 blue balls. Two balls are drawn without replacement. What is the probability both are red?"

### 2. Image Input
Upload a screenshot of a JEE calculus problem — Claude Vision extracts the text.

### 3. Audio Input
Say "Find the derivative of x squared plus 3x minus 5" — Whisper transcribes + normalises.

### 4. HITL Demo
Upload a blurry image → low OCR confidence → system prompts review → user corrects → solve.

### 5. Memory Reuse
Solve the bag-of-balls problem, then ask a similar probability problem — Router finds the past solution as a reference pattern.

---

## 📊 Evaluation Summary

| Metric | Result |
|--------|--------|
| Input modes | ✅ Text, Image (Claude Vision), Audio (Whisper) |
| Agent count | ✅ 6 (Parser, Router, Solver, Verifier, Explainer, Memory) |
| RAG pipeline | ✅ 5-doc chunked FAISS index, sources cited in UI |
| HITL triggers | ✅ 4 distinct triggers (OCR, ASR, Parser, Verifier) |
| Memory | ✅ Persistent JSON, keyword-similarity retrieval |
| Deployment | ✅ Streamlit Cloud ready |
| JEE topics | ✅ Algebra, Probability, Calculus, Linear Algebra |

---

## 🛠️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | **Required.** Claude API key |
| `OPENAI_API_KEY` | — | Optional. For Whisper audio transcription |
| `OCR_CONFIDENCE_THRESHOLD` | `0.75` | HITL trigger threshold for OCR/ASR |
| `SOLVER_CONFIDENCE_THRESHOLD` | `0.70` | HITL trigger threshold for verifier |
| `RAG_TOP_K` | `5` | Number of RAG chunks to retrieve |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `FAISS_INDEX_PATH` | `./rag/faiss_index` | FAISS index file path |
| `MEMORY_STORE_PATH` | `./memory/memory_store.json` | Memory JSON path |

---

## 📝 License

MIT — free for educational use.
