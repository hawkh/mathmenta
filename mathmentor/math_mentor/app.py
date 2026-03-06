"""
app.py  –  Math Mentor  |  Streamlit UI
A Reliable Multimodal Math Mentor powered by RAG + Multi-Agent LangGraph pipeline.

Run with:
    streamlit run app.py
"""
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# ── Load environment ─────────────────────────────────────────────────────────
load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Math Mentor – JEE AI Tutor",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] { gap: 12px; }
.agent-card {
    background: #1e1e2e; border-radius: 8px; padding: 12px 16px;
    margin-bottom: 8px; border-left: 3px solid #7c3aed;
}
.rag-card {
    background: #0f172a; border-radius: 6px; padding: 10px 14px;
    margin-bottom: 6px; border-left: 3px solid #0ea5e9;
    font-size: 0.85em;
}
.hitl-banner {
    background: #7c2d12; border-radius: 8px; padding: 14px;
    border: 1px solid #ea580c; margin: 12px 0;
}
.memory-card {
    background: #1a1a2e; border-radius: 6px; padding: 10px;
    margin: 6px 0; border-left: 3px solid #16a34a; font-size: 0.85em;
}
.confidence-high { color: #22c55e; font-weight: bold; }
.confidence-med  { color: #f59e0b; font-weight: bold; }
.confidence-low  { color: #ef4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def confidence_badge(score: float) -> str:
    pct = f"{score:.0%}"
    if score >= 0.80:
        return f'<span class="confidence-high">● {pct}</span>'
    elif score >= 0.60:
        return f'<span class="confidence-med">● {pct}</span>'
    else:
        return f'<span class="confidence-low">● {pct}</span>'


def init_session():
    defaults = {
        "step": "input",           # input | ocr_review | asr_review | solving | hitl_parse | hitl_verify | done
        "phase1_state": None,
        "phase2_state": None,
        "ocr_result": None,
        "asr_result": None,
        "extracted_text": "",
        "input_mode": "text",
        "raw_input": "",
        "memory_id": None,
        "feedback_given": False,
        "build_index_done": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_session():
    keys = list(st.session_state.keys())
    for k in keys:
        del st.session_state[k]
    init_session()


init_session()

# ── Check API key ─────────────────────────────────────────────────────────────
if not os.getenv("ANTHROPIC_API_KEY"):
    st.error("⚠️ **ANTHROPIC_API_KEY** not set. Please add it to your `.env` file.")
    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧮 Math Mentor")
    st.caption("JEE-level AI Math Tutor")
    st.divider()

    # Build RAG index on first run
    if not st.session_state.build_index_done:
        with st.spinner("Building knowledge base index…"):
            try:
                from rag.vector_store import build_index
                build_index()
                st.session_state.build_index_done = True
            except Exception as e:
                st.warning(f"Index build issue: {e}")
                st.session_state.build_index_done = True

    st.success("✅ Knowledge base ready")
    st.divider()

    st.subheader("📚 Topics Covered")
    topics = ["Algebra", "Probability", "Basic Calculus", "Linear Algebra"]
    for t in topics:
        st.markdown(f"• {t}")
    st.divider()

    # Memory browser
    st.subheader("🧠 Memory Bank")
    try:
        from memory.memory_store import get_all_records
        records = get_all_records(limit=10)
        if records:
            st.caption(f"{len(records)} problems solved")
            for r in reversed(records[-5:]):
                pp = r.get("parsed_problem", {})
                topic = pp.get("topic", "?")
                status = "✅" if r.get("is_correct") else "❌"
                st.markdown(
                    f'<div class="memory-card">{status} [{topic}] '
                    f'{pp.get("problem_text","")[:60]}…</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No problems solved yet.")
    except Exception:
        pass

    st.divider()
    if st.button("🔄 New Problem", use_container_width=True):
        reset_session()
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═════════════════════════════════════════════════════════════════════════════
st.title("🧮 Math Mentor")
st.caption("Powered by Claude · LangGraph · RAG · Multimodal Input")
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: INPUT
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.step == "input":

    tab_text, tab_image, tab_audio = st.tabs(["✏️ Text", "📷 Image / Screenshot", "🎙️ Audio"])

    with tab_text:
        st.subheader("Type your math problem")
        text_input = st.text_area(
            "Problem:",
            height=140,
            placeholder="e.g. Find the probability that two dice show a sum greater than 9…",
            key="text_area",
        )
        if st.button("Solve →", key="solve_text", type="primary", use_container_width=True):
            if text_input.strip():
                st.session_state.raw_input = text_input.strip()
                st.session_state.extracted_text = text_input.strip()
                st.session_state.input_mode = "text"
                st.session_state.step = "solving"
                st.rerun()
            else:
                st.warning("Please enter a problem first.")

    with tab_image:
        st.subheader("Upload an image of a math problem")
        uploaded = st.file_uploader(
            "JPG / PNG / WEBP", type=["jpg", "jpeg", "png", "webp"], key="img_upload"
        )
        if uploaded:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(Image.open(uploaded), caption="Uploaded image", use_column_width=True)
            with col2:
                st.info("👆 Claude will extract the math problem from this image using vision OCR.")

            if st.button("Extract Text →", key="ocr_btn", type="primary"):
                with st.spinner("Extracting text with Claude Vision…"):
                    from input_processing.ocr import extract_from_bytes
                    uploaded.seek(0)
                    suffix = Path(uploaded.name).suffix or ".png"
                    result = extract_from_bytes(uploaded.read(), suffix=suffix)
                    st.session_state.ocr_result = result
                    st.session_state.input_mode = "image"
                    st.session_state.step = "ocr_review"
                    st.rerun()

    with tab_audio:
        st.subheader("Upload or record audio")
        audio_file = st.file_uploader(
            "WAV / MP3 / M4A", type=["wav", "mp3", "m4a", "ogg"], key="audio_upload"
        )
        if audio_file:
            st.audio(audio_file)
            if not os.getenv("OPENAI_API_KEY"):
                st.warning("⚠️ OPENAI_API_KEY not set — Whisper transcription unavailable. "
                           "Please type your problem in the Text tab instead.")
            else:
                if st.button("Transcribe Audio →", key="asr_btn", type="primary"):
                    with st.spinner("Transcribing with Whisper…"):
                        from input_processing.asr import transcribe_bytes
                        audio_file.seek(0)
                        suffix = Path(audio_file.name).suffix or ".wav"
                        result = transcribe_bytes(audio_file.read(), suffix=suffix)
                        st.session_state.asr_result = result
                        st.session_state.input_mode = "audio"
                        st.session_state.step = "asr_review"
                        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2a: OCR REVIEW
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.step == "ocr_review":
    r = st.session_state.ocr_result or {}
    confidence = r.get("confidence", 0.5)
    extracted = r.get("extracted_text", "")

    st.subheader("📋 OCR Extraction Review")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**Extracted Text** (edit if needed):")
    with col2:
        st.markdown(f"Confidence: {confidence_badge(confidence)}", unsafe_allow_html=True)

    if confidence < float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.75")):
        st.markdown(
            '<div class="hitl-banner">⚠️ <b>Low OCR confidence.</b> Please review and correct the extracted text below.</div>',
            unsafe_allow_html=True,
        )
    if r.get("notes"):
        st.info(f"OCR notes: {r['notes']}")

    edited = st.text_area("Extracted problem text:", value=extracted, height=140, key="ocr_edit")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("✅ Confirm & Solve", type="primary", use_container_width=True):
            st.session_state.extracted_text = edited.strip()
            st.session_state.raw_input = edited.strip()
            st.session_state.step = "solving"
            st.rerun()
    with col_b:
        if st.button("← Re-upload", use_container_width=True):
            st.session_state.step = "input"
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2b: ASR REVIEW
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.step == "asr_review":
    r = st.session_state.asr_result or {}
    confidence = r.get("confidence", 0.5)
    transcript = r.get("transcript", "")
    raw_t = r.get("raw_transcript", "")

    st.subheader("🎙️ Transcript Review")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**Transcript** (edit to correct math notation):")
    with col2:
        st.markdown(f"Confidence: {confidence_badge(confidence)}", unsafe_allow_html=True)

    if confidence < float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.75")):
        st.markdown(
            '<div class="hitl-banner">⚠️ <b>Low transcription confidence.</b> Please review and correct the text below.</div>',
            unsafe_allow_html=True,
        )

    if raw_t != transcript:
        with st.expander("Raw ASR output (before math normalisation)"):
            st.code(raw_t)

    if r.get("notes"):
        st.info(r["notes"])

    edited = st.text_area("Transcript:", value=transcript, height=120, key="asr_edit")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("✅ Confirm & Solve", type="primary", use_container_width=True):
            st.session_state.extracted_text = edited.strip()
            st.session_state.raw_input = edited.strip()
            st.session_state.step = "solving"
            st.rerun()
    with col_b:
        if st.button("← Re-record", use_container_width=True):
            st.session_state.step = "input"
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: SOLVING  (run phase 1, check for HITL, run phase 2)
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.step == "solving":

    if st.session_state.phase1_state is None:
        # Phase 1: Parser + Router
        with st.spinner("🤖 Parsing problem…"):
            from agents.graph import run_phase1
            p1 = run_phase1(
                raw_input=st.session_state.raw_input,
                input_mode=st.session_state.input_mode,
                extracted_text=st.session_state.extracted_text,
            )
            st.session_state.phase1_state = p1

    p1 = st.session_state.phase1_state

    # If parser needs HITL, redirect
    if p1.get("needs_hitl") and not p1.get("human_correction"):
        st.session_state.step = "hitl_parse"
        st.rerun()

    # Phase 2: Solver + Verifier + Explainer + Memory
    if st.session_state.phase2_state is None:
        progress = st.progress(0, text="Running agents…")
        steps_ui = st.empty()

        agent_labels = ["Routing", "Solving (RAG)", "Verifying", "Explaining", "Saving to memory"]
        for i, label in enumerate(agent_labels):
            steps_ui.markdown(f"⚙️ **{label}…**")
            progress.progress((i + 1) / len(agent_labels), text=label)
            time.sleep(0.1)

        from agents.graph import run_phase2
        import copy
        p2_state = copy.deepcopy(p1)
        p2_state = run_phase2(p2_state)
        st.session_state.phase2_state = p2_state
        st.session_state.memory_id = p2_state.get("memory_id")

        progress.empty()
        steps_ui.empty()

        # Check if verifier needs HITL
        if p2_state.get("needs_hitl") and not p2_state.get("human_approved"):
            st.session_state.step = "hitl_verify"
            st.rerun()
        else:
            st.session_state.step = "done"
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4a: HITL – Parser clarification
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.step == "hitl_parse":
    p1 = st.session_state.phase1_state or {}

    st.subheader("🔍 Clarification Needed")
    st.markdown(
        f'<div class="hitl-banner">🧑‍💻 <b>Human Review Required</b><br>'
        f'{p1.get("hitl_reason", "The parser needs clarification.")}</div>',
        unsafe_allow_html=True,
    )

    parsed = p1.get("parsed_problem", {})
    st.markdown("**Parsed problem so far:**")
    st.json(parsed)

    st.markdown("**Please correct or clarify the problem statement:**")
    correction = st.text_area(
        "Corrected problem:",
        value=parsed.get("problem_text", st.session_state.extracted_text),
        height=120,
        key="parse_correction",
    )

    if st.button("✅ Submit Correction & Continue", type="primary"):
        if correction.strip():
            # Inject correction into state and re-run parser
            from agents.graph import run_phase1
            p1_new = run_phase1(
                raw_input=correction.strip(),
                input_mode=st.session_state.input_mode,
                extracted_text=correction.strip(),
            )
            p1_new["human_correction"] = correction.strip()
            p1_new["needs_hitl"] = False   # human resolved it
            st.session_state.phase1_state = p1_new
            st.session_state.phase2_state = None  # reset phase 2
            st.session_state.step = "solving"
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4b: HITL – Verifier review
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.step == "hitl_verify":
    p2 = st.session_state.phase2_state or {}

    st.subheader("🔎 Solution Review")
    st.markdown(
        f'<div class="hitl-banner">🧑‍💻 <b>Human Review Required</b><br>'
        f'{p2.get("hitl_reason", "Verifier confidence is low. Please review the solution.")}</div>',
        unsafe_allow_html=True,
    )

    verification = p2.get("verification", {})
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Solution steps:**")
        st.markdown(p2.get("solution", "No solution generated."))
        st.markdown(f"**Final Answer:** `{p2.get('final_answer', '')}`")

    with col2:
        st.markdown("**Verifier report:**")
        st.markdown(f"- Is correct: {'✅' if verification.get('is_correct') else '❌'}")
        st.markdown(f"- Confidence: {verification.get('confidence', 0):.0%}")
        issues = verification.get("issues_found", [])
        if issues:
            st.markdown("- **Issues found:**")
            for issue in issues:
                st.markdown(f"  • {issue}")
        st.markdown(f"- {verification.get('verification_notes','')}")

    st.divider()
    action = st.radio("Your review:", ["✅ Approve solution", "✏️ Provide corrected answer"], key="hitl_action")

    corrected = ""
    if action == "✏️ Provide corrected answer":
        corrected = st.text_area("Your corrected answer / steps:", height=100, key="corrected_ans")

    if st.button("Submit Review", type="primary"):
        import copy
        p2_updated = copy.deepcopy(p2)
        p2_updated["human_approved"] = True
        p2_updated["needs_hitl"] = False
        if corrected.strip():
            p2_updated["human_correction"] = corrected.strip()
            p2_updated["is_correct"] = True

        # Re-run explainer with updated state
        from agents.nodes import explainer_node, memory_node
        p2_updated.update(explainer_node(p2_updated))
        p2_updated.update(memory_node(p2_updated))

        st.session_state.phase2_state = p2_updated
        st.session_state.memory_id = p2_updated.get("memory_id")
        st.session_state.step = "done"
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: DONE – Show results
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.step == "done":
    p2 = st.session_state.phase2_state or {}
    p1 = st.session_state.phase1_state or {}

    # ── Header ───────────────────────────────────────────────────────────────
    parsed = p2.get("parsed_problem", {})
    verifier_confidence = p2.get("verifier_confidence", 0.5)
    is_correct = p2.get("is_correct", False)

    col_h1, col_h2, col_h3 = st.columns([3, 1, 1])
    with col_h1:
        st.subheader(f"📐 {parsed.get('topic', 'Math').title()} Problem")
    with col_h2:
        badge = "✅ Verified" if is_correct else "⚠️ Unverified"
        st.metric("Status", badge)
    with col_h3:
        st.metric("Confidence", f"{verifier_confidence:.0%}")

    st.markdown(f"**Problem:** {parsed.get('problem_text','')}")
    st.divider()

    # ── Main result tabs ──────────────────────────────────────────────────────
    r_tab1, r_tab2, r_tab3, r_tab4 = st.tabs(
        ["💡 Explanation", "🧮 Solution Steps", "📚 RAG Context", "🔍 Agent Trace"]
    )

    with r_tab1:
        explanation_data = p2.get("explanation_data", {})
        if explanation_data:
            intro = explanation_data.get("introduction", "")
            if intro:
                st.markdown(f"*{intro}*")

            key_insight = explanation_data.get("key_insight", "")
            if key_insight:
                st.info(f"🔑 **Key Insight:** {key_insight}")

            steps = explanation_data.get("steps", [])
            for s in steps:
                with st.expander(f"Step {s['step_number']}: {s['title']}", expanded=True):
                    st.markdown(s["content"])

            mistakes = explanation_data.get("common_mistakes", [])
            if mistakes:
                st.warning("⚠️ **Common Mistakes to Avoid:**\n" + "\n".join(f"• {m}" for m in mistakes))

            summary = explanation_data.get("summary", "")
            if summary:
                st.success(f"✅ **Summary:** {summary}")

            tip = explanation_data.get("practice_tip", "")
            if tip:
                st.markdown(f"💪 *Practice tip: {tip}*")
        else:
            st.markdown(p2.get("explanation", "No explanation generated."))

        st.divider()
        final = p2.get("human_correction") or p2.get("final_answer", "")
        st.markdown(f"### 🎯 Final Answer\n**{final}**")

    with r_tab2:
        solution = p2.get("solution", "No solution.")
        st.markdown(solution)

        calc_results = p2.get("calc_results", [])
        if calc_results:
            st.divider()
            st.markdown("**🖩 Calculator Verification:**")
            for calc in calc_results:
                st.code(calc)

        v = p2.get("verification", {})
        if v:
            st.divider()
            st.markdown("**🔍 Verifier Report:**")
            status = "✅ Correct" if v.get("is_correct") else "❌ Issues Found"
            st.markdown(f"- Status: {status}")
            st.markdown(f"- Confidence: {v.get('confidence', 0):.0%}")
            if v.get("issues_found"):
                for issue in v["issues_found"]:
                    st.markdown(f"  • {issue}")
            if v.get("edge_cases_checked"):
                with st.expander("Edge cases checked"):
                    for ec in v["edge_cases_checked"]:
                        st.markdown(f"• {ec}")

    with r_tab3:
        chunks = p2.get("retrieved_chunks", [])
        if chunks:
            st.markdown(f"**Retrieved {len(chunks)} relevant chunks from knowledge base:**")
            for i, chunk in enumerate(chunks, 1):
                score = chunk.get("score", 0)
                source = chunk.get("source", "unknown")
                st.markdown(
                    f'<div class="rag-card"><b>#{i} [{source}]</b> '
                    f'Relevance: {score:.2f}<br><code>{chunk["text"][:300]}{"…" if len(chunk["text"])>300 else ""}</code></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No RAG context retrieved (or retrieval failed).")

        similar = p2.get("similar_problems", [])
        if similar:
            st.divider()
            st.markdown(f"**🧠 {len(similar)} similar past problems from memory:**")
            for s in similar:
                pp = s.get("parsed_problem", {})
                st.markdown(
                    f'<div class="memory-card"><b>[{pp.get("topic","?")}]</b> '
                    f'{pp.get("problem_text","")[:120]}…<br>'
                    f'<i>Answer: {s.get("final_answer","")[:80]}</i></div>',
                    unsafe_allow_html=True,
                )

    with r_tab4:
        trace = p2.get("agent_trace", [])
        if trace:
            for t in trace:
                agent = t.pop("agent", "Agent")
                elapsed = t.pop("elapsed_s", 0)
                st.markdown(
                    f'<div class="agent-card"><b>🤖 {agent}</b> <small>({elapsed}s)</small><br>'
                    + "<br>".join(f"<code>{k}: {v}</code>" for k, v in t.items())
                    + "</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No agent trace available.")

    st.divider()

    # ── Feedback ──────────────────────────────────────────────────────────────
    if not st.session_state.feedback_given:
        st.subheader("💬 Feedback")
        st.markdown("Was this answer helpful and correct?")
        col_fb1, col_fb2, col_fb3 = st.columns([1, 1, 3])

        with col_fb1:
            if st.button("✅ Correct", type="primary", use_container_width=True):
                if st.session_state.memory_id:
                    from memory.memory_store import update_feedback
                    update_feedback(st.session_state.memory_id, "correct", True)
                st.session_state.feedback_given = True
                st.success("Thanks! Saved as a correct example for future learning.")
                st.rerun()

        with col_fb2:
            if st.button("❌ Incorrect", use_container_width=True):
                st.session_state.show_correction = True
                st.rerun()

        if st.session_state.get("show_correction"):
            with col_fb3:
                correction_text = st.text_input("What's the correct answer?", key="feedback_correction")
                if st.button("Submit Correction"):
                    if correction_text:
                        if st.session_state.memory_id:
                            from memory.memory_store import update_feedback
                            update_feedback(st.session_state.memory_id, correction_text, False)
                        st.session_state.feedback_given = True
                        st.warning("Noted! This will help improve future answers.")
                        st.rerun()
    else:
        st.success("✅ Feedback recorded. Thank you!")

    st.divider()
    if st.button("🔄 Solve Another Problem", type="primary"):
        reset_session()
        st.rerun()
