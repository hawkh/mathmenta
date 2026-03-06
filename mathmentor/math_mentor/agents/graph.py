"""
agents/graph.py
LangGraph pipeline for the Math Mentor multi-agent system.

Graph layout:
  START
    │
  [parser_node]       → structures raw input
    │
  <check_hitl_after_parse>  ─── needs HITL ──→ [hitl_parse_node] ──┐
    │ no HITL                                                        │
    ▼                                                               ▼
  [router_node]       → classifies topic, RAG query, retrieves memory
    │
  [solver_node]       → RAG retrieval + calculation + solution
    │
  [verifier_node]     → checks correctness
    │
  <check_hitl_after_verify>  ─── needs HITL ──→ [hitl_verify_node] ──┐
    │ no HITL                                                          │
    ▼                                                                  ▼
  [explainer_node]    → student-friendly explanation
    │
  [memory_node]       → persist to memory store
    │
  END
"""
from typing import TypedDict, Optional, List, Any

from langgraph.graph import StateGraph, START, END


# ── State definition ─────────────────────────────────────────────────────────
class MathMentorState(TypedDict, total=False):
    # ── Input ────────────────────────────────────────────────────────────────
    input_mode: str          # "text" | "image" | "audio"
    raw_input: str           # original user text (or corrected OCR/ASR)
    extracted_text: str      # after OCR/ASR extraction

    # ── Parser ───────────────────────────────────────────────────────────────
    parsed_problem: dict
    needs_clarification: bool

    # ── Router ───────────────────────────────────────────────────────────────
    routing: dict
    similar_problems: List[dict]

    # ── Solver ───────────────────────────────────────────────────────────────
    retrieved_chunks: List[dict]
    solution: str
    final_answer: str
    solution_data: dict
    calc_results: List[str]

    # ── Verifier ─────────────────────────────────────────────────────────────
    verification: dict
    is_correct: bool
    verifier_confidence: float

    # ── Explainer ────────────────────────────────────────────────────────────
    explanation: str
    explanation_data: dict

    # ── HITL ─────────────────────────────────────────────────────────────────
    needs_hitl: bool
    hitl_reason: str
    human_correction: Optional[str]
    human_approved: bool
    user_feedback: Optional[str]

    # ── Memory ───────────────────────────────────────────────────────────────
    memory_id: Optional[str]

    # ── Meta ─────────────────────────────────────────────────────────────────
    agent_trace: List[dict]
    error: Optional[str]


# ── Conditional edge helpers ─────────────────────────────────────────────────
def _should_hitl_after_parse(state: MathMentorState) -> str:
    """Route to hitl_parse if parser needs clarification, else continue."""
    if state.get("needs_hitl") and not state.get("human_correction"):
        return "hitl_parse"
    return "router"


def _should_hitl_after_verify(state: MathMentorState) -> str:
    """Route to hitl_verify if verifier not confident, else continue."""
    if state.get("needs_hitl") and not state.get("human_approved"):
        return "hitl_verify"
    return "explainer"


# ── Passthrough HITL nodes (actual HITL is handled in Streamlit) ──────────────
def hitl_parse_node(state: MathMentorState) -> dict:
    """
    Placeholder: signals to the orchestrator that human input is needed
    after parsing. In Streamlit, the app reads state.needs_hitl and
    shows a correction form before resuming the graph.
    """
    return {"hitl_stage": "parse"}


def hitl_verify_node(state: MathMentorState) -> dict:
    """
    Placeholder: signals to orchestrator that human review is needed
    after verification.
    """
    return {"hitl_stage": "verify"}


# ── Build graph ──────────────────────────────────────────────────────────────
def build_graph(interrupt_at_hitl: bool = False) -> Any:
    """
    Build and compile the LangGraph state machine.

    Args:
        interrupt_at_hitl: If True, add interrupt_before on HITL nodes
                           (requires MemorySaver checkpointer).
    """
    from agents.nodes import (
        parser_node,
        router_node,
        solver_node,
        verifier_node,
        explainer_node,
        memory_node,
    )

    builder = StateGraph(MathMentorState)

    # Add nodes
    builder.add_node("parser", parser_node)
    builder.add_node("router", router_node)
    builder.add_node("solver", solver_node)
    builder.add_node("verifier", verifier_node)
    builder.add_node("hitl_parse", hitl_parse_node)
    builder.add_node("hitl_verify", hitl_verify_node)
    builder.add_node("explainer", explainer_node)
    builder.add_node("memory", memory_node)

    # Edges
    builder.add_edge(START, "parser")
    builder.add_conditional_edges("parser", _should_hitl_after_parse, {
        "hitl_parse": "hitl_parse",
        "router": "router",
    })
    builder.add_edge("hitl_parse", END)   # Streamlit resumes after HITL
    builder.add_edge("router", "solver")
    builder.add_edge("solver", "verifier")
    builder.add_conditional_edges("verifier", _should_hitl_after_verify, {
        "hitl_verify": "hitl_verify",
        "explainer": "explainer",
    })
    builder.add_edge("hitl_verify", END)  # Streamlit resumes after HITL
    builder.add_edge("explainer", "memory")
    builder.add_edge("memory", END)

    if interrupt_at_hitl:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
        return builder.compile(
            checkpointer=checkpointer,
            interrupt_before=["hitl_parse", "hitl_verify"],
        )

    return builder.compile()


# ── Convenience runners ───────────────────────────────────────────────────────
def run_pipeline(
    raw_input: str,
    input_mode: str = "text",
    extracted_text: Optional[str] = None,
    human_correction: Optional[str] = None,
    human_approved: bool = False,
) -> MathMentorState:
    """
    Run the full pipeline synchronously (no HITL interrupts).
    Human corrections can be injected before running.
    """
    graph = build_graph(interrupt_at_hitl=False)

    initial_state: MathMentorState = {
        "input_mode": input_mode,
        "raw_input": raw_input,
        "extracted_text": extracted_text or raw_input,
        "human_correction": human_correction,
        "human_approved": human_approved,
        "agent_trace": [],
        "similar_problems": [],
        "retrieved_chunks": [],
        "calc_results": [],
        "needs_hitl": False,
        "hitl_reason": "",
    }

    result = graph.invoke(initial_state)
    return result


def run_phase1(raw_input: str, input_mode: str = "text", extracted_text: Optional[str] = None) -> MathMentorState:
    """
    Run only parser + router (phase 1).
    Used when we want to check for HITL before committing to full solve.
    """
    from agents.nodes import parser_node, router_node

    state: MathMentorState = {
        "input_mode": input_mode,
        "raw_input": raw_input,
        "extracted_text": extracted_text or raw_input,
        "agent_trace": [],
        "similar_problems": [],
        "retrieved_chunks": [],
        "calc_results": [],
        "needs_hitl": False,
        "hitl_reason": "",
    }

    state.update(parser_node(state))
    if not state.get("needs_hitl"):
        state.update(router_node(state))

    return state


def run_phase2(state: MathMentorState) -> MathMentorState:
    """
    Run solver + verifier + explainer + memory on an existing state.
    Called after HITL resolution.
    """
    from agents.nodes import solver_node, verifier_node, explainer_node, memory_node, router_node

    # Ensure router has run
    if "routing" not in state:
        state.update(router_node(state))

    state.update(solver_node(state))
    state.update(verifier_node(state))
    # Reset HITL after verification (human already approved or it's auto-passed)
    if state.get("needs_hitl") and state.get("human_approved"):
        state["needs_hitl"] = False
    state.update(explainer_node(state))
    state.update(memory_node(state))

    return state
