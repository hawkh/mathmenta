"""
agents/nodes.py
All five agent node functions for the Math Mentor LangGraph pipeline.

Agents:
  1. parser_node      – Clean OCR/ASR output → structured problem dict
  2. router_node      – Classify topic & plan solution strategy
  3. solver_node      – Solve using RAG context + Python calculator
  4. verifier_node    – Check correctness, flag HITL if needed
  5. explainer_node   – Generate student-friendly step-by-step explanation
  6. memory_node      – Retrieve similar past problems & persist result
"""
import json
import os
import re
import time
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from memory.memory_store import retrieve_similar
from rag.vector_store import get_retriever, retrieve
from utils.tools import safe_calculate

# ── LLM singleton ────────────────────────────────────────────────────────────
_llm: Optional[ChatAnthropic] = None


def get_llm() -> ChatAnthropic:
    global _llm
    if _llm is None:
        _llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
            max_tokens=2048,
            temperature=0.1,
        )
    return _llm


def _call_llm(system: str, user: str) -> str:
    """Call Claude with a system + user message; return response text."""
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    return response.content


def _parse_json_response(text: str) -> dict:
    """Strip markdown fences and parse JSON."""
    text = re.sub(r"```json|```", "", text).strip()
    return json.loads(text)


# ═════════════════════════════════════════════════════════════════════════════
# 1. PARSER AGENT
# ═════════════════════════════════════════════════════════════════════════════
def parser_node(state: dict) -> dict:
    """
    Convert raw text (from OCR/ASR/typed) into a clean, structured problem dict.
    Sets needs_hitl=True if clarification is needed.
    """
    trace = state.get("agent_trace", [])
    start = time.time()

    raw = state.get("extracted_text") or state.get("raw_input", "")

    system = """You are a mathematics problem parser for JEE-level math.
Your job is to clean OCR/speech output and extract a structured problem definition.

Topics in scope: algebra, probability, calculus, linear_algebra, arithmetic, geometry_basic.

ALWAYS respond with ONLY valid JSON matching this schema (no extra text, no markdown):
{
  "problem_text": "<cleaned, complete problem statement>",
  "topic": "<most specific topic: algebra|probability|calculus|linear_algebra|arithmetic|geometry_basic>",
  "variables": ["list", "of", "key", "variables"],
  "constraints": ["any constraints like x>0, n is integer, etc."],
  "what_to_find": "<what the student must compute or prove>",
  "needs_clarification": <true|false>,
  "clarification_reason": "<why clarification is needed, or empty string>"
}"""

    user = f"""Parse this math problem input (may have OCR/ASR noise):

\"\"\"{raw}\"\"\"

Return structured JSON."""

    try:
        raw_response = _call_llm(system, user)
        parsed = _parse_json_response(raw_response)
    except Exception as e:
        parsed = {
            "problem_text": raw,
            "topic": "unknown",
            "variables": [],
            "constraints": [],
            "what_to_find": "unknown",
            "needs_clarification": True,
            "clarification_reason": f"Parser error: {e}",
        }

    needs_hitl = (
        parsed.get("needs_clarification", False)
        or parsed.get("topic") == "unknown"
        or len(parsed.get("problem_text", "")) < 10
    )

    trace.append({
        "agent": "Parser",
        "input_length": len(raw),
        "topic_detected": parsed.get("topic"),
        "needs_clarification": parsed.get("needs_clarification"),
        "elapsed_s": round(time.time() - start, 2),
    })

    return {
        "parsed_problem": parsed,
        "needs_hitl": needs_hitl,
        "hitl_reason": parsed.get("clarification_reason", "") if needs_hitl else "",
        "agent_trace": trace,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 2. INTENT ROUTER AGENT
# ═════════════════════════════════════════════════════════════════════════════
def router_node(state: dict) -> dict:
    """
    Classify problem type and choose optimal solution strategy.
    Also queries memory for similar past problems.
    """
    trace = state.get("agent_trace", [])
    start = time.time()

    parsed = state.get("parsed_problem", {})
    topic = parsed.get("topic", "general")
    problem_text = parsed.get("problem_text", "")

    # Retrieve similar past problems from memory
    similar = retrieve_similar(problem_text, topic=topic, top_k=2)

    system = """You are an intent router for a math mentor AI.
Given a parsed math problem, decide the best solution strategy.

Respond ONLY with valid JSON (no markdown):
{
  "strategy": "<direct_formula|algebraic_manipulation|calculus|counting|matrix_ops|bayes|optimization|other>",
  "key_concepts": ["list of relevant math concepts"],
  "suggested_tools": ["calculator", "rag_lookup"],
  "difficulty": "<easy|medium|hard>",
  "estimated_steps": <integer 1-10>,
  "rag_query": "<best search query to find relevant formulas/methods from knowledge base>"
}"""

    user = f"""Route this problem:
Topic: {topic}
Problem: {problem_text}
Variables: {parsed.get('variables', [])}
Constraints: {parsed.get('constraints', [])}
What to find: {parsed.get('what_to_find', '')}"""

    try:
        raw_response = _call_llm(system, user)
        routing = _parse_json_response(raw_response)
    except Exception as e:
        routing = {
            "strategy": "other",
            "key_concepts": [],
            "suggested_tools": ["calculator", "rag_lookup"],
            "difficulty": "medium",
            "estimated_steps": 5,
            "rag_query": problem_text[:200],
        }

    trace.append({
        "agent": "Router",
        "strategy": routing.get("strategy"),
        "difficulty": routing.get("difficulty"),
        "estimated_steps": routing.get("estimated_steps"),
        "similar_problems_found": len(similar),
        "elapsed_s": round(time.time() - start, 2),
    })

    return {
        "routing": routing,
        "similar_problems": similar,
        "agent_trace": trace,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 3. SOLVER AGENT
# ═════════════════════════════════════════════════════════════════════════════
def solver_node(state: dict) -> dict:
    """
    Solve the problem using:
    - Retrieved RAG context (relevant formulas/templates)
    - Python calculator for numerical verification
    - Memory of similar solved problems
    """
    trace = state.get("agent_trace", [])
    start = time.time()

    parsed = state.get("parsed_problem", {})
    routing = state.get("routing", {})
    similar = state.get("similar_problems", [])

    # ── RAG retrieval ────────────────────────────────────────────────────────
    rag_query = routing.get("rag_query", parsed.get("problem_text", ""))
    try:
        index, chunks, model = get_retriever()
        top_k = int(os.getenv("RAG_TOP_K", "5"))
        retrieved = retrieve(rag_query, index, chunks, model, top_k=top_k)
    except Exception as e:
        retrieved = []
        print(f"RAG retrieval failed: {e}")

    rag_context = "\n\n---\n\n".join(
        f"[Source: {c['source']} | Relevance: {c['score']:.2f}]\n{c['text']}"
        for c in retrieved
    )

    # ── Memory context ───────────────────────────────────────────────────────
    memory_ctx = ""
    if similar:
        parts = []
        for s in similar[:2]:
            pp = s.get("parsed_problem", {})
            parts.append(
                f"Similar problem: {pp.get('problem_text', 'N/A')}\n"
                f"Solution pattern: {s.get('solution', 'N/A')[:300]}"
            )
        memory_ctx = "\n\n".join(parts)

    system = """You are an expert JEE math tutor and solver.
Solve the given problem rigorously, showing every step.

Rules:
- Use ONLY the provided RAG context for formulas; do not hallucinate.
- If you use Python calculator results, show the expression evaluated.
- Be precise: exact fractions where possible, decimal if needed.
- Identify any edge cases or domain restrictions.
- Your solution must be reproducible step-by-step.

Respond ONLY with valid JSON (no markdown):
{
  "solution_steps": [
    "Step 1: ...",
    "Step 2: ...",
    ...
  ],
  "final_answer": "<concise final answer>",
  "calculation_expressions": ["expression1", "expression2"],
  "formulas_used": ["formula1", "formula2"],
  "sources_cited": ["source_name1"],
  "solver_confidence": <float 0.0-1.0>
}"""

    user = f"""PROBLEM:
{parsed.get("problem_text", "")}

Topic: {parsed.get("topic")} | Strategy: {routing.get("strategy")} | Difficulty: {routing.get("difficulty")}
What to find: {parsed.get("what_to_find", "")}
Key concepts: {routing.get("key_concepts", [])}

RAG CONTEXT (use these formulas):
{rag_context or "No relevant context retrieved."}

SIMILAR SOLVED PROBLEMS (for pattern reference):
{memory_ctx or "No similar problems found."}"""

    try:
        raw_response = _call_llm(system, user)
        solution_data = _parse_json_response(raw_response)
    except Exception as e:
        solution_data = {
            "solution_steps": [f"Solver error: {e}"],
            "final_answer": "Unable to solve.",
            "calculation_expressions": [],
            "formulas_used": [],
            "sources_cited": [],
            "solver_confidence": 0.1,
        }

    # ── Run calculator verification on any expressions ───────────────────────
    calc_results = []
    for expr in solution_data.get("calculation_expressions", [])[:5]:
        calc_result = safe_calculate(expr)
        if calc_result["success"]:
            calc_results.append(f"{expr} = {calc_result['result']}")

    solution_text = "\n".join(solution_data.get("solution_steps", []))
    final_answer = solution_data.get("final_answer", "")

    trace.append({
        "agent": "Solver",
        "strategy_used": routing.get("strategy"),
        "rag_chunks_retrieved": len(retrieved),
        "rag_sources": list({c["source"] for c in retrieved}),
        "calc_checks": calc_results,
        "solver_confidence": solution_data.get("solver_confidence", 0.5),
        "elapsed_s": round(time.time() - start, 2),
    })

    return {
        "retrieved_chunks": retrieved,
        "solution": solution_text,
        "final_answer": final_answer,
        "solution_data": solution_data,
        "calc_results": calc_results,
        "agent_trace": trace,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 4. VERIFIER / CRITIC AGENT
# ═════════════════════════════════════════════════════════════════════════════
def verifier_node(state: dict) -> dict:
    """
    Independently verify the solution.
    Checks: correctness, domain constraints, units, edge cases.
    Triggers HITL if verifier_confidence < threshold.
    """
    trace = state.get("agent_trace", [])
    start = time.time()

    parsed = state.get("parsed_problem", {})
    solution_data = state.get("solution_data", {})
    solution = state.get("solution", "")
    final_answer = state.get("final_answer", "")

    threshold = float(os.getenv("SOLVER_CONFIDENCE_THRESHOLD", "0.70"))

    system = """You are a rigorous math verifier and critic.
Your task is to independently verify a given solution.

Check for:
1. Mathematical correctness (re-derive key steps)
2. Domain restrictions (e.g., sqrt of negative, log of zero)
3. Unit consistency
4. Edge cases (what if a variable is 0, negative, etc.)
5. Arithmetic errors

Respond ONLY with valid JSON (no markdown):
{
  "is_correct": <true|false>,
  "confidence": <float 0.0-1.0>,
  "issues_found": ["list of issues if any"],
  "corrected_answer": "<corrected answer if wrong, else empty>",
  "edge_cases_checked": ["list of edge cases considered"],
  "verification_notes": "<brief summary of your check>"
}"""

    user = f"""PROBLEM:
{parsed.get("problem_text", "")}
What to find: {parsed.get("what_to_find", "")}
Constraints: {parsed.get("constraints", [])}

SOLUTION TO VERIFY:
{solution}

FINAL ANSWER CLAIMED:
{final_answer}

FORMULAS USED:
{solution_data.get("formulas_used", [])}"""

    try:
        raw_response = _call_llm(system, user)
        verification = _parse_json_response(raw_response)
    except Exception as e:
        verification = {
            "is_correct": False,
            "confidence": 0.3,
            "issues_found": [f"Verifier error: {e}"],
            "corrected_answer": "",
            "edge_cases_checked": [],
            "verification_notes": "Verification failed due to error.",
        }

    verifier_confidence = float(verification.get("confidence", 0.5))
    needs_hitl = verifier_confidence < threshold or not verification.get("is_correct", True)

    hitl_reason = ""
    if needs_hitl:
        issues = verification.get("issues_found", [])
        if issues:
            hitl_reason = f"Verifier found issues: {'; '.join(issues)}"
        else:
            hitl_reason = f"Low verifier confidence ({verifier_confidence:.0%}). Human review recommended."

    trace.append({
        "agent": "Verifier",
        "is_correct": verification.get("is_correct"),
        "confidence": verifier_confidence,
        "issues": verification.get("issues_found", []),
        "needs_hitl": needs_hitl,
        "elapsed_s": round(time.time() - start, 2),
    })

    return {
        "verification": verification,
        "is_correct": verification.get("is_correct", False),
        "verifier_confidence": verifier_confidence,
        "needs_hitl": needs_hitl,
        "hitl_reason": hitl_reason,
        "agent_trace": trace,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 5. EXPLAINER / TUTOR AGENT
# ═════════════════════════════════════════════════════════════════════════════
def explainer_node(state: dict) -> dict:
    """
    Generate a student-friendly, step-by-step explanation.
    Adapts to the difficulty level and topic.
    """
    trace = state.get("agent_trace", [])
    start = time.time()

    parsed = state.get("parsed_problem", {})
    solution = state.get("solution", "")
    final_answer = state.get("final_answer", "")
    verification = state.get("verification", {})
    human_correction = state.get("human_correction")

    # Use corrected answer if human provided one
    actual_answer = human_correction or final_answer

    system = """You are a friendly, encouraging JEE math tutor.
Write a clear, student-friendly explanation of the solution.

Guidelines:
- Use numbered steps (1, 2, 3...)
- Define every formula you use the first time
- Highlight the KEY INSIGHT in capital letters
- Point out common mistakes to avoid
- End with a one-sentence summary of the approach
- Assume the student knows high-school math but may be learning JEE tricks

Respond ONLY with valid JSON (no markdown):
{
  "introduction": "<1-2 sentences setting up the approach>",
  "steps": [
    {"step_number": 1, "title": "short title", "content": "detailed explanation"},
    ...
  ],
  "key_insight": "<the most important conceptual insight>",
  "common_mistakes": ["mistake to avoid 1", "mistake 2"],
  "summary": "<one-sentence summary of method>",
  "practice_tip": "<one tip for similar problems>"
}"""

    user = f"""PROBLEM: {parsed.get("problem_text", "")}
TOPIC: {parsed.get("topic")}
SOLUTION STEPS: {solution}
FINAL ANSWER: {actual_answer}
VERIFICATION NOTES: {verification.get("verification_notes", "")}
ISSUES TO ADDRESS: {verification.get("issues_found", [])}"""

    try:
        raw_response = _call_llm(system, user)
        explanation_data = _parse_json_response(raw_response)
    except Exception as e:
        explanation_data = {
            "introduction": "Let me walk you through this problem.",
            "steps": [{"step_number": 1, "title": "Solution", "content": solution}],
            "key_insight": "Apply the relevant formula.",
            "common_mistakes": [],
            "summary": "See solution above.",
            "practice_tip": "Practice similar problems.",
        }

    # Format as readable text for display
    intro = explanation_data.get("introduction", "")
    steps_text = "\n".join(
        f"**Step {s['step_number']}: {s['title']}**\n{s['content']}"
        for s in explanation_data.get("steps", [])
    )
    explanation_md = f"{intro}\n\n{steps_text}"

    trace.append({
        "agent": "Explainer",
        "steps_generated": len(explanation_data.get("steps", [])),
        "elapsed_s": round(time.time() - start, 2),
    })

    return {
        "explanation": explanation_md,
        "explanation_data": explanation_data,
        "agent_trace": trace,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 6. MEMORY NODE  (saves result + retrieves patterns)
# ═════════════════════════════════════════════════════════════════════════════
def memory_node(state: dict) -> dict:
    """Save this solved problem to memory for future retrieval."""
    from memory.memory_store import save_problem

    try:
        record_id = save_problem(
            input_mode=state.get("input_mode", "text"),
            raw_input=state.get("raw_input", ""),
            parsed_problem=state.get("parsed_problem", {}),
            retrieved_chunks=state.get("retrieved_chunks", []),
            solution=state.get("solution", ""),
            explanation=state.get("explanation", ""),
            verifier_confidence=state.get("verifier_confidence", 0.0),
            is_correct=state.get("is_correct", False),
            user_feedback=state.get("user_feedback"),
            human_correction=state.get("human_correction"),
        )
    except Exception as e:
        record_id = None
        print(f"Memory save failed: {e}")

    return {"memory_id": record_id}
