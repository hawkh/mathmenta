"""
LangGraph orchestration for the Math Mentor multi-agent system.
Defines the graph structure and execution flow.
"""
from typing import Dict, Any, List, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from agents.nodes import get_agent_nodes, AgentNodes


class AgentState(TypedDict):
    """
    State schema for the agent graph.
    Tracks all information as it flows through the pipeline.
    """
    # Input
    raw_input: str
    input_type: str  # 'text', 'image', 'audio'
    
    # Parsed input
    parsed_input: Dict[str, Any]
    topic: str
    needs_clarification: bool
    
    # Routing
    routed_info: Dict[str, Any]
    problem_type: str
    difficulty: str
    strategy: str
    similar_problems: List[Dict]
    rag_context: str
    
    # Solution
    solution: Dict[str, Any]
    formatted_solution: str
    final_answer: str
    solver_confidence: float
    
    # Verification
    verification: Dict[str, Any]
    is_correct: bool
    verifier_confidence: float
    needs_human_review: bool
    review_reason: str
    
    # Explanation
    explanation: Dict[str, Any]
    
    # Memory
    session_id: str
    
    # Human-in-the-loop
    human_feedback: Dict[str, Any]
    
    # Tracking
    agent_trace: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]


class MathMentorGraph:
    """
    LangGraph-based orchestration for the Math Mentor agent system.
    
    Flow:
    Parser → Router → Solver → Verifier → [HITL if needed] → Explainer → Memory
    """
    
    def __init__(self):
        """Initialize the graph with agent nodes."""
        self.nodes = get_agent_nodes()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph."""
        builder = StateGraph(AgentState)
        
        # Add nodes
        builder.add_node("parser", self.nodes.parser_node)
        builder.add_node("router", self.nodes.router_node)
        builder.add_node("solver", self.nodes.solver_node)
        builder.add_node("verifier", self.nodes.verifier_node)
        builder.add_node("explainer", self.nodes.explainer_node)
        builder.add_node("memory", self.nodes.memory_node)
        
        # Set entry point
        builder.set_entry_point("parser")
        
        # Define edges
        builder.add_edge("parser", "router")
        builder.add_edge("router", "solver")
        builder.add_edge("solver", "verifier")
        
        # Conditional edge after verifier
        builder.add_conditional_edges(
            "verifier",
            self._route_after_verifier,
            {
                "human_review": END,  # End for human review (handled by UI)
                "continue": "explainer"
            }
        )

        builder.add_edge("explainer", "memory")
        builder.add_edge("memory", END)

        return builder.compile()
    
    def _route_after_verifier(self, state: AgentState) -> str:
        """Determine next step after verification."""
        if state.get('needs_human_review', False):
            return "human_review"
        return "continue"
    
    def run(self, raw_input: str, input_type: str = 'text') -> AgentState:
        """
        Run the complete agent pipeline.
        
        Args:
            raw_input: The input problem (text, or extracted from image/audio)
            input_type: Type of input ('text', 'image', 'audio')
            
        Returns:
            Final state with all results
        """
        # Initialize state
        initial_state = AgentState(
            raw_input=raw_input,
            input_type=input_type,
            parsed_input={},
            topic='unknown',
            needs_clarification=False,
            routed_info={},
            problem_type='computation',
            difficulty='intermediate',
            strategy='',
            similar_problems=[],
            rag_context='',
            solution={},
            formatted_solution='',
            final_answer='',
            solver_confidence=0.0,
            verification={},
            is_correct=True,
            verifier_confidence=0.0,
            needs_human_review=False,
            review_reason='',
            explanation={},
            session_id='',
            human_feedback={},
            agent_trace=[],
            confidence_scores={}
        )
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        return final_state
    
    def run_with_human_feedback(
        self,
        raw_input: str,
        input_type: str = 'text',
        human_feedback: Dict[str, Any] = None
    ) -> AgentState:
        """
        Run the pipeline with human feedback incorporated.
        
        Args:
            raw_input: The input problem
            input_type: Type of input
            human_feedback: Feedback from human reviewer
            
        Returns:
            Final state with all results
        """
        # Initialize state with human feedback
        initial_state = AgentState(
            raw_input=raw_input,
            input_type=input_type,
            parsed_input={},
            topic='unknown',
            needs_clarification=False,
            routed_info={},
            problem_type='computation',
            difficulty='intermediate',
            strategy='',
            similar_problems=[],
            rag_context='',
            solution={},
            formatted_solution='',
            final_answer='',
            solver_confidence=0.0,
            verification={},
            is_correct=True,
            verifier_confidence=0.0,
            needs_human_review=False,
            review_reason='',
            explanation={},
            session_id='',
            human_feedback=human_feedback or {},
            agent_trace=[],
            confidence_scores={}
        )
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        return final_state
    
    def run_step_by_step(self, raw_input: str, input_type: str = 'text'):
        """
        Generator that yields state after each node.
        Useful for streaming progress to the UI.
        
        Args:
            raw_input: The input problem
            input_type: Type of input
            
        Yields:
            State after each node execution
        """
        initial_state = AgentState(
            raw_input=raw_input,
            input_type=input_type,
            parsed_input={},
            topic='unknown',
            needs_clarification=False,
            routed_info={},
            problem_type='computation',
            difficulty='intermediate',
            strategy='',
            similar_problems=[],
            rag_context='',
            solution={},
            formatted_solution='',
            final_answer='',
            solver_confidence=0.0,
            verification={},
            is_correct=True,
            verifier_confidence=0.0,
            needs_human_review=False,
            review_reason='',
            explanation={},
            session_id='',
            human_feedback={},
            agent_trace=[],
            confidence_scores={}
        )
        
        # Run through each node and yield state
        current_state = initial_state
        
        # Parser
        current_state = self.nodes.parser_node(current_state)
        yield current_state
        
        # Router
        current_state = self.nodes.router_node(current_state)
        yield current_state
        
        # Solver
        current_state = self.nodes.solver_node(current_state)
        yield current_state
        
        # Verifier
        current_state = self.nodes.verifier_node(current_state)
        yield current_state
        
        # Check if human review needed
        if current_state.get('needs_human_review', False):
            yield current_state
            return
        
        # Explainer
        current_state = self.nodes.explainer_node(current_state)
        yield current_state
        
        # Memory
        current_state = self.nodes.memory_node(current_state)
        yield current_state


# Singleton instance
_graph_instance = None


def get_agent_graph() -> MathMentorGraph:
    """Get or create the agent graph singleton."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = MathMentorGraph()
    return _graph_instance
