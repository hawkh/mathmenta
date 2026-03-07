"""
LangGraph agent nodes for the Math Mentor multi-agent system.
Each node represents a specialized agent in the problem-solving pipeline.
"""
import re
import json
from typing import Dict, Any, List, Optional
from langchain.schema import HumanMessage, SystemMessage
from config import Config
from rag.retriever import get_retriever
from memory.store import get_memory_store

# Import appropriate LLM based on configuration
if Config.USE_OLLAMA:
    from langchain_ollama import ChatOllama
    print(f"[Ollama] Using model: {Config.DEFAULT_MODEL}")
else:
    from langchain_anthropic import ChatAnthropic
    print(f"[Anthropic] Using model: {Config.DEFAULT_MODEL}")


class AgentNodes:
    """
    Collection of specialized agent nodes for mathematical problem solving.
    Each node has a specific role in the pipeline.
    """

    def __init__(self):
        """Initialize agent nodes with LLM client."""
        if Config.USE_OLLAMA:
            self.llm = ChatOllama(
                model=Config.DEFAULT_MODEL,
                base_url=Config.OLLAMA_BASE_URL,
                temperature=0.3,
                num_predict=2048
            )
        else:
            self.llm = ChatAnthropic(
                model=Config.DEFAULT_MODEL,
                temperature=0.3,
                max_tokens=2048
            )
        self.retriever = get_retriever()
        self.memory = get_memory_store()
    
    def parser_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parser Agent: Cleans and structures input data.
        
        Responsibilities:
        - Remove OCR/ASR noise
        - Extract mathematical expressions
        - Identify problem components (given, find, constraints)
        - Structure into standardized format
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with parsed input
        """
        raw_input = state.get('raw_input', '')
        input_type = state.get('input_type', 'text')
        
        system_prompt = """You are a Parser Agent for a mathematical problem-solving system.
Your task is to clean and structure the input.

Extract and identify:
1. The main question/problem statement
2. Given information and variables
3. Constraints or conditions
4. What needs to be found/proved

Format the output as JSON:
{
    "cleaned_text": "Clean, clear version of the problem",
    "topic": "algebra|calculus|probability|linear_algebra|geometry|other",
    "variables": ["list", "of", "variables"],
    "given": ["list", "of", "given", "information"],
    "find": "What needs to be found",
    "constraints": ["list", "of", "constraints"],
    "ambiguities": ["any", "unclear", "parts"],
    "confidence": 0.0-1.0
}

Be precise and maintain mathematical accuracy."""

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Parse this problem:\n\n{raw_input}")
            ])
            
            # Parse JSON response
            response_text = response.content
            parsed = self._extract_json(response_text)
            
            # Check for ambiguities
            needs_clarification = len(parsed.get('ambiguities', [])) > 0
            confidence = parsed.get('confidence', 0.8)
            
            return {
                **state,
                'parsed_input': parsed,
                'topic': parsed.get('topic', 'unknown'),
                'needs_clarification': needs_clarification,
                'parser_confidence': confidence,
                'agent_trace': state.get('agent_trace', []) + [{
                    'agent': 'parser',
                    'action': 'parsed_input',
                    'confidence': confidence,
                    'ambiguities': parsed.get('ambiguities', [])
                }]
            }
            
        except Exception as e:
            return {
                **state,
                'parsed_input': {'cleaned_text': raw_input, 'topic': 'unknown'},
                'needs_clarification': True,
                'parser_confidence': 0.5,
                'agent_trace': state.get('agent_trace', []) + [{
                    'agent': 'parser',
                    'action': 'parse_error',
                    'error': str(e)
                }]
            }
    
    def router_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Router Agent: Classifies problem and determines strategy.
        
        Responsibilities:
        - Classify problem type and difficulty
        - Query memory for similar problems
        - Determine solving strategy
        - Select relevant knowledge domains
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with routing information
        """
        parsed = state.get('parsed_input', {})
        topic = state.get('topic', 'unknown')
        cleaned_text = parsed.get('cleaned_text', '')
        
        system_prompt = """You are a Router Agent for a mathematical problem-solving system.
Your task is to classify the problem and determine the solving strategy.

Analyze:
1. Problem type (computation, proof, application, etc.)
2. Difficulty level (basic, intermediate, advanced)
3. Required concepts and formulas
4. Optimal solving approach

Query memory for similar problems and retrieve relevant context.

Format the output as JSON:
{
    "problem_type": "computation|proof|application|analysis",
    "difficulty": "basic|intermediate|advanced",
    "strategy": "Description of solving approach",
    "required_concepts": ["list", "of", "concepts"],
    "similar_problems_found": true|false,
    "memory_results": ["brief descriptions of similar problems"]
}"""

        try:
            # Query memory for similar problems
            similar = self.memory.get_similar_problems(cleaned_text, topic, limit=3)
            memory_results = [s.get('input', '')[:100] + "..." for s in similar]
            
            # Get RAG context
            rag_context = self.retriever.retrieve_with_context(cleaned_text, topic)
            
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""Problem to route:
{cleaned_text}

Similar problems from memory: {memory_results if memory_results else 'None found'}

Relevant knowledge:
{rag_context}""")
            ])
            
            routed = self._extract_json(response.content)
            
            return {
                **state,
                'routed_info': routed,
                'problem_type': routed.get('problem_type', 'computation'),
                'difficulty': routed.get('difficulty', 'intermediate'),
                'strategy': routed.get('strategy', ''),
                'similar_problems': similar,
                'rag_context': rag_context,
                'agent_trace': state.get('agent_trace', []) + [{
                    'agent': 'router',
                    'action': 'routed_problem',
                    'problem_type': routed.get('problem_type'),
                    'similar_found': len(similar) > 0
                }]
            }
            
        except Exception as e:
            return {
                **state,
                'routed_info': {'problem_type': 'computation', 'difficulty': 'intermediate'},
                'problem_type': 'computation',
                'difficulty': 'intermediate',
                'strategy': 'Standard problem-solving approach',
                'similar_problems': [],
                'rag_context': '',
                'agent_trace': state.get('agent_trace', []) + [{
                    'agent': 'router',
                    'action': 'route_error',
                    'error': str(e)
                }]
            }
    
    def solver_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solver Agent: Generates the solution.
        
        Responsibilities:
        - Apply appropriate solving methods
        - Use RAG-retrieved knowledge
        - Perform calculations (via tools)
        - Generate step-by-step solution
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with solution
        """
        parsed = state.get('parsed_input', {})
        strategy = state.get('strategy', '')
        rag_context = state.get('rag_context', '')
        similar_problems = state.get('similar_problems', [])
        
        system_prompt = """You are a Solver Agent for a mathematical problem-solving system.
Your task is to solve the problem step-by-step.

Guidelines:
1. Show all work clearly
2. Explain each step
3. Use appropriate mathematical notation (LaTeX)
4. Verify calculations
5. Check that the answer makes sense

Format the output as JSON:
{
    "solution_steps": [
        {"step": 1, "description": "...", "math": "..."},
        {"step": 2, "description": "...", "math": "..."}
    ],
    "final_answer": "The final answer",
    "method_used": "Description of method",
    "confidence": 0.0-1.0,
    "verification_notes": "Any verification performed"
}"""

        try:
            # Format similar problems for context
            similar_context = ""
            if similar_problems:
                similar_context = "\nSimilar solved problems:\n"
                for sp in similar_problems:
                    similar_context += f"- {sp.get('input', '')}\n  Solution: {sp.get('solution', '')[:200]}...\n"
            
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""Solve this problem:

Problem: {parsed.get('cleaned_text', '')}

Strategy: {strategy}

Relevant knowledge:
{rag_context}
{similar_context}

Provide a complete step-by-step solution.""")
            ])
            
            solved = self._extract_json(response.content)
            
            # Format solution steps for display
            solution_steps = solved.get('solution_steps', [])
            formatted_solution = self._format_solution_steps(solution_steps)
            
            return {
                **state,
                'solution': solved,
                'formatted_solution': formatted_solution,
                'final_answer': solved.get('final_answer', ''),
                'solver_confidence': solved.get('confidence', 0.8),
                'agent_trace': state.get('agent_trace', []) + [{
                    'agent': 'solver',
                    'action': 'generated_solution',
                    'confidence': solved.get('confidence', 0.8),
                    'steps_count': len(solution_steps)
                }]
            }
            
        except Exception as e:
            return {
                **state,
                'solution': {'final_answer': f'Error: {str(e)}'},
                'formatted_solution': f'Error generating solution: {str(e)}',
                'final_answer': '',
                'solver_confidence': 0.3,
                'agent_trace': state.get('agent_trace', []) + [{
                    'agent': 'solver',
                    'action': 'solve_error',
                    'error': str(e)
                }]
            }
    
    def verifier_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verifier Agent: Independently verifies the solution.
        
        Responsibilities:
        - Check solution correctness
        - Verify domain constraints
        - Identify edge cases
        - Assess confidence level
        - Trigger HITL if confidence is low
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with verification results
        """
        parsed = state.get('parsed_input', {})
        solution = state.get('solution', {})
        
        system_prompt = """You are a Verifier Agent for a mathematical problem-solving system.
Your task is to independently verify the solution.

Check:
1. Mathematical correctness of each step
2. Domain constraints and assumptions
3. Edge cases and special conditions
4. Units and dimensional analysis (if applicable)
5. Reasonableness of the answer

Format the output as JSON:
{
    "is_correct": true|false,
    "verification_steps": [
        {"check": "...", "result": "pass|fail", "notes": "..."}
    ],
    "issues_found": ["list", "of", "issues"],
    "confidence": 0.0-1.0,
    "needs_human_review": true|false,
    "review_reason": "Why human review is needed (if applicable)"
}"""

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""Verify this solution:

Problem: {parsed.get('cleaned_text', '')}

Proposed Solution:
{solution.get('formatted_solution', '')}

Final Answer: {solution.get('final_answer', '')}

Independently verify the solution and identify any issues.""")
            ])
            
            verified = self._extract_json(response.content)
            
            # Check if human review is needed
            verifier_confidence = verified.get('confidence', 0.8)
            needs_human_review = (
                verifier_confidence < Config.VERIFIER_CONFIDENCE_THRESHOLD or
                verified.get('needs_human_review', False) or
                not verified.get('is_correct', True)
            )
            
            return {
                **state,
                'verification': verified,
                'is_correct': verified.get('is_correct', True),
                'verifier_confidence': verifier_confidence,
                'needs_human_review': needs_human_review,
                'review_reason': verified.get('review_reason', ''),
                'agent_trace': state.get('agent_trace', []) + [{
                    'agent': 'verifier',
                    'action': 'verified_solution',
                    'is_correct': verified.get('is_correct', True),
                    'confidence': verifier_confidence,
                    'needs_human_review': needs_human_review
                }]
            }
            
        except Exception as e:
            return {
                **state,
                'verification': {'is_correct': False, 'confidence': 0.3},
                'is_correct': False,
                'verifier_confidence': 0.3,
                'needs_human_review': True,
                'review_reason': f'Verification error: {str(e)}',
                'agent_trace': state.get('agent_trace', []) + [{
                    'agent': 'verifier',
                    'action': 'verify_error',
                    'error': str(e)
                }]
            }
    
    def explainer_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explainer Agent: Creates student-friendly explanation.
        
        Responsibilities:
        - Generate clear, numbered explanation
        - Highlight key insights and concepts
        - Provide alternative approaches
        - Add learning tips
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with explanation
        """
        parsed = state.get('parsed_input', {})
        solution = state.get('solution', {})
        verification = state.get('verification', {})
        
        system_prompt = """You are an Explainer Agent for a mathematical problem-solving system.
Your task is to create a clear, educational explanation.

Include:
1. Problem restatement
2. Step-by-step solution with reasoning
3. Key insights and "aha!" moments
4. Alternative approaches (if any)
5. Learning tips and related concepts
6. Common mistakes to avoid

Format the output as JSON:
{
    "restatement": "Clear restatement of the problem",
    "explanation": "Numbered explanation with steps",
    "key_insights": ["list", "of", "key", "insights"],
    "alternative_approaches": ["other", "methods"],
    "learning_tips": ["tips", "for", "students"],
    "common_mistakes": ["mistakes", "to", "avoid"]
}"""

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""Create an educational explanation for:

Problem: {parsed.get('cleaned_text', '')}

Solution:
{solution.get('formatted_solution', '')}

Verification: {"Correct" if verification.get('is_correct', True) else "Issues found"}

Make the explanation clear and helpful for a student learning this concept.""")
            ])
            
            explained = self._extract_json(response.content)
            
            return {
                **state,
                'explanation': explained,
                'agent_trace': state.get('agent_trace', []) + [{
                    'agent': 'explainer',
                    'action': 'generated_explanation',
                    'insights_count': len(explained.get('key_insights', [])),
                    'tips_count': len(explained.get('learning_tips', []))
                }]
            }
            
        except Exception as e:
            return {
                **state,
                'explanation': {'explanation': f'Error generating explanation: {str(e)}'},
                'agent_trace': state.get('agent_trace', []) + [{
                    'agent': 'explainer',
                    'action': 'explain_error',
                    'error': str(e)
                }]
            }
    
    def memory_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Memory Agent: Persists the session to storage.
        
        Responsibilities:
        - Save complete session data
        - Index for future retrieval
        - Update statistics
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with session ID
        """
        try:
            # Prepare session data
            session_data = {
                'input': state.get('raw_input', ''),
                'input_type': state.get('input_type', 'text'),
                'topic': state.get('topic', 'unknown'),
                'parsed_input': state.get('parsed_input', {}),
                'solution': state.get('solution', {}).get('formatted_solution', ''),
                'final_answer': state.get('final_answer', ''),
                'explanation': state.get('explanation', {}).get('explanation', ''),
                'retrieved_context': state.get('rag_context', ''),
                'agent_trace': state.get('agent_trace', []),
                'confidence_scores': {
                    'parser': state.get('parser_confidence', 0),
                    'solver': state.get('solver_confidence', 0),
                    'verifier': state.get('verifier_confidence', 0)
                },
                'human_feedback': state.get('human_feedback'),
                'success': state.get('is_correct', True)
            }
            
            # Save to memory
            session_id = self.memory.save_session(session_data)
            
            return {
                **state,
                'session_id': session_id,
                'agent_trace': state.get('agent_trace', []) + [{
                    'agent': 'memory',
                    'action': 'saved_session',
                    'session_id': session_id
                }]
            }
            
        except Exception as e:
            return {
                **state,
                'session_id': None,
                'agent_trace': state.get('agent_trace', []) + [{
                    'agent': 'memory',
                    'action': 'save_error',
                    'error': str(e)
                }]
            }
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response text."""
        # Try to find JSON in the response
        json_patterns = [
            r'\{[^{}]*\}',  # Simple JSON object
            r'\{[\s\S]*?\}',  # Multi-line JSON object
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        # If JSON extraction fails, return default structure
        return {
            'error': 'Failed to parse JSON response',
            'raw_response': text
        }
    
    def _format_solution_steps(self, steps: List[Dict]) -> str:
        """Format solution steps for display."""
        if not steps:
            return "No solution steps available."
        
        formatted = []
        for step in steps:
            step_num = step.get('step', 0)
            description = step.get('description', '')
            math = step.get('math', '')
            
            formatted.append(f"**Step {step_num}:** {description}")
            if math:
                formatted.append(f"  {math}")
        
        return "\n\n".join(formatted)


# Singleton instance
_nodes_instance = None


def get_agent_nodes() -> AgentNodes:
    """Get or create the agent nodes singleton."""
    global _nodes_instance
    if _nodes_instance is None:
        _nodes_instance = AgentNodes()
    return _nodes_instance
