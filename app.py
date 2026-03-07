"""
Math Mentor - Multi-Agent Mathematical Problem Solving System
Main Streamlit Application
"""
import sys
import os

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')

import streamlit as st
from PIL import Image
import json
from datetime import datetime

from config import Config
from graph import get_agent_graph
from input_processing import get_ocr_processor, get_asr_processor
from memory import get_memory_store
from rag import get_retriever


# Page configuration
st.set_page_config(
    page_title="Math Mentor - AI Problem Solver",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-trace {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .solution-box {
        background-color: #e8f5e9;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
    }
    .confidence-meter {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .confidence-high { background-color: #c8e6c9; color: #2e7d32; }
    .confidence-medium { background-color: #fff9c4; color: #f57f17; }
    .confidence-low { background-color: #ffcdd2; color: #c62828; }
</style>
""", unsafe_allow_html=True)


def initialize_session():
    """Initialize session state variables."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.current_state = None
        st.session_state.processing_complete = False
        st.session_state.awaiting_clarification = False
        st.session_state.awaiting_verification = False
        st.session_state.session_history = []


def render_sidebar():
    """Render the sidebar with input options and settings."""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Input mode selection
        input_mode = st.radio(
            "Input Mode",
            ["Text", "Image (OCR)", "Audio (Speech)"],
            help="Choose how you want to input the problem"
        )
        
        st.divider()
        
        # Display configuration info
        st.subheader("📊 System Status")
        
        # Check API key
        if Config.ANTHROPIC_API_KEY:
            st.success("✓ Anthropic API configured")
        else:
            st.error("✗ Anthropic API key missing")
        
        # Check vector store
        try:
            retriever = get_retriever()
            if retriever.load_index():
                st.success("✓ Knowledge base loaded")
            else:
                st.warning("⚠ Run build_index.py first")
        except Exception:
            st.warning("⚠ Knowledge base not built")
        
        # Check OpenAI for ASR
        if Config.OPENAI_API_KEY:
            st.success("✓ OpenAI API configured (ASR)")
        else:
            st.info("ℹ️ OpenAI API not set (ASR unavailable)")
        
        st.divider()
        
        # Memory statistics
        st.subheader("💾 Memory Stats")
        try:
            memory = get_memory_store()
            stats = memory.get_statistics()
            st.metric("Total Sessions", stats['total_sessions'])
            st.metric("Success Rate", f"{stats['success_rate']*100:.1f}%")
            st.metric("Avg Confidence", f"{stats['average_confidence']:.2f}")
        except Exception:
            st.info("No memory data yet")
        
        st.divider()
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            st.slider(
                "OCR Confidence Threshold",
                0.5, 1.0, Config.OCR_CONFIDENCE_THRESHOLD,
                0.05
            )
            st.slider(
                "Verifier Threshold",
                0.5, 1.0, Config.VERIFIER_CONFIDENCE_THRESHOLD,
                0.05
            )
        
        return input_mode


def render_text_input():
    """Render text input area."""
    st.subheader("📝 Enter Your Problem")
    text_input = st.text_area(
        "Type or paste your mathematical problem:",
        height=150,
        placeholder="e.g., Solve the quadratic equation: x² - 5x + 6 = 0"
    )
    return text_input


def render_image_input():
    """Render image upload interface."""
    st.subheader("📷 Upload Problem Image")
    uploaded_file = st.file_uploader(
        "Upload an image of the problem",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of the mathematical problem"
    )
    
    if uploaded_file is not None:
        # Display the image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.image(image, caption="Uploaded image", use_container_width=True)
        with col2:
            st.caption(f"Size: {uploaded_file.size} bytes")
            st.caption(f"Type: {uploaded_file.type}")
        return uploaded_file
    
    return None


def render_audio_input():
    """Render audio upload interface."""
    st.subheader("🎤 Upload Audio Recording")
    
    if not Config.OPENAI_API_KEY:
        st.warning(
            "⚠️ Audio input requires OpenAI API key. "
            "Please add OPENAI_API_KEY to your .env file."
        )
        return None
    
    uploaded_file = st.file_uploader(
        "Upload an audio recording",
        type=['wav', 'mp3', 'm4a', 'webm'],
        help="Record yourself reading the problem"
    )
    
    if uploaded_file is not None:
        st.audio(uploaded_file)
        return uploaded_file
    
    return None


def process_input(input_data, input_type: str) -> dict:
    """
    Process input through OCR/ASR if needed, then run agent pipeline.
    
    Args:
        input_data: Text string, or uploaded file
        input_type: 'text', 'image', or 'audio'
        
    Returns:
        State dictionary from agent pipeline
    """
    # Handle different input types
    if input_type == 'image':
        with st.spinner("🔍 Extracting text from image..."):
            ocr_processor = get_ocr_processor()
            image = Image.open(input_data)
            ocr_result = ocr_processor.process_image(image)
            
            if not ocr_result['success']:
                st.error(f"OCR failed: {ocr_result['error']}")
                return None
            
            # Show extraction result
            st.info(f"📄 Extracted text (confidence: {ocr_result['confidence']:.2f})")
            extracted_text = st.text_area(
                "Review and edit the extracted text:",
                value=ocr_result['extracted_text'],
                height=200
            )
            
            # Check if human review is needed
            if ocr_result['needs_review']:
                st.warning("⚠️ Low confidence extraction. Please verify the text above.")
            
            raw_input = extracted_text
            
    elif input_type == 'audio':
        with st.spinner("🎧 Transcribing audio..."):
            asr_processor = get_asr_processor()
            audio_data = input_data.getvalue()
            asr_result = asr_processor.process_audio(audio_data)
            
            if not asr_result['success']:
                st.error(f"ASR failed: {asr_result['error']}")
                return None
            
            # Show transcription result
            st.info(f"📄 Transcribed text (confidence: {asr_result['confidence']:.2f})")
            extracted_text = st.text_area(
                "Review and edit the transcribed text:",
                value=asr_result['extracted_text'],
                height=200
            )
            
            # Check if human review is needed
            if asr_result['needs_review']:
                st.warning("⚠️ Low confidence transcription. Please verify the text above.")
            
            raw_input = extracted_text
            
    else:  # text
        raw_input = input_data
    
    # Run agent pipeline
    with st.spinner("🤖 AI agents solving your problem..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            graph = get_agent_graph()
            
            # Run step-by-step to show progress
            states = []
            for i, state in enumerate(graph.run_step_by_step(raw_input, input_type)):
                states.append(state)
                progress = min((i + 1) / 5, 1.0)
                progress_bar.progress(progress)
                
                # Update status
                agent_trace = state.get('agent_trace', [])
                if agent_trace:
                    last_agent = agent_trace[-1].get('agent', 'unknown')
                    status_text.text(f"✓ {last_agent.capitalize()} completed")
            
            final_state = states[-1] if states else None
            
            # Store in session
            st.session_state.current_state = final_state
            st.session_state.processing_complete = True
            
            return final_state
            
        except Exception as e:
            st.error(f"Error processing: {str(e)}")
            return None


def render_results(state: dict):
    """Render the solution results."""
    st.header("📚 Solution")
    
    # Display confidence scores
    st.subheader("Confidence Scores")
    cols = st.columns(3)
    
    parser_conf = state.get('parser_confidence', 0)
    solver_conf = state.get('solver_confidence', 0)
    verifier_conf = state.get('verifier_confidence', 0)
    
    def confidence_class(conf):
        if conf >= 0.8:
            return "confidence-high"
        elif conf >= 0.6:
            return "confidence-medium"
        else:
            return "confidence-low"
    
    with cols[0]:
        st.metric("Parser", f"{parser_conf:.2f}")
    with cols[1]:
        st.metric("Solver", f"{solver_conf:.2f}")
    with cols[2]:
        st.metric("Verifier", f"{verifier_conf:.2f}")
    
    # Display solution
    st.divider()
    st.subheader("📝 Final Answer")
    
    final_answer = state.get('final_answer', 'No answer generated')
    st.markdown(f"""
    <div class="solution-box">
        <strong>Answer:</strong> {final_answer}
    </div>
    """, unsafe_allow_html=True)
    
    # Display formatted solution
    st.divider()
    st.subheader("🔢 Step-by-Step Solution")
    formatted_solution = state.get('formatted_solution', '')
    st.markdown(formatted_solution)
    
    # Display explanation
    st.divider()
    st.subheader("💡 Explanation")
    explanation = state.get('explanation', {})
    
    if explanation:
        # Key insights
        insights = explanation.get('key_insights', [])
        if insights:
            st.markdown("**Key Insights:**")
            for insight in insights:
                st.markdown(f"• {insight}")
        
        # Learning tips
        tips = explanation.get('learning_tips', [])
        if tips:
            st.markdown("**Learning Tips:**")
            for tip in tips:
                st.markdown(f"• {tip}")
        
        # Common mistakes
        mistakes = explanation.get('common_mistakes', [])
        if mistakes:
            st.markdown("**Common Mistakes to Avoid:**")
            for mistake in mistakes:
                st.markdown(f"• {mistake}")
    
    # Display RAG context
    st.divider()
    st.subheader("📖 Referenced Knowledge")
    rag_context = state.get('rag_context', '')
    if rag_context:
        with st.expander("View retrieved context"):
            st.markdown(rag_context)
    
    # Display agent trace
    st.divider()
    st.subheader("🔍 Agent Execution Trace")
    agent_trace = state.get('agent_trace', [])
    if agent_trace:
        with st.expander("View detailed agent trace"):
            for trace in agent_trace:
                st.json(trace)
    
    # Session info
    session_id = state.get('session_id')
    if session_id:
        st.success(f"✓ Session saved with ID: `{session_id}`")


def render_human_review(state: dict):
    """Render human-in-the-loop review interface."""
    st.warning("⚠️ Human Review Required")
    
    review_reason = state.get('review_reason', 'Low confidence in solution')
    st.info(f"Reason: {review_reason}")
    
    # Show the proposed solution
    st.subheader("Proposed Solution")
    st.markdown(state.get('formatted_solution', ''))
    st.markdown(f"**Final Answer:** {state.get('final_answer', '')}")
    
    # Verification details
    verification = state.get('verification', {})
    st.subheader("Verification Results")
    st.json(verification)
    
    # User actions
    st.subheader("Your Decision")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✓ Accept Solution", use_container_width=True):
            # Mark as accepted and continue
            st.session_state.human_approved = True
            st.rerun()
    
    with col2:
        if st.button("✗ Provide Correction", use_container_width=True):
            st.session_state.human_correction_mode = True
            st.rerun()
    
    # Correction input
    if st.session_state.get('human_correction_mode', False):
        correction = st.text_area(
            "Enter the correct solution or feedback:",
            height=200
        )
        if st.button("Submit Correction"):
            st.session_state.human_correction = correction
            st.session_state.human_correction_mode = False
            st.rerun()


def main():
    """Main application function."""
    # Initialize session
    initialize_session()

    # Render sidebar first
    input_mode = render_sidebar()

    # Render header
    st.markdown('<h1 class="main-header">🧮 Math Mentor</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Multi-Agent AI System for Mathematical Problem Solving</p>',
        unsafe_allow_html=True
    )

    # Main content area
    if not st.session_state.processing_complete:
        # Input section - show immediately after header
        st.divider()
        if input_mode == "Text":
            user_input = render_text_input()
            input_type = 'text'
        elif input_mode == "Image (OCR)":
            uploaded_file = render_image_input()
            user_input = uploaded_file
            input_type = 'image'
        else:  # Audio
            uploaded_file = render_audio_input()
            user_input = uploaded_file
            input_type = 'audio'
        
        # Process button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "🚀 Solve Problem",
                type="primary",
                use_container_width=True,
                disabled=(user_input is None if input_type != 'text' else not user_input)
            ):
                if input_type == 'text' and not user_input.strip():
                    st.warning("Please enter a problem")
                else:
                    state = process_input(user_input, input_type)
                    if state:
                        st.rerun()
        
        # Show recent sessions
        st.divider()
        st.subheader("📜 Recent Sessions")
        try:
            memory = get_memory_store()
            recent = memory.get_recent_sessions(5)
            if recent:
                for session in recent:
                    with st.expander(
                        f"{session.get('topic', 'unknown').title()} - "
                        f"{session.get('input', '')[:50]}..."
                    ):
                        st.write(f"**Answer:** {session.get('final_answer', 'N/A')}")
                        st.write(f"**Success:** {'✓' if session.get('success') else '✗'}")
                        st.write(f"**Time:** {session.get('timestamp', 'Unknown')}")
            else:
                st.info("No recent sessions")
        except Exception:
            pass
    
    else:
        # Results view
        state = st.session_state.current_state
        
        # Check if human review is needed
        if state.get('needs_human_review', False):
            render_human_review(state)
        else:
            render_results(state)
        
        # Action buttons
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Solve Another", use_container_width=True):
                st.session_state.current_state = None
                st.session_state.processing_complete = False
                st.rerun()
        
        with col2:
            if st.button("📋 Copy Answer", use_container_width=True):
                st.code(state.get('final_answer', ''))
        
        with col3:
            # Rating
            rating = st.selectbox(
                "Rate this solution",
                ["", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐", "⭐"],
                key="rating"
            )
            if rating:
                session_id = state.get('session_id')
                if session_id:
                    memory = get_memory_store()
                    rating_value = len(rating) - rating.count('⭐')
                    memory.update_session_feedback(session_id, rating_value)
                    st.success("Feedback saved!")


if __name__ == "__main__":
    main()
