"""Simple test to verify Streamlit + Ollama is working"""
import streamlit as st

st.title("🧮 Math Mentor - Test")

st.write("If you can see this, Streamlit is working!")

# Simple input test
user_input = st.text_area(
    "Enter a math problem:",
    placeholder="e.g., Solve x + 2 = 5",
    height=100
)

if st.button("Test"):
    if user_input:
        st.success(f"You entered: {user_input}")
    else:
        st.warning("Please enter something")

st.write("---")
st.write("Sidebar test:")
with st.sidebar:
    st.write("Sidebar is working!")
    mode = st.radio("Mode", ["Text", "Image"])
    st.write(f"Selected: {mode}")
