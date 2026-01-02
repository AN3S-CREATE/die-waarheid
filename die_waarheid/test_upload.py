"""
Simple test for Streamlit file upload functionality
"""

import streamlit as st
import tempfile
import os

st.title("🧪 File Upload Test")

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# File uploader
uploaded = st.file_uploader(
    "Drop files here",
    type=['txt', 'pdf', 'wav', 'mp3', 'opus'],
    accept_multiple_files=True,
    key="test_uploader"
)

if uploaded:
    st.success(f"✅ {len(uploaded)} file(s) detected")
    
    for file in uploaded:
        st.write(f"📄 {file.name} ({file.size} bytes)")
        
        # Try to save
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp:
                tmp.write(file.getbuffer())
                st.success(f"✅ Saved to: {tmp.name}")
                os.unlink(tmp.name)  # Clean up
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Show session state
st.subheader("Session State")
st.json(st.session_state.uploaded_files)
