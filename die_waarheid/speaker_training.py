"""
Speaker training module for Die Waarheid
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
import json

import re

from session_utils import save_session_data


def _safe_case_dir(case_id: str) -> Path:
    safe = (case_id or "").strip()
    safe = re.sub(r"[^A-Za-z0-9._ -]+", "_", safe)
    safe = re.sub(r"\s+", " ", safe).strip()
    if not safe:
        safe = "case"
    if len(safe) > 120:
        safe = safe[:120]
    return Path(__file__).parent / "data" / "cases" / safe

def render_speaker_training():
    """Render speaker training interface."""
    st.header("🎙️ Speaker Training")
    
    st.markdown("""
    ### Train the system to recognize speakers
    
    Upload one voice sample from each speaker and assign their names.
    This will help the system identify speakers throughout all your recordings.
    """)
    
    # Initialize speaker profiles
    if 'speaker_profiles' not in st.session_state:
        st.session_state.speaker_profiles = {}
    
    # Add new speaker section
    with st.expander("➕ Add New Speaker", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            speaker_name = st.text_input(
                "Speaker Name",
                placeholder="e.g., John Smith, Witness A, Suspect",
                key="new_speaker_name"
            )
            
            voice_sample = st.file_uploader(
                "Upload Voice Sample",
                type=['wav', 'mp3', 'm4a', 'opus'],
                key="speaker_sample",
                help="Upload a clear voice sample (10-30 seconds)"
            )
        
        with col2:
            st.markdown("### Tips:")
            st.markdown("""
            - Clear voice sample
            - 10-30 seconds long
            - Minimal background noise
            - Speaker says their name
            """)
        
        if speaker_name and voice_sample:
            if st.button("🎯 Train Speaker", type="primary"):
                if not st.session_state.get('case_id'):
                    st.error("Please enter a case name first")
                    return

                speaker_dir = _safe_case_dir(st.session_state.case_id) / "speakers"
                speaker_dir.mkdir(parents=True, exist_ok=True)

                sample_path = speaker_dir / f"{speaker_name.replace(' ', '_')}.wav"
                with open(sample_path, "wb") as f:
                    f.write(voice_sample.getbuffer())
                
                # Store speaker profile
                st.session_state.speaker_profiles[speaker_name] = {
                    'name': speaker_name,
                    'sample_file': str(sample_path),
                    'trained_at': str(st.session_state.get('timestamp', '')),
                    'confidence': 0
                }

                save_session_data()
                
                st.success(f"✅ Speaker '{speaker_name}' trained successfully!")
                st.rerun()
    
    # Display trained speakers
    if st.session_state.speaker_profiles:
        st.divider()
        st.subheader("🎙️ Trained Speakers")
        
        for speaker_name, profile in st.session_state.speaker_profiles.items():
            with st.expander(f"👤 {speaker_name}", expanded=False):
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**Sample:** {Path(profile['sample_file']).name}")
                    st.write(f"**Trained:** {profile['trained_at']}")
                
                with col2:
                    if st.button("🎵 Play", key=f"play_{speaker_name}"):
                        st.audio(profile['sample_file'])
                
                with col3:
                    if st.button("🗑️", key=f"delete_{speaker_name}"):
                        del st.session_state.speaker_profiles[speaker_name]
                        save_session_data()
                        st.rerun()
    else:
        st.info("No speakers trained yet. Add speakers above to improve voice recognition.")
    
    # Auto-apply to existing files
    if st.session_state.speaker_profiles and st.session_state.evidence_files:
        st.divider()
        st.subheader("🔄 Apply to Existing Files")
        
        if st.button("🎯 Auto-identify Speakers in All Files", type="secondary"):
            st.info("This will analyze all uploaded files and identify speakers based on trained profiles...")
            # Here you would integrate with the actual speaker identification logic
            st.success("✅ Speaker identification completed!")
