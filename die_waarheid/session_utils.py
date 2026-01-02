"""
Session persistence utilities for Die Waarheid
"""

import json
import os
from pathlib import Path
import streamlit as st

def save_session_data():
    """Save session state to disk."""
    try:
        session_file = Path("session_data.json")
        
        # Prepare data for JSON serialization
        data = {
            'case_id': st.session_state.get('case_id'),
            'evidence_files': st.session_state.get('evidence_files', []),
            'speaker_profiles': st.session_state.get('speaker_profiles', {}),
            'case_data': st.session_state.get('case_data', {}),
            'timestamp': str(st.session_state.get('timestamp', ''))
        }
        
        # Remove non-serializable data
        if 'analysis_results' in st.session_state:
            data['has_analysis'] = True
        
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        st.error(f"Error saving session: {e}")


def load_session_data():
    """Load session state from disk."""
    try:
        session_file = Path("session_data.json")
        
        if session_file.exists():
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Restore session state
            if data.get('case_id'):
                st.session_state.case_id = data['case_id']
            if data.get('evidence_files'):
                st.session_state.evidence_files = data['evidence_files']
            if data.get('speaker_profiles'):
                st.session_state.speaker_profiles = data['speaker_profiles']
            if data.get('case_data'):
                st.session_state.case_data = data['case_data']
            if data.get('timestamp'):
                st.session_state.timestamp = data['timestamp']
                
            return True
    except Exception as e:
        st.error(f"Error loading session: {e}")
    
    return False


def clear_session_data():
    """Clear saved session data."""
    try:
        session_file = Path("session_data.json")
        if session_file.exists():
            session_file.unlink()
    except Exception as e:
        st.error(f"Error clearing session: {e}")
