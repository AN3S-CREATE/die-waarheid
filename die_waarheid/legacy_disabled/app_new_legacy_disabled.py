import streamlit as st
import os
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from src.processor import process_whatsapp_folder, enrich_audio_forensics
from src.brain import analyze_conversation
from src.forensics import analyze_audio_file

# Page configuration
st.set_page_config(
    page_title="🔬 DIE WAARHEID: Forensic WhatsApp Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for forensic theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f1f1f;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stProgress .progress-bar {
        background-color: #ff4b4b;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .stress-high { color: #ff4b4b; font-weight: bold; }
    .stress-medium { color: #ffa500; font-weight: bold; }
    .stress-low { color: #00ff00; font-weight: bold; }
    .contradiction-severe { background-color: #ffebee; border-left: 5px solid #f44336; }
    .contradiction-moderate { background-color: #fff8e1; border-left: 5px solid #ff9800; }
    .contradiction-minor { background-color: #e8f5e8; border-left: 5px solid #4caf50; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_data' not in st.session_state:
    st.session_state.chat_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = {'stage': 'idle', 'progress': 0, 'current_file': ''}

def calculate_stress_index(row: pd.Series) -> float:
    """Calculate forensic stress index using the specified formula."""
    pitch_vol = row.get('pitch_volatility', 0) or 0
    silence_ratio = row.get('silence_ratio', 0) or 0
    energy_spikes = row.get('max_loudness', 0) or 0
    
    # Normalize energy spikes to 0-100 scale (assuming max ~1.0)
    energy_normalized = min(energy_spikes * 100, 100)
    
    stress_index = (pitch_vol * 0.4) + (silence_ratio * 0.3) + (energy_normalized * 0.3)
    return min(stress_index, 100)  # Cap at 100

def classify_emotional_state(stress_index: float) -> str:
    """Classify emotional state based on stress index."""
    if stress_index < 30:
        return "Calm"
    elif stress_index < 60:
        return "Anxious"
    elif stress_index < 80:
        return "Highly Stressed"
    else:
        return "Panic/Rage"

def detect_contradictions(df: pd.DataFrame) -> pd.DataFrame:
    """Detect contradictions between text claims and audio evidence."""
    contradictions = []
    
    for idx, row in df.iterrows():
        if row.get('message_type') != 'audio' or not row.get('transcript'):
            continue
            
        transcript = row['transcript'].lower()
        speaker_count = row.get('speaker_count', 1) or 1
        
        # Check for "alone" claims vs multiple speakers
        if any(word in transcript for word in ['alone', 'by myself', 'solo']) and speaker_count > 1:
            contradictions.append({
                'timestamp': row['timestamp'],
                'claim_type': 'Location vs Audio',
                'description': f"Claimed to be alone → Audio: {speaker_count} speakers detected",
                'severity': 'Severe',
                'evidence_link': row.get('audio_path', ''),
                'transcript': row['transcript']
            })
        
        # Check for "quiet" vs high stress
        stress_idx = calculate_stress_index(row)
        if any(word in transcript for word in ['calm', 'fine', 'okay']) and stress_idx > 70:
            contradictions.append({
                'timestamp': row['timestamp'],
                'claim_type': 'Emotional State vs Bio-signals',
                'description': f"Claimed to be calm → Stress index: {stress_idx:.1f}",
                'severity': 'Moderate',
                'evidence_link': row.get('audio_path', ''),
                'transcript': row['transcript']
            })
    
    return pd.DataFrame(contradictions)

def render_sidebar():
    """Render the forensic control panel sidebar."""
    st.sidebar.markdown("# 🔬 DIE WAARHEID")
    st.sidebar.markdown("### Forensic Control Panel")
    
    # API Keys Section
    st.sidebar.markdown("#### 🔑 API Configuration")
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password", 
                                      value=os.getenv("GEMINI_API_KEY", ""))
    hf_token = st.sidebar.text_input("HuggingFace Token", type="password",
                                    value=os.getenv("HUGGINGFACE_TOKEN", ""))
    
    # Update environment variables
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
    if hf_token:
        os.environ["HUGGINGFACE_TOKEN"] = hf_token
    
    st.sidebar.markdown("---")
    
    # Data Input Section
    st.sidebar.markdown("#### 📁 Evidence Input")
    input_method = st.sidebar.selectbox(
        "Input Method",
        ["Upload ZIP", "Local ZIP Path", "Case Folder"]
    )
    
    if input_method == "Upload ZIP":
        uploaded_file = st.sidebar.file_uploader(
            "Upload WhatsApp Export ZIP",
            type=['zip'],
            help="Upload a WhatsApp export file"
        )
        if uploaded_file and st.sidebar.button("🔍 Analyze Upload"):
            process_uploaded_file(uploaded_file)
    
    elif input_method == "Local ZIP Path":
        zip_path = st.sidebar.text_input("Local ZIP file path")
        if zip_path and st.sidebar.button("🔍 Analyze Local ZIP"):
            process_local_zip(zip_path)
    
    elif input_method == "Case Folder":
        folder_path = st.sidebar.text_input("Case folder path")
        if folder_path and st.sidebar.button("🔍 Analyze Case Folder"):
            process_case_folder(folder_path)
    
    st.sidebar.markdown("---")
    
    # Processing Status
    st.sidebar.markdown("#### 📊 Processing Status")
    status = st.session_state.processing_status
    progress_bar = st.sidebar.progress(status['progress'] / 100)
    st.sidebar.write(f"**Stage:** {status['stage']}")
    if status['current_file']:
        st.sidebar.write(f"**Current:** {status['current_file']}")
    
    # Quick Stats
    if st.session_state.chat_data is not None:
        st.sidebar.markdown("---")
        st.sidebar.markdown("#### 📈 Quick Stats")
        df = st.session_state.chat_data
        total_messages = len(df)
        audio_messages = len(df[df['message_type'] == 'audio'])
        st.sidebar.write(f"**Total Messages:** {total_messages}")
        st.sidebar.write(f"**Audio Files:** {audio_messages}")
        
        if 'analysis_successful' in df.columns:
            successful = df['analysis_successful'].sum()
            st.sidebar.write(f"**Processed:** {successful}/{audio_messages}")

def process_uploaded_file(uploaded_file):
    """Process uploaded ZIP file."""
    st.session_state.processing_status = {'stage': 'Extracting...', 'progress': 10, 'current_file': uploaded_file.name}
    st.rerun()

def process_local_zip(zip_path):
    """Process local ZIP file."""
    st.session_state.processing_status = {'stage': 'Extracting...', 'progress': 10, 'current_file': zip_path}
    st.rerun()

def process_case_folder(folder_path):
    """Process case folder with multiple ZIPs."""
    st.session_state.processing_status = {'stage': 'Scanning folder...', 'progress': 5, 'current_file': folder_path}
    st.rerun()

def render_main_dashboard():
    """Render the main forensic dashboard."""
    st.markdown('<div class="main-header">🔬 DIE WAARHEID FORENSIC DASHBOARD</div>', unsafe_allow_html=True)
    
    if st.session_state.chat_data is None:
        st.markdown("### 📋 Welcome to Die Waarheid")
        st.markdown("""
        **Complete Forensic WhatsApp Analysis System**
        
        Please load evidence using the control panel on the left to begin forensic analysis.
        
        **Capabilities:**
        - 🔍 Multi-layer forensic analysis
        - 📊 Stress pattern detection
        - 🎯 Contradiction identification
        - 🗣️ Speaker diarization
        - 🧠 Psychological profiling
        """)
        return
    
    df = st.session_state.chat_data
    
    # Calculate forensic metrics
    if 'pitch_volatility' in df.columns:
        df['stress_index'] = df.apply(calculate_stress_index, axis=1)
        df['emotional_state'] = df['stress_index'].apply(classify_emotional_state)
    
    # Executive Summary Cards
    st.markdown("## 📊 Executive Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_messages = len(df)
        st.metric("Total Messages", total_messages)
    
    with col2:
        audio_count = len(df[df['message_type'] == 'audio'])
        st.metric("Voice Notes", audio_count)
    
    with col3:
        if 'stress_index' in df.columns:
            avg_stress = df['stress_index'].mean()
            stress_color = "stress-high" if avg_stress > 60 else "stress-medium" if avg_stress > 30 else "stress-low"
            st.markdown(f"<div class='metric-card'><h3>Avg Stress</h3><h2 class='{stress_color}'>{avg_stress:.1f}</h2></div>", unsafe_allow_html=True)
    
    with col4:
        contradictions = detect_contradictions(df)
        st.metric("Contradictions", len(contradictions), delta="🚨" if len(contradictions) > 0 else "✅")
    
    # Phase 1 Visualizations
    st.markdown("## 📈 Phase 1: Core Analytics")
    
    # 1. Message Volume Chart
    st.markdown("### 📊 Message Volume Over Time")
    if 'timestamp' in df.columns:
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily_counts = df.groupby(['date', 'sender']).size().reset_index(name='count')
        
        fig_volume = px.bar(
            daily_counts, 
            x='date', 
            y='count', 
            color='sender',
            title="Daily Message Volume by Sender",
            labels={'count': 'Messages', 'date': 'Date'}
        )
        st.plotly_chart(fig_volume, use_container_width=True)
    
    # 2. Sentiment Polarity Over Time
    st.markdown("### 📈 Sentiment Analysis Timeline")
    if 'timestamp' in df.columns and 'message' in df.columns:
        # Simple sentiment analysis (placeholder for now)
        df['sentiment'] = df['message'].apply(lambda x: 0.5 if isinstance(x, str) and len(x) > 0 else 0)
        
        fig_sentiment = px.line(
            df, 
            x='timestamp', 
            y='sentiment',
            title="Sentiment Polarity Over Time",
            labels={'sentiment': 'Sentiment Score (-1 to 1)', 'timestamp': 'Time'}
        )
        fig_sentiment.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    # 3. Basic Timeline View
    st.markdown("### 📅 Communication Timeline")
    if 'timestamp' in df.columns:
        fig_timeline = px.scatter(
            df,
            x='timestamp',
            y='sender',
            color='message_type',
            title="Communication Timeline by Message Type",
            labels={'timestamp': 'Time', 'sender': 'Sender'}
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Contradictions Table (if any detected)
    contradictions = detect_contradictions(df)
    if not contradictions.empty:
        st.markdown("## 🚨 Contradictions Detected")
        for _, row in contradictions.iterrows():
            severity_class = f"contradiction-{row['severity'].lower()}"
            st.markdown(f"""
            <div class='{severity_class}' style='padding: 1rem; margin: 0.5rem 0; border-radius: 5px;'>
                <strong>{row['timestamp']}</strong> - {row['claim_type']}<br>
                {row['description']}<br>
                <em>Transcript: "{row['transcript'][:100]}..."</em>
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main application entry point."""
    render_sidebar()
    render_main_dashboard()

if __name__ == "__main__":
    main()
