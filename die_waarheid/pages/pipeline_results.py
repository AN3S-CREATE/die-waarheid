"""
Pipeline Results Page
Chronological table view of all voice note analysis results
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline_processor import PipelineProcessor
from config import AUDIO_DIR

st.set_page_config(page_title="Pipeline Results", page_icon="📊", layout="wide")

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = PipelineProcessor()

if 'results_df' not in st.session_state:
    st.session_state.results_df = None

st.title("📊 Forensic Analysis Pipeline - Results Table")
st.markdown("**Chronological view of all voice note analysis with AI interpretation**")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Pipeline Controls")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload Voice Notes",
        type=['mp3', 'wav', 'opus', 'ogg', 'm4a', 'aac'],
        accept_multiple_files=True
    )
    
    # Or select from audio directory
    st.divider()
    st.subheader("Or Process Existing Files")
    
    audio_files = list(AUDIO_DIR.rglob("*.mp3")) + \
                  list(AUDIO_DIR.rglob("*.wav")) + \
                  list(AUDIO_DIR.rglob("*.opus")) + \
                  list(AUDIO_DIR.rglob("*.ogg"))
    
    st.metric("Available Files", len(audio_files))
    
    process_count = st.number_input(
        "Number of files to process",
        min_value=1,
        max_value=len(audio_files) if audio_files else 1,
        value=min(10, len(audio_files)) if audio_files else 1
    )
    
    # Settings
    st.divider()
    language = st.selectbox(
        "Language",
        ["af", "en", "nl"],
        format_func=lambda x: {"af": "Afrikaans", "en": "English", "nl": "Dutch"}[x]
    )
    
    model_size = st.selectbox(
        "Whisper Model",
        ["tiny", "small", "medium", "large"],
        index=1
    )
    
    # Process button
    if st.button("🚀 Run Pipeline", type="primary", use_container_width=True):
        with st.spinner("Processing through pipeline..."):
            files_to_process = []
            
            # Handle uploaded files
            if uploaded_files:
                import tempfile
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                        tmp.write(uploaded_file.read())
                        files_to_process.append(Path(tmp.name))
            
            # Or use existing files
            elif audio_files:
                files_to_process = audio_files[:process_count]
            
            if files_to_process:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                for i, file_path in enumerate(files_to_process):
                    status_text.text(f"Processing {i+1}/{len(files_to_process)}: {file_path.name}")
                    result = st.session_state.pipeline.process_voice_note(
                        file_path,
                        language=language,
                        model_size=model_size
                    )
                    results.append(result)
                    progress_bar.progress((i + 1) / len(files_to_process))
                
                # Convert to DataFrame
                st.session_state.results_df = pd.DataFrame(results)
                st.success(f"✅ Processed {len(results)} files!")
                st.rerun()
    
    # Export button
    if st.session_state.results_df is not None:
        st.divider()
        if st.button("💾 Export Results", use_container_width=True):
            output_path = Path("pipeline_results.json")
            st.session_state.pipeline.export_results(output_path)
            st.success(f"Exported to {output_path}")

# Main content area
if st.session_state.results_df is not None and len(st.session_state.results_df) > 0:
    df = st.session_state.results_df
    
    # Summary stats at top
    st.header("📈 Summary Statistics")
    stats = st.session_state.pipeline.get_summary_stats()
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Total Files", stats.get('total_files', 0))
    with col2:
        st.metric("High Risk", stats.get('high_risk_count', 0), 
                 delta=None if stats.get('high_risk_count', 0) == 0 else "⚠️")
    with col3:
        st.metric("Deception", stats.get('deception_count', 0),
                 delta=None if stats.get('deception_count', 0) == 0 else "🚨")
    with col4:
        st.metric("Gaslighting", stats.get('gaslighting_count', 0),
                 delta=None if stats.get('gaslighting_count', 0) == 0 else "⚠️")
    with col5:
        st.metric("Avg Stress", f"{stats.get('avg_stress_level', 0):.1f}%")
    with col6:
        st.metric("Avg Risk", f"{stats.get('avg_risk_score', 0):.1f}")
    
    st.divider()
    
    # Results table
    st.header("📋 Chronological Analysis Results")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        show_high_risk_only = st.checkbox("Show High Risk Only (>70)")
    with col2:
        show_deception_only = st.checkbox("Show Deception Detected Only")
    with col3:
        min_stress = st.slider("Min Stress Level", 0, 100, 0)
    
    # Apply filters
    filtered_df = df.copy()
    if show_high_risk_only:
        filtered_df = filtered_df[filtered_df['risk_score'] > 70]
    if show_deception_only:
        filtered_df = filtered_df[filtered_df['deception_indicators'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)]
    if min_stress > 0:
        filtered_df = filtered_df[filtered_df['stress_level'] >= min_stress]
    
    st.info(f"Showing {len(filtered_df)} of {len(df)} results")
    
    # Display each result as an expandable card
    for idx, row in filtered_df.iterrows():
        risk_color = "🔴" if row['risk_score'] > 70 else "🟡" if row['risk_score'] > 40 else "🟢"
        
        with st.expander(
            f"{risk_color} **{row['filename']}** | Risk: {row['risk_score']}/100 | Stress: {row['stress_level']:.1f}% | {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}",
            expanded=False
        ):
            # Create columns for organized display
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Transcription
                st.subheader("📝 Transcription")
                st.text_area(
                    "Text",
                    value=row.get('transcription', 'No transcription'),
                    height=100,
                    key=f"trans_{idx}",
                    label_visibility="collapsed"
                )
                
                # AI Interpretation
                st.subheader("🤖 AI Analysis")
                ai_text = row.get('ai_interpretation', 'No AI analysis available')
                st.markdown(f"**Interpretation:** {ai_text}")
                
                # Deception Indicators
                if row.get('deception_indicators') and len(row['deception_indicators']) > 0:
                    st.subheader("🚨 Deception Indicators")
                    for indicator in row['deception_indicators']:
                        st.warning(f"⚠️ {indicator}")
                
                # Pattern Detection
                patterns_detected = []
                if row.get('gaslighting', {}).get('gaslighting_detected'):
                    patterns_detected.append("🎭 Gaslighting")
                if row.get('toxicity', {}).get('toxicity_detected'):
                    patterns_detected.append("☠️ Toxic Language")
                if row.get('narcissism', {}).get('narcissistic_patterns_detected'):
                    patterns_detected.append("👑 Narcissistic Patterns")
                
                if patterns_detected:
                    st.subheader("🔍 Detected Patterns")
                    st.error(" | ".join(patterns_detected))
            
            with col2:
                # Forensic Metrics
                st.subheader("📊 Forensic Metrics")
                
                st.metric("Risk Score", f"{row['risk_score']}/100")
                st.metric("Stress Level", f"{row['stress_level']:.1f}%")
                st.metric("Pitch Volatility", f"{row.get('pitch_volatility', 0):.2f}")
                st.metric("Silence Ratio", f"{row.get('silence_ratio', 0)*100:.1f}%")
                st.metric("Duration", f"{row.get('duration', 0):.1f}s")
                st.metric("Spectral Centroid", f"{row.get('spectral_centroid', 0):.0f} Hz")
                
                # Speaker ID
                st.divider()
                st.subheader("🎤 Speaker")
                st.info(row.get('identified_speaker', 'Unknown'))
                
                # Sentiment
                sentiment = row.get('sentiment', 'neutral')
                sentiment_emoji = {"positive": "😊", "negative": "😠", "neutral": "😐"}.get(sentiment, "😐")
                st.metric("Sentiment", f"{sentiment_emoji} {sentiment.title()}")
            
            # Errors if any
            if row.get('errors') and len(row['errors']) > 0:
                st.divider()
                st.subheader("⚠️ Processing Errors")
                for error in row['errors']:
                    st.error(error)

else:
    # No results yet - show instructions
    st.info("👆 Upload voice notes or select existing files from the sidebar to begin pipeline processing")
    
    st.markdown("""
    ### 🔄 How the Pipeline Works
    
    For each voice note, the system automatically:
    
    1. **🎙️ Transcribes** the audio using Whisper AI
    2. **🔬 Analyzes** forensic characteristics (stress, pitch, silence)
    3. **🚨 Detects** deception indicators from voice patterns
    4. **🤖 Interprets** using AI (Gemini) for psychological insights
    5. **🎭 Identifies** gaslighting, toxicity, and narcissistic patterns
    6. **🎤 Identifies** the speaker (if trained)
    7. **📊 Calculates** overall risk score
    
    **All results appear in one chronological table** - no clicking between pages!
    """)
    
    # Show example of what the table will look like
    st.subheader("📋 Example Results Table")
    st.image("https://via.placeholder.com/1200x400/f0f2f6/333333?text=Chronological+Results+Table+with+All+Analytics", 
             caption="Each row shows: Filename | Timestamp | Transcription | AI Analysis | Forensics | Deception Alerts | Risk Score")
