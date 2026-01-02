"""
REAL Die Waarheid Dashboard - No fake outputs
Uses only the actual forensic pipeline with full audit trails.
"""

import streamlit as st
import pandas as pd
import json
import sys
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Import real modules
from session_utils import save_session_data, load_session_data, clear_session_data
from verification_harness import VerificationHarness
from ingestion_engine import IngestionEngine
from real_analysis_engine import RealAnalysisEngine
from speaker_training import render_speaker_training

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Die Waarheid - REAL Forensic Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-success {
        color: #2e7d32;
        font-weight: bold;
    }
    .status-failed {
        color: #c62828;
        font-weight: bold;
    }
    .status-processing {
        color: #f57c00;
        font-weight: bold;
    }
    .audit-badge {
        background-color: #e3f2fd;
        border: 1px solid #2196F3;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        color: #1976D2;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    if 'case_id' not in st.session_state:
        st.session_state.case_id = None
    if 'evidence_files' not in st.session_state:
        st.session_state.evidence_files = []
    if 'speaker_profiles' not in st.session_state:
        st.session_state.speaker_profiles = {}
    if 'current_run' not in st.session_state:
        st.session_state.current_run = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # Load saved data
    load_session_data()

def get_case_dir(case_id: str) -> Path:
    """Get case directory path."""
    safe_name = "".join(c for c in case_id if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_name = safe_name.replace(' ', '_')[:100]
    return Path("data") / "cases" / safe_name

def render_header():
    """Render the main header."""
    st.markdown('<p class="main-header">🔍 Die Waarheid</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">REAL Forensic Analysis Dashboard - Afrikaans Primary with Full Audit Trail</p>', unsafe_allow_html=True)
    
    # Show audit badge
    st.markdown('<span class="audit-badge">📋 FORENSIC GRADE WITH AUDIT TRAIL</span>', unsafe_allow_html=True)

def render_sidebar():
    """Render the sidebar."""
    with st.sidebar:
        st.title("📋 Case Management")
        
        case_name = st.text_input(
            "Case Name",
            value=st.session_state.case_id or "",
            placeholder="Enter case name..."
        )
        
        if case_name and case_name != st.session_state.case_id:
            st.session_state.case_id = case_name
            st.session_state.current_run = None
            st.session_state.analysis_results = None
            save_session_data()
        
        st.divider()
        
        # Case directory info
        if st.session_state.case_id:
            case_dir = get_case_dir(st.session_state.case_id)
            if case_dir.exists():
                st.success(f"📁 Case folder exists")
                st.write(f"Path: `{case_dir}`")
                
                # Show manifest info
                manifest_file = case_dir / "manifest.json"
                if manifest_file.exists():
                    with open(manifest_file, 'r') as f:
                        manifest = json.load(f)
                    st.write(f"📊 Files: {len(manifest.get('files', []))}")
            else:
                st.warning("📁 Case folder not created yet")
        
        st.divider()
        
        # Quick actions
        st.subheader("🚀 Quick Actions")
        
        if st.button("📁 Ingest Files", type="primary"):
            st.session_state.show_ingestion = True
            st.rerun()
        
        if st.button("🔍 Verify Pipeline"):
            st.session_state.show_verification = True
            st.rerun()
        
        if st.button("🧪 Run Analysis", disabled=not st.session_state.case_id):
            st.session_state.show_analysis = True
            st.rerun()
        
        st.divider()
        
        # Speaker profiles
        if st.session_state.speaker_profiles:
            st.subheader("🎙️ Speakers")
            for name in st.session_state.speaker_profiles.keys():
                st.write(f"👤 {name}")

def render_ingestion():
    """Render file ingestion interface."""
    st.header("📁 File Ingestion")
    
    if not st.session_state.case_id:
        st.error("Please enter a case name first")
        return
    
    case_dir = get_case_dir(st.session_state.case_id)
    
    # Ingestion options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📂 Folder Ingestion")
        folder_path = st.text_input("Folder path", placeholder="C:\\path\\to\\files")
        if st.button("📥 Ingest Folder", disabled=not folder_path):
            with st.spinner("Ingesting files..."):
                try:
                    engine = IngestionEngine(case_dir)
                    results = engine.ingest_folder(Path(folder_path), parallel=True)
                    st.success(f"✅ Ingested {len(results)} files")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    with col2:
        st.subheader("📦 ZIP Ingestion")
        zip_path = st.text_input("ZIP file path", placeholder="C:\\path\\to\\archive.zip")
        if st.button("📥 Ingest ZIP", disabled=not zip_path):
            with st.spinner("Extracting and ingesting..."):
                try:
                    engine = IngestionEngine(case_dir)
                    results = engine.ingest_zip(Path(zip_path), parallel=True)
                    st.success(f"✅ Ingested {len(results)} files")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    # Show current files
    if case_dir.exists():
        manifest_file = case_dir / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
            
            st.divider()
            st.subheader(f"📊 Current Files ({len(manifest.get('files', []))})")
            
            if manifest.get('files'):
                df = pd.DataFrame(manifest['files'])
                st.dataframe(df[['name', 'type', 'size', 'ingested_at']], use_container_width=True)

def render_verification():
    """Render pipeline verification interface."""
    st.header("🔍 Pipeline Verification")
    
    st.warning("""
    ⚠️ **Verification Required Before Analysis**
    
    Run verification on a small sample of files to ensure the pipeline is working correctly 
    before processing your entire case.
    """)
    
    # Select files for verification
    case_dir = get_case_dir(st.session_state.case_id)
    manifest_file = case_dir / "manifest.json"
    
    if not manifest_file.exists():
        st.error("No files found. Please ingest files first.")
        return
    
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)
    
    # File selection
    st.subheader("📋 Select Files for Verification")
    
    # Show up to 20 files for selection
    files = manifest['files'][:20]
    selected_files = st.multiselect(
        "Choose files (max 10)",
        files,
        format_func=lambda x: x['name'],
        max_selections=10
    )
    
    if selected_files:
        st.write(f"Selected {len(selected_files)} files")
        
        if st.button("🧪 Run Verification", type="primary"):
            with st.spinner("Running verification..."):
                try:
                    # Create verification run
                    run_id = f"verify_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    run_dir = case_dir / "runs" / run_id
                    
                    harness = VerificationHarness(run_dir)
                    file_paths = [f['full_path'] for f in selected_files]
                    
                    results = harness.run_verification(file_paths)
                    
                    # Show results
                    st.success("✅ Verification Complete")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Completed", results['completed'])
                    with col2:
                        st.metric("Failed", results['failed'])
                    with col3:
                        st.metric("Success Rate", f"{results['completed']/results['total_files']*100:.1f}%")
                    
                    # Show file results
                    for file_result in results['files']:
                        status_class = "status-success" if file_result['status'] == 'completed' else "status-failed"
                        st.markdown(f"<p class='{status_class}'>📄 {file_result['file']}: {file_result['status'].upper()}</p>", unsafe_allow_html=True)
                        
                        if file_result['status'] == 'completed':
                            with st.expander(f"View stages for {file_result['file']}", expanded=False):
                                for stage, data in file_result['stages'].items():
                                    if data.get('status') == 'success':
                                        st.success(f"✅ {stage}: SUCCESS")
                                        if 'text' in data:
                                            st.text_area("Text", data['text'], height=100)
                                        if 'confidence' in data:
                                            st.write(f"Confidence: {data['confidence']}")
                                    else:
                                        st.error(f"❌ {stage}: FAILED")
                    
                    # Download link
                    summary_file = run_dir / "summary.json"
                    if summary_file.exists():
                        with open(summary_file, 'r') as f:
                            summary_data = json.load(f)
                        st.download_button(
                            "📥 Download Verification Report",
                            data=json.dumps(summary_data, indent=2),
                            file_name=f"verification_{run_id}.json"
                        )
                
                except Exception as e:
                    st.error(f"❌ Verification failed: {e}")
                    logger.exception("Verification failed")

def render_analysis():
    """Render real analysis interface."""
    st.header("🧪 Real Analysis")
    
    if not st.session_state.case_id:
        st.error("Please enter a case name first")
        return
    
    case_dir = get_case_dir(st.session_state.case_id)
    manifest_file = case_dir / "manifest.json"
    
    if not manifest_file.exists():
        st.error("No files found. Please ingest files first.")
        return
    
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)
    
    total_files = len(manifest.get('files', []))
    
    st.info(f"""
    📊 **Ready to Analyze**
    
    - Total files: {total_files}
    - Primary language: Afrikaans
    - Background analysis: Enabled
    - Speaker profiles: {len(st.session_state.speaker_profiles)} loaded
    
    ⚠️ **Note**: This will run the REAL forensic pipeline. Processing time depends on file count and size.
    """)
    
    # Analysis options
    with st.expander("⚙️ Analysis Options", expanded=False):
        use_background = st.checkbox("Enable background sound analysis", value=True)
        enable_code_switch = st.checkbox("Enable code-switch detection", value=True)
        max_files = st.number_input("Max files to process (0 for all)", min_value=0, max_value=total_files, value=0)
    
    if st.button("🚀 Run REAL Analysis", type="primary"):
        with st.spinner("Initializing real analysis engine..."):
            try:
                # Create analysis run
                run_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                run_dir = case_dir / "runs" / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                
                # Get file list
                file_list = [f['full_path'] for f in manifest['files']]
                if max_files > 0:
                    file_list = file_list[:max_files]
                
                # Initialize real engine
                engine = RealAnalysisEngine(case_dir)
                
                # Create progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Run analysis with progress updates
                start_time = time.time()
                
                status_text.text("🔄 Starting real pipeline...")
                progress_bar.progress(0.1)
                
                # Process files
                results = engine.run_analysis(file_list, st.session_state.speaker_profiles)
                
                progress_bar.progress(1.0)
                status_text.text("✅ Analysis complete!")
                
                # Show results
                st.success("🎉 REAL Analysis Complete!")
                
                # Summary metrics
                summary = results.get('summary', {})
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Files Processed", summary.get('processed', 0))
                with col2:
                    st.metric("Speakers Detected", summary.get('total_speakers', 0))
                with col3:
                    st.metric("Languages", len(summary.get('languages_detected', [])))
                with col4:
                    st.metric("Duration", f"{summary.get('duration_seconds', 0):.1f}s")
                
                # Show detailed results
                st.divider()
                st.subheader("📋 Analysis Results")
                
                # Transcriptions
                if results.get('transcriptions'):
                    st.write("🎤 **Transcriptions**")
                    for i, trans in enumerate(results['transcriptions'][:5]):
                        with st.expander(f"📄 {trans['file']}", expanded=False):
                            st.write(f"Confidence: {trans['confidence']}%")
                            st.text_area("Transcript", trans['transcript'], height=100)
                
                # Speaker diarization
                if results.get('speaker_diarization'):
                    st.write("👥 **Speaker Diarization**")
                    for diar in results['speaker_diarization'][:3]:
                        with st.expander(f"📄 {diar['file']}", expanded=False):
                            st.write(f"Speakers: {diar['speaker_count']}")
                            for seg in diar['segments'][:5]:
                                st.write(f"- {seg['speaker']}: {seg['start']:.1f}s - {seg['end']:.1f}s")
                
                # Background sounds
                if results.get('background_sounds'):
                    st.write("🔊 **Background Sounds**")
                    for sound in results['background_sounds'][:5]:
                        if sound['sounds']:
                            st.write(f"📄 {sound['file']}: {', '.join([s['type'] for s in sound['sounds'][:3]])}")
                
                # Download results
                results_file = run_dir / f"analysis_{run_id}.json"
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        results_data = json.load(f)
                    st.download_button(
                        "📥 Download Full Analysis Report",
                        data=json.dumps(results_data, indent=2, default=str),
                        file_name=f"analysis_{run_id}.json"
                    )
                
                # Store in session
                st.session_state.analysis_results = results
                st.session_state.current_run = run_id
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                logger.exception("Analysis failed")

def main():
    """Main application."""
    init_session_state()
    render_header()
    render_sidebar()
    
    # Main content area
    if st.session_state.get('show_ingestion'):
        render_ingestion()
    elif st.session_state.get('show_verification'):
        render_verification()
    elif st.session_state.get('show_analysis'):
        render_analysis()
    else:
        # Default view
        st.header("🏠 Welcome to Die Waarheid")
        
        st.info("""
        🔍 **REAL Forensic Analysis System**
        
        This is the REAL Die Waarheid dashboard with:
        - Actual Whisper + Afrikaans processing
        - Full audit trails and provenance
        - Background sound analysis
        - Code-switch detection
        - Speaker diarization with trained profiles
        
        No fake outputs - everything is processed through the real forensic pipeline.
        """)
        
        # Quick start guide
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 📁 1. Ingest Files
            - Use folder or ZIP import
            - Handles 10k–100k files
            - Automatic hashing
            """)
        
        with col2:
            st.markdown("""
            ### 🔍 2. Verify Pipeline
            - Test on sample files
            - Check Afrikaans capture
            - Validate all stages
            """)
        
        with col3:
            st.markdown("""
            ### 🧪 3. Run Analysis
            - Real forensic pipeline
            - Full audit trail
            - Downloadable reports
            """)
        
        # Recent activity
        if st.session_state.case_id:
            case_dir = get_case_dir(st.session_state.case_id)
            runs_dir = case_dir / "runs"
            
            if runs_dir.exists():
                st.divider()
                st.subheader("📊 Recent Runs")
                
                runs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
                
                if runs:
                    for run_dir in runs[:5]:
                        run_name = run_dir.name
                        run_time = datetime.fromtimestamp(run_dir.stat().st_mtime)
                        st.write(f"📁 {run_name} - {run_time.strftime('%Y-%m-%d %H:%M')}")
                else:
                    st.write("No runs yet")

if __name__ == "__main__":
    main()
