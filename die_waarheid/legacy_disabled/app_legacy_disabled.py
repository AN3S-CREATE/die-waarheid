"""
🔬 DIE WAARHEID: Complete Forensic WhatsApp Analysis System

Production-ready forensic dashboard with multi-layer analysis capabilities
"""

import streamlit as st
import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import configuration
from config import (
    BASE_DIR, DATA_DIR, TEMP_DIR, CREDENTIALS_DIR, MOBITABLES_DIR, REPORTS_DIR, EXPORTS_DIR,
    GEMINI_API_KEY, HUGGINGFACE_TOKEN,
    STREAMLIT_CONFIG, LOG_LEVEL, LOG_FILE
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import forensic modules
try:
    from src.gdrive_handler import gdrive_handler
except Exception as e:
    logger.warning(f"Could not import gdrive_handler: {e}")
    gdrive_handler = None

try:
    from src.chat_parser import whatsapp_parser
except Exception as e:
    logger.error(f"Could not import chat_parser: {e}")
    whatsapp_parser = None

try:
    from src.ai_analyzer import ai_analyzer
except Exception as e:
    logger.warning(f"Could not import ai_analyzer: {e}")
    ai_analyzer = None

try:
    from src.mobitab_builder import mobitab_builder
except Exception as e:
    logger.warning(f"Could not import mobitab_builder: {e}")
    mobitab_builder = None

try:
    from src.visualizations import ForensicVisualizer
    forensic_visualizer = ForensicVisualizer()
except Exception as e:
    logger.warning(f"Could not import visualizations: {e}")
    forensic_visualizer = None

try:
    from src.forensics import analyze_audio_file
except Exception as e:
    logger.warning(f"Could not import forensics: {e}")
    analyze_audio_file = None

# Configure Streamlit page
st.set_page_config(**STREAMLIT_CONFIG)

# Custom CSS for forensic theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f1f1f;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .stProgress .progress-bar {
        background-color: #ff4b4b;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stress-high { color: #ff4b4b; font-weight: bold; }
    .stress-medium { color: #ffa500; font-weight: bold; }
    .stress-low { color: #00ff00; font-weight: bold; }
    .contradiction-severe { 
        background-color: #ffebee; 
        border-left: 5px solid #f44336; 
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .contradiction-moderate { 
        background-color: #fff8e1; 
        border-left: 5px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .contradiction-minor { 
        background-color: #e8f5e8; 
        border-left: 5px solid #4caf50;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .phase-indicator {
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'chat_data' not in st.session_state:
        st.session_state.chat_data = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'gdrive_authenticated' not in st.session_state:
        st.session_state.gdrive_authenticated = False
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'dashboard'
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = {
            'stage': 'idle',
            'progress': 0,
            'current_file': '',
            'errors': []
        }
    
    # Auto-cleanup old temp files on startup
    cleanup_old_temp_files()
    
    # Auto-load saved analysis if exists and no data loaded
    auto_load_saved_analysis()

def auto_load_saved_analysis():
    """Automatically load saved analysis on startup"""
    saved_data_path = DATA_DIR / "saved_analysis.parquet"
    saved_results_path = DATA_DIR / "saved_analysis_results.json"
    
    if saved_data_path.exists() and st.session_state.chat_data is None:
        try:
            # Load the analyzed data
            st.session_state.chat_data = pd.read_parquet(saved_data_path)
            
            # Load AI results if exists
            if saved_results_path.exists():
                import json
                with open(saved_results_path, 'r') as f:
                    st.session_state.analysis_results = json.load(f)
            
            logger.info(f"Auto-loaded analysis with {len(st.session_state.chat_data)} messages")
        except Exception as e:
            logger.warning(f"Could not auto-load saved analysis: {e}")

def cleanup_old_temp_files():
    """Clean up temporary files older than 24 hours"""
    import time
    current_time = time.time()
    one_day_ago = current_time - (24 * 60 * 60)
    
    if os.path.exists(TEMP_DIR):
        for root, dirs, files in os.walk(TEMP_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    file_age = os.path.getmtime(file_path)
                    if file_age < one_day_ago:
                        os.remove(file_path)
                        logger.info(f"Cleaned up old temp file: {file_path}")
                except OSError as e:
                    logger.warning(f"Could not remove {file_path}: {e}")
        
        # Remove empty directories
        for root, dirs, files in os.walk(TEMP_DIR, topdown=False):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                try:
                    if not os.listdir(dir_path):
                        os.rmdir(dir_path)
                except OSError:
                    pass

# Navigation
def render_navigation():
    """Render multi-page navigation"""
    st.sidebar.markdown("# 🧭 Navigation")
    
    pages = {
        'dashboard': '🏠 Dashboard',
        'ingestion': '📥 Data Ingestion',
        'analysis': '🔬 Forensic Analysis',
        'visualizations': '📊 Visualizations',
        'reports': '📄 Reports',
        'settings': '⚙️ Settings'
    }
    
    # Page selector
    selected = st.sidebar.selectbox(
        "Select Page",
        list(pages.keys()),
        format_func=lambda x: pages[x],
        index=list(pages.keys()).index(st.session_state.current_page)
    )
    
    st.session_state.current_page = selected
    
    # Phase indicators
    st.sidebar.markdown("### 📈 Analysis Progress")
    phases = [
        ('Data Loaded', st.session_state.chat_data is not None and not st.session_state.chat_data.empty),
        ('Audio Processed', st.session_state.chat_data is not None and 'analysis_successful' in st.session_state.chat_data.columns),
        ('AI Analysis', bool(st.session_state.analysis_results)),
        ('Report Generated', False)  # TODO: Implement report tracking
    ]
    
    for phase_name, completed in phases:
        icon = "✅" if completed else "⏳"
        st.sidebar.markdown(f"{icon} {phase_name}")

# Dashboard Page
def render_dashboard():
    """Render main dashboard page"""
    st.markdown('<div class="main-header">🔬 DIE WAARHEID FORENSIC DASHBOARD</div>', unsafe_allow_html=True)
    
    if st.session_state.chat_data is None:
        # Welcome screen
        st.markdown("""
        ### 📋 Welcome to Die Waarheid
        
        **Complete Forensic WhatsApp Analysis System**
        
        Please load evidence using the **Data Ingestion** page to begin forensic analysis.
        
        #### 🔬 Capabilities:
        - **Multi-layer forensic analysis** with bio-signal detection
        - **Stress pattern detection** using advanced audio processing
        - **Contradiction identification** with AI-powered analysis
        - **Speaker diarization** and voice biometric analysis
        - **Psychological profiling** with Gemini AI
        - **Export-ready reports** in multiple formats
        
        ---
        
        #### 🚀 Quick Start:
        1. Navigate to **Data Ingestion** (sidebar)
        2. Load WhatsApp evidence (ZIP, local path, or Google Drive)
        3. Run forensic analysis
        4. Explore visualizations and generate reports
        """)
        return
    
    df = st.session_state.chat_data
    
    # Executive Summary
    st.markdown("## 📊 Executive Summary")
    
    # Calculate metrics
    total_messages = len(df)
    audio_messages = len(df[df['message_type'] == 'audio'])
    participants = df['sender'].nunique()
    
    # Stress metrics
    avg_stress = 0
    high_stress_count = 0
    if 'stress_index' in df.columns:
        avg_stress = df['stress_index'].mean()
        high_stress_count = (df['stress_index'] > 70).sum()
    
    # Contradictions
    contradictions = forensic_visualizer._detect_contradictions(df)
    contradiction_count = len(contradictions)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Messages", f"{total_messages:,}")
    with col2:
        st.metric("Voice Notes", audio_messages)
    with col3:
        stress_color = "stress-high" if avg_stress > 60 else "stress-medium" if avg_stress > 30 else "stress-low"
        st.markdown(f"<div class='metric-card'><h3>Avg Stress</h3><h2 class='{stress_color}'>{avg_stress:.1f}</h2></div>", unsafe_allow_html=True)
    with col4:
        st.metric("Contradictions", contradiction_count, delta="🚨" if contradiction_count > 0 else "✅")
    
    # Recent Activity
    st.markdown("## 📈 Recent Activity")
    if not df.empty:
        recent = df.tail(10).sort_values('timestamp', ascending=False)
        
        for _, row in recent.iterrows():
            timestamp = row['timestamp'].strftime('%Y-%m-%d %H:%M')
            sender = row['sender']
            message = row['message'][:100] + "..." if len(str(row['message'])) > 100 else row['message']
            
            st.markdown(f"**{timestamp}** - {sender}: {message}")
    
    # Quick Actions
    st.markdown("## ⚡ Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎙️ Process Audio", type="primary"):
            st.session_state.current_page = 'analysis'
            st.rerun()
    
    with col2:
        if st.button("🧠 Run AI Analysis"):
            st.session_state.current_page = 'analysis'
            st.rerun()
    
    with col3:
        if st.button("📊 View Visualizations"):
            st.session_state.current_page = 'visualizations'
            st.rerun()

# Data Ingestion Page
def render_ingestion():
    """Render data ingestion page"""
    st.markdown("## 📥 Data Ingestion")
    
    # Source selection
    st.markdown("### Select Evidence Source")
    source_method = st.radio(
        "Source Type",
        ["Upload ZIP", "Local ZIP Path", "Case Folder", "Google Drive"],
        horizontal=True
    )
    
    if source_method == "Google Drive":
        render_gdrive_ingestion()
    else:
        render_local_ingestion(source_method)

def render_gdrive_ingestion():
    """Render Google Drive ingestion interface"""
    st.markdown("#### 🌐 Google Drive Integration")
    
    if not gdrive_handler:
        st.error("Google Drive handler not available. Please check your configuration.")
        return
    
    # Authentication status
    if not st.session_state.gdrive_authenticated:
        if st.button("🔐 Authenticate with Google Drive"):
            with st.spinner("Authenticating..."):
                try:
                    if gdrive_handler.authenticate():
                        st.session_state.gdrive_authenticated = True
                        st.success("Successfully authenticated with Google Drive!")
                        st.rerun()
                    else:
                        st.error("Authentication failed. Please check your credentials.")
                except Exception as e:
                    st.error(f"Authentication error: {e}")
    else:
        st.success("✅ Authenticated with Google Drive")
        
        # File listing
        st.markdown("##### Available Files")
        file_type = st.selectbox("Filter by type", ["All", "WhatsApp Exports", "Audio Files"])
        
        if st.button("🔄 Refresh File List"):
            with st.spinner("Loading files..."):
                try:
                    if file_type == "WhatsApp Exports":
                        files = gdrive_handler.list_files("name contains 'WhatsApp' and name contains '.zip'")
                    elif file_type == "Audio Files":
                        files = gdrive_handler.list_files("name contains '.opus' or name contains '.m4a'")
                    else:
                        files = gdrive_handler.list_files()
                    
                    if files:
                        st.dataframe(pd.DataFrame(files))
                    else:
                        st.info("No files found")
                except Exception as e:
                    st.error(f"Error listing files: {e}")
        
        # Download selected files
        st.markdown("##### Download Files")
        if st.button("⬇️ Download Selected WhatsApp Exports"):
            with st.spinner("Downloading files..."):
                try:
                    downloaded = gdrive_handler.download_whatsapp_exports()
                    if downloaded:
                        st.success(f"Downloaded {len(downloaded)} files")
                        # Process downloaded files
                        process_downloaded_files(downloaded)
                    else:
                        st.warning("No files downloaded")
                except Exception as e:
                    st.error(f"Download error: {e}")

def render_local_ingestion(source_method):
    """Render local file ingestion"""
    if not whatsapp_parser:
        st.error("WhatsApp parser not available. Please check your configuration.")
        return
    
    if source_method == "Upload ZIP":
        uploaded_file = st.file_uploader(
            "Upload WhatsApp Export ZIP",
            type=['zip'],
            help="Select a WhatsApp export file"
        )
        
        if uploaded_file and st.button("🔍 Process Upload"):
            with st.spinner("Processing upload..."):
                try:
                    # Save uploaded file temporarily
                    temp_path = TEMP_DIR / uploaded_file.name
                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process the file
                    df = whatsapp_parser.parse_export_file(temp_path)
                    st.session_state.chat_data = df
                    st.success(f"Processed {len(df)} messages")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing file: {e}")
                    logger.exception("Error processing uploaded file")
    
    elif source_method == "Local ZIP Path":
        zip_path = st.text_input("Enter full path to ZIP file")
        
        if zip_path and st.button("🔍 Process Local ZIP"):
            if not os.path.exists(zip_path):
                st.error("File does not exist")
            else:
                with st.spinner("Processing ZIP..."):
                    try:
                        df = whatsapp_parser.parse_export_file(Path(zip_path))
                        st.session_state.chat_data = df
                        st.success(f"Processed {len(df)} messages")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing file: {e}")
                        logger.exception("Error processing local ZIP")
    
    elif source_method == "Case Folder":
        folder_path = st.text_input("Enter path to case folder")
        
        if folder_path and st.button("🔍 Process Case Folder"):
            if not os.path.isdir(folder_path):
                st.error("Folder does not exist")
            else:
                with st.spinner("Scanning and processing..."):
                    try:
                        # Find all ZIP files in folder
                        zip_files = list(Path(folder_path).glob("*.zip"))
                        if not zip_files:
                            st.warning("No ZIP files found in folder")
                            return
                        
                        all_data = []
                        for zip_file in zip_files:
                            st.write(f"Processing {zip_file.name}...")
                            try:
                                df = whatsapp_parser.parse_export_file(zip_file)
                                df['source_zip'] = zip_file.name
                                all_data.append(df)
                            except Exception as e:
                                logger.error(f"Error processing {zip_file.name}: {e}")
                                st.error(f"Error processing {zip_file.name}: {e}")
                                continue
                        
                        if not all_data:
                            st.error("No files could be processed")
                            return
                        
                        # Combine all data
                        combined_df = pd.concat(all_data, ignore_index=True)
                        st.session_state.chat_data = combined_df
                        st.success(f"Processed {len(combined_df)} messages from {len(zip_files)} files")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error processing folder: {e}")
                        logger.exception("Error processing case folder")

def process_downloaded_files(file_paths):
    """Process downloaded files from Google Drive"""
    if not whatsapp_parser:
        st.error("WhatsApp parser not available")
        return
    
    all_data = []
    chat_files = []
    media_files = []
    
    # First pass: find chat files and collect all media
    for zip_path in file_paths:
        try:
            # Extract to temp directory to scan
            import zipfile
            import tempfile
            import shutil
            
            temp_dir = Path(tempfile.mkdtemp())
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Look for chat file
            chat_file = whatsapp_parser._find_chat_file(temp_dir)
            if chat_file:
                chat_files.append((zip_path, chat_file))
                logger.info(f"Found chat file in {zip_path.name}")
            
            # Collect media files from this ZIP
            for ext in ['*.opus', '*.m4a', '*.mp3', '*.wav', '*.jpg', '*.jpeg', '*.png', '*.mp4']:
                media_files.extend(temp_dir.rglob(ext))
            
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        except Exception as e:
            logger.error(f"Error scanning {zip_path}: {e}")
    
    # Process the chat file(s) - use the first one found
    if chat_files:
        zip_path, chat_file = chat_files[0]
        logger.info(f"Processing chat from {zip_path.name}")
        try:
            df = whatsapp_parser.parse_export_file(zip_path)
            df['source_zip'] = zip_path.name
            all_data.append(df)
        except Exception as e:
            logger.error(f"Error processing chat file {zip_path}: {e}")
    
    # If no chat file, create timeline from media
    if not all_data and media_files:
        logger.info("No chat file found, creating media-only timeline")
        # Create a basic timeline from media files
        messages = []
        for media_file in media_files:
            timestamp = whatsapp_parser.extract_date_from_media_filename(media_file.name)
            if timestamp:
                messages.append({
                    'timestamp': timestamp,
                    'sender': 'Unknown',
                    'message': f'<Media: {media_file.name}>',
                    'message_type': 'media',
                    'media_path': str(media_file)
                })
        
        if messages:
            df = pd.DataFrame(messages)
            df = df.sort_values('timestamp')
            all_data.append(df)
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values('timestamp')
        st.session_state.chat_data = combined_df
        logger.info(f"Combined dataset: {len(combined_df)} messages from {len(file_paths)} ZIP files")
        st.success(f"Loaded {len(combined_df)} messages")

# Forensic Analysis Page
def render_analysis():
    """Render forensic analysis page"""
    st.markdown("## 🔬 Forensic Analysis")
    
    if st.session_state.chat_data is None:
        st.warning("Please load data first from the Data Ingestion page.")
        return
    
    # Check if there are audio files
    if 'message_type' in st.session_state.chat_data.columns:
        audio_files = st.session_state.chat_data[st.session_state.chat_data['message_type'] == 'audio']
    else:
        audio_files = pd.DataFrame()
    
    if audio_files.empty:
        st.info("No audio files found in the loaded data. You can still run AI analysis on text messages.")
    else:
        st.write(f"Found {len(audio_files)} audio file(s) for analysis")
        
        # Memory optimization warning
        if len(audio_files) > 10:
            st.warning("⚠️ Processing many audio files can use significant memory. Consider processing in batches.")
        
        # Audio Analysis Section
        st.markdown("### 🎵 Audio Processing")
        
        # Auto-save checkbox
        auto_save = st.checkbox("💾 Auto-save analysis after completion", value=True, help="Automatically save results to avoid re-processing")
        
        if st.button("🎯 Analyze Audio Files"):
            if not analyze_audio_file:
                st.error("Audio analysis module not available")
            else:
                with st.spinner("Analyzing audio files... This may take a few minutes as models are loaded."):
                    import gc
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, (_, row) in enumerate(audio_files.iterrows()):
                        media_file = row.get('media_file', f'file_{idx}')
                        status_text.text(f"Processing {media_file} ({idx+1}/{len(audio_files)})")
                        
                        if row.get('media_path'):
                            try:
                                result = analyze_audio_file(row['media_path'])
                                # Store results in dataframe
                                st.session_state.chat_data.loc[
                                    st.session_state.chat_data.get('media_file') == row.get('media_file'), 
                                    ['transcript', 'language', 'speaker_count', 'pitch_volatility', 
                                     'silence_ratio', 'max_loudness', 'analysis_successful']
                                ] = [
                                    result.get('transcript'),
                                    result.get('language'),
                                    result.get('speaker_count'),
                                    result.get('pitch_volatility'),
                                    result.get('silence_ratio'),
                                    result.get('max_loudness'),
                                    result.get('analysis_successful', False)
                                ]
                            except Exception as e:
                                logger.error(f"Error analyzing {media_file}: {e}")
                                st.warning(f"Could not analyze {media_file}")
                        
                        # Clear memory after every 5 files
                        if (idx + 1) % 5 == 0:
                            gc.collect()
                            st.write(f"🧹 Cleared memory after processing {idx + 1} files")
                        
                        progress_bar.progress((idx + 1) / len(audio_files))
                    
                    # Final garbage collection
                    gc.collect()
                    status_text.text("Audio analysis complete!")
                    st.success(f"Analyzed {len(audio_files)} audio files")
                    
                    # Auto-save if enabled
                    if auto_save:
                        try:
                            saved_data_path = DATA_DIR / "saved_analysis.parquet"
                            st.session_state.chat_data.to_parquet(saved_data_path, index=False)
                            st.success(f"✅ Analysis auto-saved to {saved_data_path}")
                        except Exception as e:
                            st.warning(f"Could not auto-save: {e}")
                    
                    st.rerun()
    
    # AI Analysis section
    if GEMINI_API_KEY and ai_analyzer:
        st.markdown("### 🧠 AI-Powered Analysis")
        
        if st.button("🔮 Run AI Analysis"):
            with st.spinner("Running AI analysis..."):
                try:
                    df = st.session_state.chat_data
                    analysis_type = st.selectbox("Analysis Type", ["General", "Psychological", "Deception"])
                    results = ai_analyzer.analyze_conversation(df, analysis_type.lower().replace(' ', '_'))
                    st.session_state.analysis_results = results
                    st.success("AI analysis completed!")
                    
                    # Display results
                    if isinstance(results, dict):
                        for key, value in results.items():
                            if isinstance(value, dict) and 'analysis' in value:
                                st.markdown(f"### {key.replace('_', ' ').title()}")
                                st.markdown(value['analysis'])
                    
                except Exception as e:
                    st.error(f"AI analysis failed: {e}")
                    logger.exception("AI analysis error")
    elif not GEMINI_API_KEY:
        st.warning("Gemini API key not configured. Please add it to Settings.")
    elif not ai_analyzer:
        st.warning("AI analyzer module not available. Please check your configuration.")
    
    # Live processing status
    if forensic_visualizer and st.session_state.processing_status['stage'] != 'idle':
        st.markdown("### 📊 Processing Status")
        try:
            forensic_visualizer.render_live_processing_status()
        except Exception as e:
            logger.warning(f"Could not render processing status: {e}")

# Visualizations Page
def render_visualizations():
    """Render visualizations page"""
    st.markdown("## 📊 Forensic Visualizations")
    
    if st.session_state.chat_data is None:
        st.warning("Please load data first from the Data Ingestion page.")
        return
    
    if not forensic_visualizer:
        st.error("Visualization module not available. Please check your configuration.")
        return
    
    df = st.session_state.chat_data
    
    # Phase selector
    st.markdown("### Select Visualization Phase")
    phase = st.selectbox(
        "Phase",
        ["Phase 1: Core Analytics", "Phase 2: Forensic Deep Dive", "Phase 3: Advanced Analysis"],
        index=0
    )
    
    try:
        if phase == "Phase 1: Core Analytics":
            st.markdown('<div class="phase-indicator">📈 Phase 1: Core Analytics</div>', unsafe_allow_html=True)
            
            if hasattr(forensic_visualizer, 'render_message_volume_chart'):
                forensic_visualizer.render_message_volume_chart(df)
            if hasattr(forensic_visualizer, 'render_sentiment_timeline'):
                forensic_visualizer.render_sentiment_timeline(df)
            if hasattr(forensic_visualizer, 'render_basic_timeline'):
                forensic_visualizer.render_basic_timeline(df)
        
        elif phase == "Phase 2: Forensic Deep Dive":
            st.markdown('<div class="phase-indicator">🔬 Phase 2: Forensic Deep Dive</div>', unsafe_allow_html=True)
            
            if hasattr(forensic_visualizer, 'render_stress_heatmap'):
                forensic_visualizer.render_stress_heatmap(df)
            if hasattr(forensic_visualizer, 'render_contradiction_explorer'):
                forensic_visualizer.render_contradiction_explorer(df)
            if hasattr(forensic_visualizer, 'render_mobitable_timeline'):
                forensic_visualizer.render_mobitable_timeline(df)
            if hasattr(forensic_visualizer, 'render_audio_player'):
                forensic_visualizer.render_audio_player(df)
        
        else:  # Phase 3
            st.markdown('<div class="phase-indicator">⚡ Phase 3: Advanced Analysis</div>', unsafe_allow_html=True)
            
            if hasattr(forensic_visualizer, 'render_toxicity_dashboard'):
                forensic_visualizer.render_toxicity_dashboard(df)
            if hasattr(forensic_visualizer, 'render_voice_biometric'):
                forensic_visualizer.render_voice_biometric(df)
            if hasattr(forensic_visualizer, 'render_speaker_timeline'):
                forensic_visualizer.render_speaker_timeline(df)
            if hasattr(forensic_visualizer, 'render_live_processing_status'):
                forensic_visualizer.render_live_processing_status()
    
    except Exception as e:
        st.error(f"Error rendering visualizations: {e}")
        logger.exception("Visualization error")

# Reports Page
def render_reports():
    """Render reports page"""
    st.markdown("## 📄 Forensic Reports")
    
    if st.session_state.chat_data is None:
        st.warning("Please load data first from the Data Ingestion page.")
        return
    
    # Report generation options
    st.markdown("### Generate Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📋 Generate MobiTable"):
            if not mobitab_builder:
                st.error("MobiTable builder not available")
            else:
                try:
                    df = st.session_state.chat_data
                    mobitable = mobitab_builder.build_mobitable(df)
                    
                    # Export options
                    export_format = st.selectbox("Format", ["Markdown", "CSV", "Excel"])
                    
                    if export_format == "Markdown":
                        path = mobitab_builder.export_to_markdown(mobitable)
                        st.success(f"Exported to {path}")
                    elif export_format == "CSV":
                        path = mobitab_builder.export_to_csv(mobitable)
                        st.success(f"Exported to {path}")
                    else:
                        path = mobitab_builder.export_to_excel(mobitable)
                        st.success(f"Exported to {path}")
                except Exception as e:
                    st.error(f"Error generating MobiTable: {e}")
                    logger.exception("MobiTable generation error")
    
    with col2:
        if st.button("🧠 Generate AI Report"):
            if not ai_analyzer:
                st.error("AI analyzer not available")
            elif st.session_state.analysis_results:
                try:
                    report = ai_analyzer.generate_forensic_report(st.session_state.analysis_results)
                    st.markdown(report)
                    
                    # Download button
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name=f"forensic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                except Exception as e:
                    st.error(f"Error generating report: {e}")
                    logger.exception("Report generation error")
            else:
                st.warning("Please run AI analysis first")
    
    # Summary statistics
    if st.session_state.chat_data is not None and mobitab_builder:
        st.markdown("### 📊 Summary Statistics")
        try:
            df = st.session_state.chat_data
            summary = mobitab_builder.create_summary_table(df)
            st.dataframe(summary, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate summary: {e}")

# Settings Page
def render_settings():
    """Render settings page"""
    st.markdown("## ⚙️ Settings")
    
    # API Keys
    st.markdown("### 🔑 API Configuration")
    
    gemini_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=GEMINI_API_KEY or ""
    )
    
    hf_token = st.text_input(
        "HuggingFace Token",
        type="password",
        value=HUGGINGFACE_TOKEN or ""
    )
    
    if st.button("💾 Save API Keys"):
        os.environ["GEMINI_API_KEY"] = gemini_key
        os.environ["HUGGINGFACE_TOKEN"] = hf_token
        st.success("API keys saved!")
    
    # Model Settings
    st.markdown("### 🤖 Model Settings")
    
    whisper_model = st.selectbox(
        "Whisper Model",
        ["tiny", "base", "small", "medium", "large"],
        index=1
    )
    
    os.environ["WHISPER_MODEL"] = whisper_model
    
    # Data Management
    st.markdown("### 📁 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear All Data"):
            st.session_state.chat_data = None
            st.session_state.analysis_results = None
            st.session_state.processing_status = {'stage': 'idle', 'progress': 0, 'current_file': '', 'errors': []}
            st.success("All data cleared!")
            st.rerun()
    
    with col2:
        if st.button("🧹 Clear Memory Cache"):
            # Clear cached models
            import gc
            import src.forensics
            import src.brain
            
            # Reset global model variables
            src.forensics._whisper_model = None
            src.forensics._diarization_pipeline = None
            src.brain._gemini_model = None
            
            # Force garbage collection
            gc.collect()
            
            st.success("Memory cache cleared! Models will reload on next use.")
    
    # Persistence Options
    st.markdown("### 💾 Save & Load Analysis")
    
    # Check for saved data
    saved_data_path = DATA_DIR / "saved_analysis.parquet"
    saved_results_path = DATA_DIR / "saved_analysis_results.json"
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save Current Analysis"):
            if st.session_state.chat_data is not None:
                try:
                    # Save the analyzed data
                    st.session_state.chat_data.to_parquet(saved_data_path, index=False)
                    
                    # Save AI results if exists
                    if st.session_state.analysis_results:
                        import json
                        with open(saved_results_path, 'w') as f:
                            json.dump(st.session_state.analysis_results, f, default=str)
                    
                    st.success(f"Analysis saved to {saved_data_path}")
                except Exception as e:
                    st.error(f"Error saving analysis: {e}")
            else:
                st.warning("No data to save!")
    
    with col2:
        if st.button("📂 Load Saved Analysis"):
            if saved_data_path.exists():
                try:
                    # Load the analyzed data
                    st.session_state.chat_data = pd.read_parquet(saved_data_path)
                    
                    # Load AI results if exists
                    if saved_results_path.exists():
                        import json
                        with open(saved_results_path, 'r') as f:
                            st.session_state.analysis_results = json.load(f)
                    
                    st.success(f"Loaded analysis with {len(st.session_state.chat_data)} messages")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading analysis: {e}")
            else:
                st.info("No saved analysis found!")
    
    # Show saved data info
    if saved_data_path.exists():
        file_size = saved_data_path.stat().st_size / 1024 / 1024  # MB
        file_date = datetime.fromtimestamp(saved_data_path.stat().st_mtime)
        st.info(f"💾 Saved analysis: {len(pd.read_parquet(saved_data_path))} messages, {file_size:.1f} MB, saved on {file_date.strftime('%Y-%m-%d %H:%M')}")
    
    # Storage Management
    st.markdown("### 💾 Storage Management")
    
    # Calculate folder sizes
    import shutil
    
    def get_folder_size(folder_path):
        total_size = 0
        if os.path.exists(folder_path):
            for dirpath, dirnames, filenames in os.walk(folder_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    try:
                        total_size += os.path.getsize(fp)
                    except OSError:
                        pass
        return total_size / 1024 / 1024  # Convert to MB
    
    # Display storage usage
    col1, col2, col3 = st.columns(3)
    
    data_size = get_folder_size(DATA_DIR)
    temp_size = get_folder_size(TEMP_DIR)
    mobitables_size = get_folder_size(MOBITABLES_DIR)
    
    with col1:
        st.metric("Data Directory", f"{data_size:.1f} MB")
    
    with col2:
        st.metric("Temp Files", f"{temp_size:.1f} MB")
    
    with col3:
        st.metric("Reports", f"{mobitables_size:.1f} MB")
    
    # Storage cleanup options
    st.markdown("#### Cleanup Options")
    
    cleanup_col1, cleanup_col2, cleanup_col3 = st.columns(3)
    
    with cleanup_col1:
        if st.button("🗑️ Clear Temp Files", help="Delete temporary extraction files"):
            if os.path.exists(TEMP_DIR):
                shutil.rmtree(TEMP_DIR)
                os.makedirs(TEMP_DIR, exist_ok=True)
                st.success("Temporary files cleared!")
            else:
                st.info("No temporary files to clear.")
    
    with cleanup_col2:
        if st.button("🗑️ Clear Reports", help="Delete generated reports"):
            if os.path.exists(MOBITABLES_DIR):
                shutil.rmtree(MOBITABLES_DIR)
                os.makedirs(MOBITABLES_DIR, exist_ok=True)
                st.success("Reports cleared!")
            else:
                st.info("No reports to clear.")
    
    with cleanup_col3:
        if st.button("🗑️ Clear All Storage", help="Delete all temporary files and reports"):
            for folder in [TEMP_DIR, MOBITABLES_DIR]:
                if os.path.exists(folder):
                    shutil.rmtree(folder)
                    os.makedirs(folder, exist_ok=True)
            st.success("All storage cleared!")
    
    # Memory Usage Display
    st.markdown("### 💾 Memory Usage")
    
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("RSS Memory", f"{memory_info.rss / 1024 / 1024:.1f} MB")
    
    with col2:
        st.metric("VMS Memory", f"{memory_info.vms / 1024 / 1024:.1f} MB")
    
    with col3:
        cpu_percent = process.cpu_percent()
        st.metric("CPU Usage", f"{cpu_percent:.1f}%")
    
    # System Info
    st.markdown("### ℹ️ System Information")
    
    # Get disk usage
    disk_usage = shutil.disk_usage(DATA_DIR)
    
    st.json({
        "Version": "1.0.0",
        "Data Directory": str(DATA_DIR),
        "Credentials Directory": str(CREDENTIALS_DIR),
        "Log File": str(LOG_FILE),
        "Google Drive Authenticated": st.session_state.gdrive_authenticated,
        "Python Version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "Disk Free": f"{disk_usage.free / 1024 / 1024 / 1024:.1f} GB",
        "Disk Used": f"{disk_usage.used / 1024 / 1024 / 1024:.1f} GB",
        "Disk Total": f"{disk_usage.total / 1024 / 1024 / 1024:.1f} GB"
    })

# Main application
def main():
    """Main application entry point"""
    # Initialize session state
    init_session_state()
    
    # Render navigation
    render_navigation()
    
    # Render current page
    page_handlers = {
        'dashboard': render_dashboard,
        'ingestion': render_ingestion,
        'analysis': render_analysis,
        'visualizations': render_visualizations,
        'reports': render_reports,
        'settings': render_settings
    }
    
    handler = page_handlers.get(st.session_state.current_page, render_dashboard)
    handler()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<center><small>🔬 DIE WAARHEID - Forensic WhatsApp Analysis System v1.0.0</small></center>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
