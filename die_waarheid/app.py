"""
Die Waarheid - Main Streamlit Application
Forensic-Grade WhatsApp Communication Analysis Platform
"""

import streamlit as st
import logging
from pathlib import Path
from datetime import datetime, timedelta
import sys

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    APP_NAME,
    APP_VERSION,
    APP_DESCRIPTION,
    validate_config,
    get_config_summary,
    AUDIO_DIR,
    TEXT_DIR,
    TEMP_DIR,
    REPORTS_DIR
)

from src.gdrive_handler import GDriveHandler
from src.chat_parser import WhatsAppParser
from src.whisper_transcriber import WhisperTranscriber
from src.forensics import ForensicsEngine
from src.mobitab_builder import MobitabBuilder
from src.ai_analyzer import AIAnalyzer
from src.profiler import PsychologicalProfiler
from src.visualizations import VisualizationEngine
from src.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


def setup_page():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title=f"{APP_NAME} v{APP_VERSION}",
        page_icon="🕵️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-left: 4px solid #ff6b6b;
        border-radius: 0.25rem;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-left: 4px solid #28a745;
        border-radius: 0.25rem;
    }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render application header"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title(f"🕵️ {APP_NAME}")
        st.markdown(f"*{APP_DESCRIPTION}*")
    
    with col2:
        st.metric("Version", APP_VERSION)
    
    st.divider()


def render_sidebar():
    """Render sidebar navigation and settings"""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        errors, warnings = validate_config()
        
        if errors:
            st.error("Configuration Errors:")
            for error in errors:
                st.error(error)
        
        if warnings:
            st.warning("Configuration Warnings:")
            for warning in warnings:
                st.warning(warning)
        
        if not errors:
            st.success("✅ Configuration Valid")
        
        st.divider()
        
        st.header("📊 Navigation")
        page = st.radio(
            "Select Analysis Module:",
            [
                "🏠 Home",
                "📥 Data Import",
                "🎙️ Speaker Training",
                "🎙️ Audio Analysis",
                "💬 Chat Analysis",
                "🧠 AI Analysis",
                "📈 Visualizations",
                "📄 Report Generation",
                "⚙️ Settings"
            ]
        )
        
        return page


def page_home():
    """Home page"""
    st.header("Welcome to Die Waarheid")
    
    # Count REAL data
    chat_files_count = len(list(TEXT_DIR.glob("*.txt")))
    audio_files_count = sum(len(list(AUDIO_DIR.rglob(f"*.{ext}"))) for ext in ['mp3', 'wav', 'opus', 'ogg', 'm4a', 'aac'])
    
    # Determine analysis status
    if audio_files_count > 0 or chat_files_count > 0:
        status = "Ready"
        status_detail = f"{audio_files_count + chat_files_count} files loaded"
    else:
        status = "Waiting"
        status_detail = "No files loaded"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Messages", chat_files_count, f"From {len(list(TEXT_DIR.glob('*.txt')))} files")
    
    with col2:
        st.metric("Audio Files", audio_files_count, f"Across {len([d for d in AUDIO_DIR.iterdir() if d.is_dir()])} directories")
    
    with col3:
        st.metric("Analysis Status", status, status_detail)
    
    st.divider()
    
    # Show real data summary
    if audio_files_count > 0 or chat_files_count > 0:
        st.subheader("📊 Your Data Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📁 Audio Files by Type:**")
            audio_types = {}
            for ext in ['mp3', 'wav', 'opus', 'ogg', 'm4a', 'aac']:
                count = len(list(AUDIO_DIR.rglob(f"*.{ext}")))
                if count > 0:
                    audio_types[ext.upper()] = count
            
            for ext, count in sorted(audio_types.items()):
                st.write(f"  {ext}: {count:,} files")
        
        with col2:
            st.write("**📝 Chat Files:**")
            if chat_files_count > 0:
                chat_files = list(TEXT_DIR.glob("*.txt"))
                for chat_file in chat_files[:5]:  # Show first 5
                    st.write(f"  📄 {chat_file.name}")
                if len(chat_files) > 5:
                    st.write(f"  ... and {len(chat_files) - 5} more files")
            else:
                st.write("  No chat files loaded")
        
        # Quick actions
        st.subheader("🚀 Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Upload More Files"):
                st.session_state.navigate_to = "📥 Data Import"
        
        with col2:
            if st.button("🎙️ Train Speakers"):
                st.session_state.navigate_to = "🎙️ Speaker Training"
        
        with col3:
            if st.button("📈 View Analytics"):
                st.session_state.navigate_to = "📈 Visualizations"
        
        # Handle navigation
        if hasattr(st.session_state, 'navigate_to'):
            st.rerun()
    
    st.subheader("Quick Start")
    
    with st.expander("📖 How to Use", expanded=True):
        st.markdown("""
        1. **Import Data**: Upload WhatsApp chats and audio files
        2. **Process Audio**: Transcribe voice notes and extract bio-signals
        3. **Analyze Chat**: Parse messages and detect patterns
        4. **AI Analysis**: Run Gemini-powered psychological profiling
        5. **Visualize**: View interactive charts and timelines
        6. **Generate Report**: Create comprehensive forensic reports
        
        **Key Features:**
        - 🎙️ Audio forensics with stress detection
        - 💬 WhatsApp chat parsing and analysis
        - 🧠 AI-powered psychological profiling
        - 📊 Interactive visualizations
        - 📄 Multi-format report generation
        """)
    
    with st.expander("⚠️ Important Notes"):
        st.markdown("""
        - All data is processed locally on your machine
        - API keys are stored securely in .env file
        - Temporary files are automatically cleaned up
        - Reports are saved in the reports directory
        """)


def page_data_import():
    """Data import page"""
    st.header("📥 Data Import")
    
    tab1, tab2, tab3 = st.tabs(["Google Drive", "Manual Upload", "Status"])
    
    with tab1:
        st.subheader("Google Drive Integration")
        
        if st.button("🔐 Authenticate with Google Drive"):
            with st.spinner("Authenticating..."):
                handler = GDriveHandler()
                success, message = handler.authenticate()
                
                if success:
                    st.success(message)
                    st.session_state.gdrive_handler = handler
                else:
                    st.error(message)
        
        if 'gdrive_handler' in st.session_state and st.session_state.gdrive_handler.authenticated:
            st.success("✅ Authenticated with Google Drive")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Download Text Files"):
                    with st.spinner("Downloading text files..."):
                        files, message = st.session_state.gdrive_handler.download_text_files()
                        st.info(message)
                        if files:
                            st.success(f"Downloaded {len(files)} text files")
            
            with col2:
                if st.button("🎙️ Download Audio Files"):
                    with st.spinner("Downloading audio files..."):
                        files, message = st.session_state.gdrive_handler.download_audio_files()
                        st.info(message)
                        if files:
                            st.success(f"Downloaded {len(files)} audio files")
    
    with tab2:
        st.subheader("Manual File Upload")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Upload Chat Export**")
            
            # Debug information
            with st.expander("🔍 Upload Debug Info"):
                st.write("**Current Upload Configuration:**")
                st.write("- Multiple files: ENABLED")
                st.write("- File types: .txt")
                st.write("- Max size per file: 50MB")
                st.write("- Browser support: Chrome, Firefox, Edge, Safari")
                st.write("**Tips for multiple file selection:**")
                st.write("1. Click 'Browse files'")
                st.write("2. Hold Ctrl/Cmd and click multiple files")
                st.write("3. Or drag & drop multiple files onto the upload area")
            
            # Primary upload method
            chat_files = st.file_uploader(
                "Select WhatsApp chat export files (.txt)", 
                type=['txt'], 
                accept_multiple_files=True,
                key="chat_files_primary"
            )
            
            # Alternative upload method for troubleshooting
            if st.button("🔄 Try Alternative Upload Method"):
                st.session_state.show_alternative_upload = True
            
            if st.session_state.get('show_alternative_upload', False):
                st.write("**Alternative Upload (Individual Files):**")
                uploaded_file = st.file_uploader(
                    "Upload one file at a time", 
                    type=['txt'],
                    key="chat_file_single"
                )
                
                if uploaded_file:
                    if 'chat_files_queue' not in st.session_state:
                        st.session_state.chat_files_queue = []
                    
                    # Add to queue
                    if uploaded_file.name not in [f.name for f in st.session_state.chat_files_queue]:
                        st.session_state.chat_files_queue.append(uploaded_file)
                        st.success(f"✅ Added {uploaded_file.name} to queue")
                    
                    # Show queue
                    st.write(f"**Queue ({len(st.session_state.chat_files_queue)} files):**")
                    for i, f in enumerate(st.session_state.chat_files_queue):
                        st.write(f"  {i+1}. {f.name}")
                    
                    # Process queue button
                    if st.button(f"📤 Process {len(st.session_state.chat_files_queue)} Files"):
                        chat_files = st.session_state.chat_files_queue
                        st.session_state.chat_files_queue = []
                        st.session_state.show_alternative_upload = False
            
            if chat_files:
                st.info(f"📁 Processing {len(chat_files)} file(s): {', '.join([f.name for f in chat_files[:3]])}{'...' if len(chat_files) > 3 else ''}")
                
                with st.spinner(f"Processing {len(chat_files)} chat file(s)..."):
                    saved_files = []
                    failed_files = []
                    
                    for i, chat_file in enumerate(chat_files):
                        try:
                            st.write(f"📄 Processing file {i+1}/{len(chat_files)}: {chat_file.name}")
                            
                            # Validate file size (max 50MB per file)
                            if chat_file.size > 50 * 1024 * 1024:  # 50MB limit
                                failed_files.append((chat_file.name, "File too large (max 50MB)"))
                                st.error(f"❌ {chat_file.name}: File too large")
                                continue
                            
                            # Save chat file to permanent storage
                            chat_save_path = Path(TEXT_DIR) / chat_file.name
                            
                            # Ensure unique filename
                            counter = 1
                            original_path = chat_save_path
                            while chat_save_path.exists():
                                stem = original_path.stem
                                suffix = original_path.suffix
                                chat_save_path = original_path.parent / f"{stem}_{counter}{suffix}"
                                counter += 1
                            
                            with open(chat_save_path, "wb") as f:
                                f.write(chat_file.getbuffer())
                            
                            # Parse the chat file
                            parser = WhatsAppParser()
                            success, message = parser.parse_file(chat_save_path)
                            
                            if success:
                                saved_files.append((chat_save_path, parser))
                                st.success(f"✅ Processed: {chat_file.name}")
                            else:
                                # Clean up failed file
                                if chat_save_path.exists():
                                    chat_save_path.unlink()
                                failed_files.append((chat_file.name, message))
                                st.error(f"❌ {chat_file.name}: {message}")
                                
                        except Exception as e:
                            failed_files.append((chat_file.name, str(e)))
                            st.error(f"❌ {chat_file.name}: {str(e)}")
                    
                    # Display results
                    if saved_files:
                        st.success(f"🎉 Successfully processed {len(saved_files)} chat files")
                        
                        # Show metadata for first few files
                        for i, (file_path, parser) in enumerate(saved_files[:3]):
                            with st.expander(f"📄 {file_path.name}"):
                                metadata = parser.get_metadata()
                                st.json(metadata)
                        
                        if len(saved_files) > 3:
                            st.info(f"... and {len(saved_files) - 3} more files")
                        
                        # Store parsers in session state
                        st.session_state.chat_parsers = [parser for _, parser in saved_files]
                    
                    if failed_files:
                        st.error(f"❌ Failed to process {len(failed_files)} files:")
                        for name, error in failed_files[:3]:
                            st.error(f"  {name}: {error}")
                        if len(failed_files) > 3:
                            st.error(f"... and {len(failed_files) - 3} more errors")
        
        with col2:
            st.write("**Upload Audio Files**")
            audio_files = st.file_uploader(
                "Select audio files",
                type=['mp3', 'wav', 'opus', 'ogg', 'm4a', 'aac'],
                accept_multiple_files=True
            )
            
            if audio_files:
                # Validate total upload size
                total_size = sum(f.size for f in audio_files)
                max_total_size = 500 * 1024 * 1024  # 500MB total limit per upload
                
                if total_size > max_total_size:
                    st.error(f"❌ Total upload size too large: {total_size / (1024*1024):.1f}MB (max 500MB)")
                    st.error("Please upload files in smaller batches")
                else:
                    with st.spinner(f"Saving {len(audio_files)} audio files..."):
                        saved_files = []
                        failed_files = []
                        
                        for audio_file in audio_files:
                            try:
                                # Validate individual file size (max 100MB per file)
                                if audio_file.size > 100 * 1024 * 1024:  # 100MB limit
                                    failed_files.append((audio_file.name, "File too large (max 100MB)"))
                                    continue
                                
                                # Validate file type
                                if audio_file.type and not any(audio_file.type.startswith(t) for t in ['audio/', 'video/']):
                                    failed_files.append((audio_file.name, "Invalid file type"))
                                    continue
                                
                                # Create organized date-based storage
                                if audio_file.name.startswith("PTT-") and "-" in audio_file.name:
                                    # Extract date from PTT files (PTT-YYYYMMDD-WAXXXX)
                                    date_part = audio_file.name.split("-")[1]
                                    if len(date_part) >= 8 and date_part[:8].isdigit():
                                        year = date_part[:4]
                                        month = date_part[4:6]
                                        date_folder = Path(AUDIO_DIR) / "organized" / f"{year}-{month}"
                                        date_folder.mkdir(parents=True, exist_ok=True)
                                        save_path = date_folder / audio_file.name
                                    else:
                                        save_path = Path(AUDIO_DIR) / audio_file.name
                                else:
                                    save_path = Path(AUDIO_DIR) / audio_file.name
                                
                                # Ensure unique filename
                                counter = 1
                                original_path = save_path
                                while save_path.exists():
                                    stem = original_path.stem
                                    suffix = original_path.suffix
                                    save_path = original_path.parent / f"{stem}_{counter}{suffix}"
                                    counter += 1
                                
                                # Save file to permanent storage
                                with open(save_path, "wb") as f:
                                    f.write(audio_file.getbuffer())
                                
                                saved_files.append(save_path)
                                
                            except Exception as e:
                                failed_files.append((audio_file.name, str(e)))
                        
                        # Display results
                        if saved_files:
                            st.success(f"✅ Successfully saved {len(saved_files)} audio files ({sum(Path(f).stat().st_size for f in saved_files) / (1024*1024):.1f}MB)")
                            
                            # Show saved locations
                            for file_path in saved_files[:3]:  # Show first 3
                                rel_path = file_path.relative_to(AUDIO_DIR)
                                st.info(f"📁 {rel_path}")
                            if len(saved_files) > 3:
                                st.info(f"... and {len(saved_files) - 3} more files")
                        
                        if failed_files:
                            st.error(f"❌ Failed to save {len(failed_files)} files:")
                            for name, error in failed_files[:3]:  # Show first 3 errors
                                st.error(f"  {name}: {error}")
                            if len(failed_files) > 3:
                                st.error(f"... and {len(failed_files) - 3} more errors")
    
    with tab3:
        st.subheader("Import Status & File Management")
        
        # Count actual files
        chat_files_count = len(list(TEXT_DIR.glob("*.txt")))
        
        # Debug: Show what directories exist
        audio_subdirs = [d for d in AUDIO_DIR.iterdir() if d.is_dir()]
        st.write(f"🔍 Debug: Found {len(audio_subdirs)} audio subdirectories: {[d.name for d in audio_subdirs[:5]]}")
        
        # Count audio files by type for debugging
        audio_counts = {}
        for ext in ['mp3', 'wav', 'opus', 'ogg', 'm4a', 'aac']:
            audio_counts[ext] = len(list(AUDIO_DIR.rglob(f"*.{ext}")))
        
        st.write(f"🔍 Debug: Audio files by type: {audio_counts}")
        
        audio_files_count = sum(audio_counts.values())
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Chat Files", chat_files_count)
        
        with col2:
            st.metric("Audio Files", audio_files_count)
        
        with col3:
            st.metric("Total Storage", f"{audio_files_count + chat_files_count}")
        
        # File management section
        st.subheader("📁 File Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Storage Locations**")
            st.info(f"📝 Chat files: `{TEXT_DIR}`")
            st.info(f"🎵 Audio files: `{AUDIO_DIR}`")
            st.info(f"📅 Organized: `{AUDIO_DIR}/organized`")
            
            if st.button("📊 View Storage Statistics"):
                with st.spinner("Analyzing storage..."):
                    # Get file statistics by type
                    audio_stats = {}
                    for ext in ['mp3', 'wav', 'opus', 'ogg', 'm4a', 'aac']:
                        count = len(list(AUDIO_DIR.rglob(f"*.{ext}")))
                        if count > 0:
                            audio_stats[ext.upper()] = count
                    
                    if audio_stats:
                        st.write("**Audio Files by Type:**")
                        for ext, count in audio_stats.items():
                            st.write(f"  {ext}: {count:,} files")
                    
                    # Get date organization stats
                    organized_dir = AUDIO_DIR / "organized"
                    if organized_dir.exists():
                        date_dirs = [d for d in organized_dir.iterdir() if d.is_dir() and d.name.match(r"\d{4}-\d{2}")]
                        if date_dirs:
                            st.write("**Files by Date:**")
                            for date_dir in sorted(date_dirs):
                                count = len(list(date_dir.rglob("*.*")))
                                st.write(f"  {date_dir.name}: {count:,} files")
        
        with col2:
            st.write("**Quick Actions**")
            
            if st.button("🔄 Refresh File Count"):
                st.rerun()
            
            if st.button("📋 Export File List"):
                with st.spinner("Generating file list..."):
                    all_files = []
                    
                    # Chat files
                    for chat_file in TEXT_DIR.glob("*.txt"):
                        all_files.append(f"CHAT: {chat_file.name}")
                    
                    # Audio files
                    for audio_file in AUDIO_DIR.rglob("*.*"):
                        if audio_file.suffix.lower() in ['.mp3', '.wav', '.opus', '.ogg', '.m4a', '.aac']:
                            rel_path = audio_file.relative_to(AUDIO_DIR)
                            all_files.append(f"AUDIO: {rel_path}")
                    
                    if all_files:
                        file_list = "\n".join(sorted(all_files))
                        st.download_button(
                            label="📥 Download File List",
                            data=file_list,
                            file_name="die_waarheid_file_list.txt",
                            mime="text/plain"
                        )
                    else:
                        st.warning("No files found")
            
            if st.button("🧹 Cleanup Temporary Files"):
                with st.spinner("Cleaning temporary files..."):
                    cleaned = 0
                    for temp_file in TEMP_DIR.glob("*.*"):
                        if temp_file.is_file():
                            temp_file.unlink()
                            cleaned += 1
                    st.success(f"✅ Cleaned {cleaned} temporary files")


def page_audio_analysis():
    """Audio analysis page"""
    st.header("🎙️ Audio Analysis")
    
    st.subheader("Forensic Audio Processing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Analysis Settings**")
        
        whisper_model = st.selectbox(
            "Whisper Model Size",
            ["tiny", "base", "small", "medium", "large"],
            index=3
        )
        
        language = st.selectbox(
            "Language",
            ["af", "en", "es", "fr", "de"],
            index=0
        )
    
    with col2:
        st.write("**Processing Options**")
        
        extract_pitch = st.checkbox("Extract Pitch Contour", value=True)
        calculate_stress = st.checkbox("Calculate Stress Level", value=True)
        transcribe = st.checkbox("Transcribe Audio", value=True)
    
    if st.button("▶️ Start Audio Analysis"):
        st.info("Audio analysis would process files from data/audio/ directory")
        st.write("Features to analyze:")
        st.write("- Pitch volatility")
        st.write("- Silence ratio")
        st.write("- Intensity metrics")
        st.write("- MFCC variance")
        st.write("- Zero crossing rate")
        st.write("- Spectral centroid")


def page_chat_analysis():
    """Chat analysis page"""
    st.header("💬 Chat Analysis")
    
    st.subheader("WhatsApp Message Analysis")
    
    if st.button("📊 Analyze Chat Messages"):
        st.info("Chat analysis would process files from data/text/ directory")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Message Statistics**")
            st.metric("Total Messages", "0")
            st.metric("Unique Senders", "0")
            st.metric("Date Range", "N/A")
        
        with col2:
            st.write("**Pattern Detection**")
            st.metric("Toxicity Detected", "0")
            st.metric("Gaslighting Patterns", "0")
            st.metric("Contradictions", "0")


def page_ai_analysis():
    """AI analysis page"""
    st.header("🧠 AI Analysis")
    
    st.subheader("Gemini-Powered Psychological Profiling")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Analysis Options**")
        
        analyze_emotions = st.checkbox("Analyze Emotions", value=True)
        detect_patterns = st.checkbox("Detect Behavioral Patterns", value=True)
        profile_participants = st.checkbox("Create Participant Profiles", value=True)
        detect_contradictions = st.checkbox("Detect Contradictions", value=True)
    
    with col2:
        st.write("**AI Settings**")
        
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2)
        max_tokens = st.slider("Max Tokens", 1000, 8192, 4096)
    
    if st.button("🚀 Run AI Analysis"):
        st.info("AI analysis would use Gemini to analyze conversation patterns")
        st.write("Analysis includes:")
        st.write("- Psychological profiling")
        st.write("- Contradiction detection")
        st.write("- Toxicity assessment")
        st.write("- Trust score calculation")


def page_visualizations():
    """Visualizations page"""
    st.header("📈 Visualizations")
    
    st.subheader("Interactive Analysis Charts")
    
    chart_type = st.selectbox(
        "Select Chart Type",
        [
            "Stress Timeline",
            "Speaker Distribution",
            "Message Frequency",
            "Bio-Signal Heatmap",
            "Forensic Flags",
            "Emotion Distribution",
            "Conversation Flow",
            "Dashboard"
        ]
    )
    
    if st.button("📊 Generate Chart"):
        st.info(f"Would generate {chart_type} visualization")
        st.write("Chart features:")
        st.write("- Interactive Plotly charts")
        st.write("- Hover information")
        st.write("- Export to image")
        st.write("- Zoom and pan controls")


def page_report_generation():
    """Report generation page"""
    st.header("📄 Report Generation")
    
    st.subheader("Forensic Analysis Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Case Information**")
        
        case_id = st.text_input("Case ID", "CASE_001")
        chat_name = st.text_input("Chat Name", "Investigation")
    
    with col2:
        st.write("**Report Options**")
        
        export_formats = st.multiselect(
            "Export Formats",
            ["Markdown", "HTML", "JSON", "PDF"],
            default=["Markdown", "HTML"]
        )
    
    if st.button("📝 Generate Report"):
        st.info(f"Would generate report for case {case_id}")
        st.write("Report sections:")
        st.write("- Executive Summary")
        st.write("- Psychological Profile")
        st.write("- Detected Contradictions")
        st.write("- Bio-Signal Analysis")
        st.write("- Recommendations")
        st.write("- Legal Disclaimer")


def page_speaker_training():
    """Speaker training page"""
    st.header("🎙️ Speaker Training")
    
    st.write("""
    Train the AI to distinguish between two speakers by providing voice samples.
    This ensures consistent speaker identification even if usernames change.
    """)
    
    # Initialize speaker identification system
    if 'speaker_system' not in st.session_state:
        try:
            from src.speaker_identification import SpeakerIdentificationSystem
            st.session_state.speaker_system = SpeakerIdentificationSystem("MAIN_CASE")
            st.success("✅ Speaker identification system initialized")
        except Exception as e:
            st.error(f"❌ Failed to initialize speaker system: {e}")
            return
    
    system = st.session_state.speaker_system
    
    # Check if investigation is initialized
    participants = system.get_all_participants()
    
    if len(participants) < 2:
        st.subheader("🔧 Initialize Investigation")
        st.write("First, set up the two participants for this investigation:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            participant_a_name = st.text_input("Participant A Name:", value="Speaker A", key="participant_a")
        
        with col2:
            participant_b_name = st.text_input("Participant B Name:", value="Speaker B", key="participant_b")
        
        if st.button("🚀 Initialize Investigation"):
            if participant_a_name and participant_b_name:
                try:
                    participant_a_id, participant_b_id = system.initialize_investigation(
                        participant_a_name, participant_b_name
                    )
                    st.success(f"✅ Investigation initialized!")
                    st.info(f"Participant A: {participant_a_id}")
                    st.info(f"Participant B: {participant_b_id}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to initialize: {e}")
            else:
                st.error("Please enter names for both participants")
    
    else:
        st.subheader("📊 Current Participants")
        
        for profile in participants:
            with st.expander(f"👤 {profile.primary_username} ({profile.assigned_role.value})"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Messages", profile.message_count)
                
                with col2:
                    st.metric("Voice Notes", profile.voice_note_count)
                
                with col3:
                    st.metric("Confidence", f"{profile.confidence_score:.2f}")
                
                st.write(f"**Voice Fingerprints:** {len(profile.voice_fingerprints)}")
                st.write(f"**Alternate Usernames:** {', '.join(profile.alternate_usernames) if profile.alternate_usernames else 'None'}")
        
        st.subheader("🎯 Voice Sample Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Train Speaker A**")
            
            # Get participant A
            participant_a = next((p for p in participants if p.assigned_role.value == "participant_a"), None)
            
            if participant_a:
                speaker_a_audio = st.file_uploader(
                    f"Upload voice sample for {participant_a.primary_username}",
                    type=['mp3', 'wav', 'opus', 'ogg', 'm4a', 'aac'],
                    key="speaker_a_audio"
                )
                
                if speaker_a_audio:
                    st.info(f"📁 Selected: {speaker_a_audio.name}")
                    
                    if st.button(f"🎙️ Train {participant_a.primary_username}", key="train_a"):
                        with st.spinner(f"Training {participant_a.primary_username}..."):
                            try:
                                # Save audio file temporarily
                                temp_path = Path(TEMP_DIR) / speaker_a_audio.name
                                with open(temp_path, "wb") as f:
                                    f.write(speaker_a_audio.getbuffer())
                                
                                # Register voice note
                                participant_id = system.register_voice_note(
                                    participant_a.primary_username,
                                    temp_path,
                                    datetime.now()
                                )
                                
                                if participant_id:
                                    st.success(f"✅ Voice sample registered for {participant_a.primary_username}")
                                    
                                    # Clean up temp file
                                    if temp_path.exists():
                                        temp_path.unlink()
                                    
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to register voice sample")
                                    
                            except Exception as e:
                                st.error(f"❌ Training failed: {e}")
        
        with col2:
            st.write("**Train Speaker B**")
            
            # Get participant B
            participant_b = next((p for p in participants if p.assigned_role.value == "participant_b"), None)
            
            if participant_b:
                speaker_b_audio = st.file_uploader(
                    f"Upload voice sample for {participant_b.primary_username}",
                    type=['mp3', 'wav', 'opus', 'ogg', 'm4a', 'aac'],
                    key="speaker_b_audio"
                )
                
                if speaker_b_audio:
                    st.info(f"📁 Selected: {speaker_b_audio.name}")
                    
                    if st.button(f"🎙️ Train {participant_b.primary_username}", key="train_b"):
                        with st.spinner(f"Training {participant_b.primary_username}..."):
                            try:
                                # Save audio file temporarily
                                temp_path = Path(TEMP_DIR) / speaker_b_audio.name
                                with open(temp_path, "wb") as f:
                                    f.write(speaker_b_audio.getbuffer())
                                
                                # Register voice note
                                participant_id = system.register_voice_note(
                                    participant_b.primary_username,
                                    temp_path,
                                    datetime.now()
                                )
                                
                                if participant_id:
                                    st.success(f"✅ Voice sample registered for {participant_b.primary_username}")
                                    
                                    # Clean up temp file
                                    if temp_path.exists():
                                        temp_path.unlink()
                                    
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to register voice sample")
                                    
                            except Exception as e:
                                st.error(f"❌ Training failed: {e}")
        
        st.subheader("🧪 Test Speaker Recognition")
        
        test_audio = st.file_uploader(
            "Upload a voice note to test speaker identification",
            type=['mp3', 'wav', 'opus', 'ogg', 'm4a', 'aac'],
            key="test_audio"
        )
        
        if test_audio:
            if st.button("🔍 Test Speaker Identification"):
                with st.spinner("Analyzing voice..."):
                    try:
                        # Save test audio temporarily
                        temp_path = Path(TEMP_DIR) / test_audio.name
                        with open(temp_path, "wb") as f:
                            f.write(test_audio.getbuffer())
                        
                        # Test identification
                        participant_id, confidence = system.identify_speaker(
                            "test_user", 
                            audio_file=temp_path
                        )
                        
                        if participant_id:
                            profile = system.get_participant(participant_id)
                            if profile:
                                st.success(f"🎯 Identified as: {profile.primary_username}")
                                st.info(f"📊 Confidence: {confidence:.2f}")
                                st.info(f"🔖 Role: {profile.assigned_role.value}")
                            else:
                                st.warning("⚠️ Speaker identified but profile not found")
                        else:
                            st.warning("⚠️ Could not identify speaker")
                        
                        # Clean up temp file
                        if temp_path.exists():
                            temp_path.unlink()
                            
                    except Exception as e:
                        st.error(f"❌ Testing failed: {e}")
        
        st.subheader("📈 Training Status")
        
        summary = system.get_investigation_summary()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Participants", len(summary['participants']))
        
        with col2:
            st.metric("Total Voice Notes", summary['total_voice_notes'])
        
        with col3:
            avg_confidence = sum(p['confidence_score'] for p in summary['participants']) / len(summary['participants'])
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        
        if st.button("🔄 Refresh Status"):
            st.rerun()


def page_settings():
    """Settings page"""
    st.header("⚙️ Settings")
    
    tab1, tab2, tab3 = st.tabs(["Configuration", "Directories", "About"])
    
    with tab1:
        st.subheader("Application Configuration")
        
        config_summary = get_config_summary()
        
        for key, value in config_summary.items():
            st.write(f"**{key.replace('_', ' ').title()}**: {value}")
    
    with tab2:
        st.subheader("Directory Structure")
        
        directories = {
            "Audio": str(AUDIO_DIR),
            "Text": str(TEXT_DIR),
            "Temp": str(TEMP_DIR),
            "Reports": str(REPORTS_DIR)
        }
        
        for name, path in directories.items():
            st.write(f"**{name}**: `{path}`")
    
    with tab3:
        st.subheader("About Die Waarheid")
        
        st.markdown(f"""
        **{APP_NAME}** v{APP_VERSION}
        
        {APP_DESCRIPTION}
        
        ### Features
        - 🎙️ Advanced audio forensics with bio-signal detection
        - 💬 WhatsApp chat parsing and analysis
        - 🧠 AI-powered psychological profiling
        - 📊 Interactive visualizations
        - 📄 Multi-format report generation
        
        ### Technology Stack
        - **Framework**: Streamlit
        - **Audio**: Librosa, Whisper
        - **AI**: Google Gemini
        - **Visualization**: Plotly
        - **Data**: Pandas, NumPy
        
        ### Author
        AN3S Workspace
        
        ### Legal
        This application is provided for forensic analysis purposes.
        All findings should be verified by qualified professionals.
        """)


def main():
    """Main application entry point"""
    setup_page()
    render_header()
    
    page = render_sidebar()
    
    if page == "🏠 Home":
        page_home()
    elif page == "📥 Data Import":
        page_data_import()
    elif page == "🎙️ Speaker Training":
        page_speaker_training()
    elif page == "🎙️ Audio Analysis":
        page_audio_analysis()
    elif page == "💬 Chat Analysis":
        page_chat_analysis()
    elif page == "🧠 AI Analysis":
        page_ai_analysis()
    elif page == "📈 Visualizations":
        page_visualizations()
    elif page == "📄 Report Generation":
        page_report_generation()
    elif page == "⚙️ Settings":
        page_settings()
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption(f"🕵️ {APP_NAME} v{APP_VERSION}")
    
    with col2:
        st.caption(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    with col3:
        st.caption("AN3S Workspace")


if __name__ == "__main__":
    main()
