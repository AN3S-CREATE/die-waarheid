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
                "🎙️ Transcribe Audio",  # Main function - transcription
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
                st.session_state.current_page = "📥 Data Import"
                st.rerun()
        
        with col2:
            if st.button("🎙️ Train Speakers"):
                st.session_state.current_page = "🎙️ Speaker Training"
                st.rerun()
        
        with col3:
            if st.button("📈 View Analytics"):
                st.session_state.current_page = "📈 Visualizations"
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
                try:
                    handler = GDriveHandler()
                except Exception as e:
                    st.error(f"Failed to initialize Google Drive: {str(e)}")
                    return
                
                try:
                    success, message = handler.authenticate()
                except Exception as e:
                    success, message = False, f"Authentication failed: {str(e)}"
                
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
        logger.info(f"Found {len(audio_subdirs)} audio subdirectories")
        
        # Count audio files by type for debugging
        audio_counts = {}
        for ext in ['mp3', 'wav', 'opus', 'ogg', 'm4a', 'aac']:
            audio_counts[ext] = len(list(AUDIO_DIR.rglob(f"*.{ext}")))
        
        logger.info(f"Audio files by type: {audio_counts}")
        
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


def page_transcribe_audio():
    """Main transcription page - core function"""
    st.header("🎙️ Transcribe Audio")
    st.subheader("📝 Convert Voice Notes to Text")
    
    st.write("""
    **Main Function**: Transcribe your voice notes to readable text using Whisper AI.
    
    This is the core functionality - convert your 10,241 voice notes into searchable text.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🎯 Transcription Settings**")
        
        # Model selection
        model_size = st.selectbox(
            "Whisper Model (Smaller = Faster, Larger = More Accurate)",
            ["tiny", "base", "small", "medium", "large"],
            index=2,  # Default to "small" for balance
            help="Tiny: Fastest, Large: Most accurate"
        )
        
        # Language selection
        language = st.selectbox(
            "Language",
            [
                ("af", "Afrikaans"),
                ("en", "English"), 
                ("nl", "Dutch"),
                ("auto", "Auto-detect")
            ],
            index=0,
            help="Select the language spoken in the audio"
        )
        
        # Processing options
        batch_process = st.checkbox("Process Multiple Files", value=True)
        
    with col2:
        st.write("**📁 File Selection**")
        
        if batch_process:
            st.write("**Batch Processing Mode**")
            st.info(f"Found {len(list(AUDIO_DIR.rglob('*.opus')))} opus files ready")
            
            if st.button("🚀 Transcribe All Voice Notes"):
                transcribe_all_files(model_size, language)
        else:
            st.write("**Single File Mode**")
            # Single file upload
            audio_file = st.file_uploader(
                "Upload audio file to transcribe",
                type=['mp3', 'wav', 'opus', 'ogg', 'm4a', 'aac'],
                key="transcribe_single"
            )
            
            if audio_file and st.button("🎙️ Transcribe File"):
                transcribe_single_file(audio_file, model_size, language)
    
    # Recent transcriptions
    st.divider()
    st.subheader("📋 Recent Transcriptions")
    
    if 'transcriptions' in st.session_state and st.session_state.transcriptions:
        for i, transcription in enumerate(st.session_state.transcriptions[-5:]):
            with st.expander(f"📝 {transcription['filename']}"):
                st.write("**Transcription:**")
                st.write(transcription['text'])
                st.write(f"**Language:** {transcription['language']}")
                st.write(f"**Model:** {transcription['model']}")
                st.write(f"**Duration:** {transcription['duration']:.2f}s")
                
                if st.button(f"📥 Copy Text {i}", key=f"copy_{i}"):
                    st.write("Text copied to clipboard!")
    else:
        st.info("No transcriptions yet. Upload and transcribe audio files to see results here.")


def transcribe_single_file(audio_file, model_size, language):
    """Transcribe a single audio file"""
    with st.spinner(f"Loading {model_size} model and transcribing..."):
        try:
            # Save uploaded file temporarily
            temp_path = Path(TEMP_DIR) / audio_file.name
            with open(temp_path, "wb") as f:
                f.write(audio_file.getbuffer())
            
            # Initialize transcriber
            from src.whisper_transcriber import WhisperTranscriber
            transcriber = WhisperTranscriber(model_size)
            
            # Transcribe
            result = transcriber.transcribe(temp_path, language=language)
            
            if result['success']:
                st.success("✅ Transcription complete!")
                
                # Store result
                if 'transcriptions' not in st.session_state:
                    st.session_state.transcriptions = []
                
                st.session_state.transcriptions.append({
                    'filename': audio_file.name,
                    'text': result['text'],
                    'language': language,
                    'model': model_size,
                    'duration': result.get('duration', 0)
                })
                
                # Display result
                st.subheader("📝 Transcription Result:")
                st.text_area("Transcribed Text:", result['text'], height=200)
                
                # Copy button
                if st.button("📋 Copy to Clipboard"):
                    st.write("Text copied!")
                
                # Download button
                st.download_button(
                    label="📥 Download Text",
                    data=result['text'],
                    file_name=f"transcription_{audio_file.name}.txt",
                    mime="text/plain"
                )
                
            else:
                st.error(f"❌ Transcription failed: {result['message']}")
            
            # Clean up
            if temp_path.exists():
                temp_path.unlink()
                
        except Exception as e:
            st.error(f"❌ Error during transcription: {e}")


def transcribe_all_files(model_size, language):
    """Transcribe all audio files in batch"""
    with st.spinner("Initializing batch transcription..."):
        try:
            from src.whisper_transcriber import WhisperTranscriber
            transcriber = WhisperTranscriber(model_size)
            
            # Get all audio files
            audio_files = []
            for ext in ['mp3', 'wav', 'opus', 'ogg', 'm4a', 'aac']:
                audio_files.extend(AUDIO_DIR.rglob(f"*.{ext}"))
            
            if not audio_files:
                st.warning("⚠️ No audio files found")
                return
            
            st.info(f"📁 Found {len(audio_files)} files to transcribe")
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process files
            transcribed_count = 0
            failed_count = 0
            
            for i, audio_file in enumerate(audio_files):
                status_text.text(f"Transcribing {audio_file.name} ({i+1}/{len(audio_files)})")
                
                try:
                    result = transcriber.transcribe(audio_file, language=language)
                    
                    if result['success']:
                        # Store result
                        if 'transcriptions' not in st.session_state:
                            st.session_state.transcriptions = []
                        
                        st.session_state.transcriptions.append({
                            'filename': audio_file.name,
                            'text': result['text'],
                            'language': language,
                            'model': model_size,
                            'duration': result.get('duration', 0)
                        })
                        
                        transcribed_count += 1
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    st.error(f"Error with {audio_file.name}: {e}")
                    failed_count += 1
                
                # Update progress
                progress = (i + 1) / len(audio_files)
                progress_bar.progress(progress)
            
            # Summary
            st.success(f"✅ Batch transcription complete!")
            st.info(f"Transcribed: {transcribed_count} files | Failed: {failed_count} files")
            
            if transcribed_count > 0:
                # Download all transcriptions
                if st.button("📥 Download All Transcriptions"):
                    all_text = "\n\n".join([f"=== {t['filename']} ===\n{t['text']}" for t in st.session_state.transcriptions])
                    st.download_button(
                        label="Download Complete Transcript",
                        data=all_text,
                        file_name="all_transcriptions.txt",
                        mime="text/plain"
                    )
            
        except Exception as e:
            st.error(f"❌ Batch transcription failed: {e}")


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
        with st.spinner("Initializing forensics engine..."):
            try:
                from src.forensics import ForensicsEngine
                engine = ForensicsEngine(use_cache=True)
                st.success("✅ Forensics engine initialized")
                
                # Get audio files from your data directory
                audio_files = []
                for ext in ['mp3', 'wav', 'opus', 'ogg', 'm4a', 'aac']:
                    audio_files.extend(AUDIO_DIR.rglob(f"*.{ext}"))
                
                if not audio_files:
                    st.warning("⚠️ No audio files found in data/audio directory")
                    return
                
                st.info(f"📁 Found {len(audio_files)} audio files")
                
                # Batch processing options
                batch_size = st.selectbox("Batch Size", [10, 50, 100, 500], index=1)
                max_files = st.selectbox("Max Files to Process", [100, 500, 1000, "All"], index=0)
                
                if max_files != "All":
                    audio_files = audio_files[:max_files]
                
                # Process files
                progress_bar = st.progress(0)
                results_container = st.container()
                
                processed_results = []
                
                for i, audio_file in enumerate(audio_files[:batch_size]):
                    with st.spinner(f"Processing {audio_file.name}..."):
                        try:
                            # Real forensic analysis
                            result = engine.analyze(audio_file)
                            
                            if result['success']:
                                processed_results.append({
                                    'filename': audio_file.name,
                                    'stress_level': result['stress_level'],
                                    'pitch_volatility': result['pitch_volatility'],
                                    'silence_ratio': result['silence_ratio'],
                                    'duration': result['duration'],
                                    'intensity_max': result['intensity']['max'],
                                    'spectral_centroid': result['spectral_centroid']
                                })
                                
                                # Show real results
                                with results_container:
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric(f"Stress: {audio_file.name[:20]}...", f"{result['stress_level']:.2f}")
                                    with col2:
                                        st.metric("Pitch Volatility", f"{result['pitch_volatility']:.2f}")
                                    with col3:
                                        st.metric("Silence Ratio", f"{result['silence_ratio']:.3f}")
                            else:
                                st.error(f"❌ Failed to analyze {audio_file.name}: {result['message']}")
                            
                            # Update progress
                            progress = (i + 1) / min(batch_size, len(audio_files))
                            progress_bar.progress(progress)
                            
                        except Exception as e:
                            st.error(f"❌ Error processing {audio_file.name}: {e}")
                
                # Summary statistics
                if processed_results:
                    st.subheader("📊 Analysis Results")
                    
                    avg_stress = sum(r['stress_level'] for r in processed_results) / len(processed_results)
                    avg_pitch_vol = sum(r['pitch_volatility'] for r in processed_results) / len(processed_results)
                    avg_silence = sum(r['silence_ratio'] for r in processed_results) / len(processed_results)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Files Analyzed", len(processed_results))
                    with col2:
                        st.metric("Avg Stress Level", f"{avg_stress:.2f}")
                    with col3:
                        st.metric("Avg Pitch Volatility", f"{avg_pitch_vol:.2f}")
                    with col4:
                        st.metric("Avg Silence Ratio", f"{avg_silence:.3f}")
                    
                    # High stress files
                    high_stress = [r for r in processed_results if r['stress_level'] > 50]
                    if high_stress:
                        st.subheader("🚨 High Stress Files")
                        for result in high_stress[:5]:  # Show top 5
                            st.write(f"📄 {result['filename']}: Stress {result['stress_level']:.2f}")
                    
                    # Download results
                    if st.button("📥 Download Results"):
                        import json
                        results_json = json.dumps(processed_results, indent=2)
                        st.download_button(
                            label="Download Analysis Results",
                            data=results_json,
                            file_name="forensic_analysis_results.json",
                            mime="application/json"
                        )
                
            except Exception as e:
                st.error(f"❌ Failed to initialize forensics engine: {e}")
                st.write("Make sure all dependencies are installed and audio files are accessible.")


def page_chat_analysis():
    """Chat analysis page"""
    st.header("💬 Chat Analysis")
    
    st.subheader("WhatsApp Message Analysis")
    
    if st.button("📊 Analyze Chat Messages"):
        with st.spinner("Analyzing chat messages..."):
            try:
                from src.chat_parser import WhatsAppParser
                parser = WhatsAppParser()
                
                # Get all text files
                chat_files = list(TEXT_DIR.glob("*.txt"))
                
                if not chat_files:
                    st.warning("⚠️ No chat files found in data/text directory")
                    return
                
                total_messages = 0
                unique_senders = set()
                date_range = []
                
                for chat_file in chat_files:
                    success, message = parser.parse_file(chat_file)
                    if success:
                        # Get messages from database or parser
                        # This is simplified - in real implementation would get from DB
                        with open(chat_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            lines = content.split('\n')
                            

                        for line in lines:
                            if '[' in line and ']' in line and ':' in line:
                                total_messages += 1
                                # Extract sender (simplified)
                                if '] ' in line:
                                    sender = line.split('] ')[1].split(':')[0]
                                    unique_senders.add(sender)
                
                # Display real results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Message Statistics**")
                    st.metric("Total Messages", total_messages)
                    st.metric("Unique Senders", len(unique_senders))
                    st.metric("Chat Files", len(chat_files))
                
                with col2:
                    st.write("**Analysis Status**")
                    st.metric("Messages Processed", total_messages)
                    st.metric("Senders Identified", len(unique_senders))
                    st.metric("Files Analyzed", len(chat_files))
                
                if unique_senders:
                    st.subheader("👥 Identified Senders:")
                    for sender in sorted(list(unique_senders))[:10]:
                        st.write(f"• {sender}")
                    if len(unique_senders) > 10:
                        st.write(f"... and {len(unique_senders) - 10} more")
                
            except Exception as e:
                st.error(f"❌ Chat analysis failed: {e}")


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
            logger.info("Attempting to initialize speaker system...")
            st.session_state.speaker_system = SpeakerIdentificationSystem("MAIN_CASE")
            st.success("✅ Speaker identification system initialized")
        except Exception as e:
            st.error(f"❌ Failed to initialize speaker system: {e}")
            import traceback
            st.error(f"🔍 Full error: {traceback.format_exc()}")
            return
    
    system = st.session_state.speaker_system
    
    # Check if investigation is initialized
    participants = system.get_all_participants()
    
    if len(participants) < 2:
        st.subheader("🔧 Initialize Investigation")
        st.write("First, set up the two participants for this investigation:")
        
        # Add reset button
        if st.button("🗑️ Clear All Speaker Data", type="secondary"):
            try:
                # Clear session state
                if 'speaker_system' in st.session_state:
                    del st.session_state.speaker_system
                
                # Clear database (delete investigations.db)
                from config import DATA_DIR
                db_path = DATA_DIR / "investigations.db"
                if db_path.exists():
                    db_path.unlink()
                    st.success("✅ Speaker database cleared")
                
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to clear data: {e}")
        
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
        
        st.divider()
        
        # Background Audio Analysis Section
        st.subheader("🔍 Background Audio Analysis")
        st.write("Advanced background noise detection, vocal separation, and environmental analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🎵 Analyze Background Audio**")
            
            background_audio = st.file_uploader(
                "Upload audio file for background analysis",
                type=['mp3', 'wav', 'opus', 'ogg', 'm4a', 'aac'],
                key="background_audio"
            )
            
            if background_audio:
                st.info(f"📁 Selected: {background_audio.name}")
                
                if st.button("🔍 Analyze Background Audio", key="analyze_background"):
                    with st.spinner("Analyzing background audio..."):
                        try:
                            # Save audio file temporarily
                            temp_path = Path(TEMP_DIR) / background_audio.name
                            with open(temp_path, "wb") as f:
                                f.write(background_audio.getbuffer())
                            
                            # Analyze background audio
                            from src.background_audio import analyze_background_audio
                            analysis_result = analyze_background_audio(temp_path)
                            
                            if "error" not in analysis_result:
                                st.success("✅ Background audio analysis complete!")
                                
                                # Store results in session state
                                if 'background_analyses' not in st.session_state:
                                    st.session_state.background_analyses = []
                                st.session_state.background_analyses.append(analysis_result)
                                
                                # Clean up temp file
                                if temp_path.exists():
                                    temp_path.unlink()
                                
                                st.rerun()
                            else:
                                st.error(f"❌ Analysis failed: {analysis_result['error']}")
                                
                        except Exception as e:
                            st.error(f"❌ Background analysis failed: {e}")
        
        with col2:
            st.write("**📊 Recent Analyses**")
            
            if 'background_analyses' in st.session_state and st.session_state.background_analyses:
                for i, analysis in enumerate(st.session_state.background_analyses[-3:]):  # Show last 3
                    with st.expander(f"🔍 {Path(analysis['file_path']).name}"):
                        st.write(f"**Duration:** {analysis['duration']:.2f}s")
                        st.write(f"**Overall Clarity:** {analysis['overall_clarity']:.2f}")
                        
                        # Background noise summary
                        if analysis['background_noise']:
                            noise_types = [n.noise_type for n in analysis['background_noise']]
                            st.write(f"**Noise Types:** {', '.join(noise_types)}")
                        
                        # Background vocals summary
                        if analysis['background_vocals']:
                            vocal_count = len(analysis['background_vocals'])
                            st.write(f"**Vocal Segments:** {vocal_count}")
                        
                        # Environmental context
                        env = analysis['environmental_context']
                        st.write(f"**Location:** {env.location_type}")
                        st.write(f"**Room Size:** {env.room_size}")
                        st.write(f"**Crowd Density:** {env.crowd_density}")
                        
                        # Audio separation
                        separation = analysis['audio_separation']
                        st.write(f"**Signal-to-Background:** {separation['signal_to_background_ratio']:.1f} dB")
                        
                        if st.button(f"📋 View Full Report {i}", key=f"full_report_{i}"):
                            st.json(analysis)
            else:
                st.info("No background analyses yet. Upload and analyze audio files to see results here.")
        
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
    
    # Get current page from session state or sidebar
    if "current_page" not in st.session_state:
        st.session_state.current_page = render_sidebar()
    else:
        page = render_sidebar()
        if page != st.session_state.current_page:
            st.session_state.current_page = page
    
    page = st.session_state.current_page
    
    if page == "🏠 Home":
        page_home()
    elif page == "📥 Data Import":
        page_data_import()
    elif page == "🎙️ Transcribe Audio":
        page_transcribe_audio()
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
