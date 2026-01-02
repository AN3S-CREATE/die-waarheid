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
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Messages", "0", "Ready to analyze")
    
    with col2:
        st.metric("Audio Files", "0", "Ready to process")
    
    with col3:
        st.metric("Analysis Status", "Idle", "Waiting for input")
    
    st.divider()
    
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
            chat_file = st.file_uploader("Select WhatsApp chat export (.txt)", type=['txt'])
            
            if chat_file:
                with st.spinner("Processing chat file..."):
                    parser = WhatsAppParser()
                    success, message = parser.parse_file(Path(TEMP_DIR) / chat_file.name)
                    
                    if success:
                        st.success(message)
                        st.session_state.chat_parser = parser
                        st.info(f"Metadata: {parser.get_metadata()}")
        
        with col2:
            st.write("**Upload Audio Files**")
            audio_files = st.file_uploader(
                "Select audio files",
                type=['mp3', 'wav', 'opus', 'ogg', 'm4a', 'aac'],
                accept_multiple_files=True
            )
            
            if audio_files:
                st.info(f"Selected {len(audio_files)} audio files")
    
    with tab3:
        st.subheader("Import Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Chat Files", "0")
        
        with col2:
            st.metric("Audio Files", "0")
        
        with col3:
            st.metric("Last Import", "Never")


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
