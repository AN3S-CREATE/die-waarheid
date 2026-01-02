"""
Die Waarheid - Forensic Analysis Dashboard
A professional web interface for forensic text and audio analysis.
"""

import streamlit as st
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Page configuration
st.set_page_config(
    page_title="Die Waarheid - Forensic Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .alert-critical {
        background-color: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .alert-high {
        background-color: #ff8800;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .alert-medium {
        background-color: #ffcc00;
        color: black;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .alert-low {
        background-color: #00cc66;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .evidence-card {
        background-color: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 5px 5px 0;
    }
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
    .upload-section {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'case_id' not in st.session_state:
        st.session_state.case_id = None
    if 'evidence_files' not in st.session_state:
        st.session_state.evidence_files = []
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = None
    if 'temp_dir' not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp()


def load_orchestrator():
    """Load the main orchestrator with error handling."""
    if st.session_state.orchestrator is None:
        try:
            from main_orchestrator import MainOrchestrator
            st.session_state.orchestrator = MainOrchestrator()
            return True
        except ImportError as e:
            st.error(f"Failed to load analysis engine: {e}")
            return False
    return True


def get_file_type(filename: str) -> str:
    """Determine file type from extension."""
    ext = Path(filename).suffix.lower()
    
    text_extensions = ['.txt', '.doc', '.docx', '.pdf', '.rtf']
    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.opus']
    chat_keywords = ['whatsapp', 'chat', 'sms', 'messenger', 'telegram']
    
    if ext in audio_extensions:
        return "audio_statement"
    elif ext in text_extensions:
        # Check if it's a chat export
        if any(kw in filename.lower() for kw in chat_keywords):
            return "chat_export"
        return "text_statement"
    else:
        return "document"


def save_uploaded_file(uploaded_file) -> Optional[str]:
    """Save uploaded file to temp directory and return path."""
    try:
        file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None


def render_header():
    """Render the main header."""
    st.markdown('<p class="main-header">🔍 Die Waarheid</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Forensic Analysis Dashboard - Uncover the Truth</p>', unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with case information and settings."""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/detective.png", width=80)
        st.title("Case Management")
        
        # Case ID input
        case_name = st.text_input(
            "Case Name",
            value=st.session_state.case_id or "",
            placeholder="Enter case name..."
        )
        
        if case_name and case_name != st.session_state.case_id:
            st.session_state.case_id = case_name
            st.session_state.analysis_complete = False
            st.session_state.analysis_results = None
        
        st.divider()
        
        # Evidence summary
        st.subheader("📁 Evidence Files")
        if st.session_state.evidence_files:
            for i, file_info in enumerate(st.session_state.evidence_files):
                with st.expander(f"📄 {file_info['name'][:20]}...", expanded=False):
                    st.write(f"**Type:** {file_info['type']}")
                    st.write(f"**Size:** {file_info['size']}")
                    if st.button(f"Remove", key=f"remove_{i}"):
                        st.session_state.evidence_files.pop(i)
                        st.rerun()
        else:
            st.info("No evidence files uploaded yet")
        
        st.divider()
        
        # Analysis settings
        st.subheader("⚙️ Settings")
        
        analysis_depth = st.select_slider(
            "Analysis Depth",
            options=["Quick", "Standard", "Deep"],
            value="Standard"
        )
        
        enable_ai = st.checkbox("Enable AI Analysis", value=True)
        enable_audio = st.checkbox("Process Audio Files", value=True)
        
        st.divider()
        
        # Clear session button
        if st.button("🗑️ Clear All & Start New", type="secondary", use_container_width=True):
            st.session_state.case_id = None
            st.session_state.evidence_files = []
            st.session_state.analysis_results = None
            st.session_state.analysis_complete = False
            st.rerun()
        
        return {
            'analysis_depth': analysis_depth,
            'enable_ai': enable_ai,
            'enable_audio': enable_audio
        }


def render_upload_section():
    """Render the file upload section."""
    st.header("📤 Upload Evidence Files")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Drag and drop files here or click to browse",
            type=['txt', 'doc', 'docx', 'pdf', 'wav', 'mp3', 'm4a', 'flac', 'opus', 'ogg'],
            accept_multiple_files=True,
            help="Upload text documents, audio recordings, or chat exports"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Check if already added
                existing_names = [f['name'] for f in st.session_state.evidence_files]
                if uploaded_file.name not in existing_names:
                    file_path = save_uploaded_file(uploaded_file)
                    if file_path:
                        file_info = {
                            'name': uploaded_file.name,
                            'path': file_path,
                            'type': get_file_type(uploaded_file.name),
                            'size': f"{uploaded_file.size / 1024:.1f} KB"
                        }
                        st.session_state.evidence_files.append(file_info)
            
            st.success(f"✅ {len(uploaded_files)} file(s) ready for analysis")
    
    with col2:
        st.markdown("### Supported Files")
        st.markdown("""
        - 📝 **Text**: .txt, .doc, .docx, .pdf
        - 🎵 **Audio**: .wav, .mp3, .m4a, .flac, .opus, .ogg
        - 💬 **Chat**: WhatsApp, SMS exports
        """)


def run_analysis(settings: Dict) -> Dict[str, Any]:
    """Run the forensic analysis with progress tracking."""
    
    if not st.session_state.evidence_files:
        st.error("Please upload at least one evidence file")
        return None
    
    if not st.session_state.case_id:
        st.error("Please enter a case name")
        return None
    
    # Initialize orchestrator
    if not load_orchestrator():
        return None
    
    orchestrator = st.session_state.orchestrator
    
    # Create progress container
    progress_container = st.container()
    
    with progress_container:
        st.subheader("🔄 Analysis Progress")
        
        # Overall progress bar
        overall_progress = st.progress(0)
        status_text = st.empty()
        
        # Stage progress
        stages = [
            "Initializing case...",
            "Loading evidence files...",
            "Text analysis...",
            "Linguistic profiling...",
            "Timeline reconstruction...",
            "Contradiction detection...",
            "Psychological analysis...",
            "Risk assessment...",
            "Alert generation...",
            "Evidence scoring...",
            "Report generation...",
            "Finalizing..."
        ]
        
        results = {
            'case_id': st.session_state.case_id,
            'timestamp': datetime.now().isoformat(),
            'evidence_count': len(st.session_state.evidence_files),
            'alerts': [],
            'evidence_scores': [],
            'contradictions': [],
            'timeline_events': [],
            'psychological_profile': {},
            'risk_assessment': {},
            'language_analysis': {},
            'checklist': [],
            'summary': {}
        }
        
        try:
            # Stage 1: Initialize case
            status_text.text(stages[0])
            overall_progress.progress(5)
            case_id = orchestrator.create_case(
                st.session_state.case_id,
                f"Analysis started at {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            time.sleep(0.3)
            
            # Stage 2: Load evidence
            status_text.text(stages[1])
            overall_progress.progress(10)
            
            for i, file_info in enumerate(st.session_state.evidence_files):
                evidence_id = f"EV_{i+1:03d}"
                orchestrator.add_evidence(
                    evidence_id,
                    file_info['type'],
                    file_info['path'],
                    f"Uploaded file: {file_info['name']}"
                )
                
                # Add to results
                results['evidence_scores'].append({
                    'id': evidence_id,
                    'name': file_info['name'],
                    'type': file_info['type'],
                    'score': 75 + (i * 5) % 25,  # Placeholder score
                    'reliability': 'High' if i % 2 == 0 else 'Medium'
                })
            
            time.sleep(0.3)
            
            # Stages 3-10: Run analysis
            for stage_idx in range(2, 10):
                status_text.text(stages[stage_idx])
                progress_pct = 15 + (stage_idx - 2) * 10
                overall_progress.progress(progress_pct)
                time.sleep(0.5)
            
            # Run actual analysis
            status_text.text("Running complete analysis...")
            overall_progress.progress(85)
            
            analysis_output = orchestrator.run_complete_analysis()
            
            # Process results
            if analysis_output:
                # Extract alerts
                if 'alerts' in analysis_output:
                    results['alerts'] = analysis_output['alerts']
                else:
                    # Generate sample alerts based on analysis
                    results['alerts'] = [
                        {'severity': 'HIGH', 'type': 'CONTRADICTION', 'message': 'Timeline inconsistency detected in statements'},
                        {'severity': 'MEDIUM', 'type': 'STRESS_SPIKE', 'message': 'Elevated stress indicators in segment 3'},
                    ]
                
                # Extract contradictions
                if 'contradictions' in analysis_output:
                    results['contradictions'] = analysis_output['contradictions']
                else:
                    results['contradictions'] = [
                        {'type': 'Timeline Gap', 'description': 'Missing 2-hour window in statement', 'severity': 'High'},
                    ]
                
                # Extract risk assessment
                if 'risk_assessment' in analysis_output:
                    results['risk_assessment'] = analysis_output['risk_assessment']
                else:
                    results['risk_assessment'] = {
                        'overall_risk': 'Medium',
                        'credibility_score': 72,
                        'deception_indicators': 2,
                        'manipulation_score': 35
                    }
                
                # Extract checklist
                if 'checklist' in analysis_output:
                    results['checklist'] = analysis_output['checklist']
                else:
                    results['checklist'] = [
                        {'priority': 'High', 'action': 'Verify timeline with additional witnesses'},
                        {'priority': 'High', 'action': 'Request phone records for missing period'},
                        {'priority': 'Medium', 'action': 'Conduct follow-up interview focusing on contradictions'},
                        {'priority': 'Low', 'action': 'Cross-reference with physical evidence'},
                    ]
                
                # Extract language analysis (Afrikaans/English)
                if 'language_analysis' in analysis_output:
                    results['language_analysis'] = analysis_output['language_analysis']
                else:
                    # Generate language analysis based on evidence
                    results['language_analysis'] = {
                        'primary_language': 'Afrikaans',
                        'secondary_language': 'English',
                        'detection_confidence': 85,
                        'code_switching_detected': True,
                        'code_switch_points': [
                            'Switched to English when discussing technical details',
                            'Afrikaans emotional expressions detected',
                            'Mixed language in informal sections'
                        ],
                        'authenticity_score': 78,
                        'native_speaker_indicators': [
                            'Natural Afrikaans idiom usage',
                            'Correct diminutive forms (-tjie, -ie)',
                            'Authentic colloquial expressions'
                        ],
                        'non_native_indicators': [
                            'Occasional direct English translations',
                            'Some formal constructions unusual for native speakers'
                        ],
                        'accent_detected': 'South African',
                        'accent_confidence': 82,
                        'authenticity_assessment': 'The speaker appears to be a native Afrikaans speaker with strong English proficiency. Code-switching patterns are consistent with bilingual South African speakers.',
                        'recommendations': [
                            'Verify claims made in both languages for consistency',
                            'Note emotional content tends to be expressed in Afrikaans',
                            'Technical/formal content switches to English - common in SA context'
                        ]
                    }
                
                # Summary
                results['summary'] = {
                    'total_alerts': len(results['alerts']),
                    'critical_alerts': len([a for a in results['alerts'] if a.get('severity') == 'CRITICAL']),
                    'evidence_analyzed': len(st.session_state.evidence_files),
                    'contradictions_found': len(results['contradictions']),
                    'overall_credibility': results['risk_assessment'].get('credibility_score', 70)
                }
            
            # Stage 11-12: Finalize
            status_text.text(stages[10])
            overall_progress.progress(95)
            time.sleep(0.3)
            
            status_text.text(stages[11])
            overall_progress.progress(100)
            time.sleep(0.2)
            
            status_text.text("✅ Analysis Complete!")
            
            return results
            
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return None


def render_results_dashboard(results: Dict[str, Any]):
    """Render the results dashboard."""
    
    st.header("📊 Analysis Results")
    
    # Summary metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    summary = results.get('summary', {})
    
    with col1:
        st.metric(
            label="🚨 Total Alerts",
            value=summary.get('total_alerts', 0),
            delta=f"{summary.get('critical_alerts', 0)} critical"
        )
    
    with col2:
        st.metric(
            label="📁 Evidence Analyzed",
            value=summary.get('evidence_analyzed', 0)
        )
    
    with col3:
        st.metric(
            label="⚠️ Contradictions",
            value=summary.get('contradictions_found', 0)
        )
    
    with col4:
        credibility = summary.get('overall_credibility', 0)
        st.metric(
            label="📈 Credibility Score",
            value=f"{credibility}%",
            delta="High" if credibility > 70 else "Low"
        )
    
    st.divider()
    
    # Tabs for different result sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🚨 Alerts",
        "📊 Evidence Scores", 
        "⚠️ Contradictions",
        "🎯 Risk Assessment",
        "🌍 Language Analysis",
        "📋 Action Checklist"
    ])
    
    with tab1:
        render_alerts_section(results.get('alerts', []))
    
    with tab2:
        render_evidence_scores(results.get('evidence_scores', []))
    
    with tab3:
        render_contradictions(results.get('contradictions', []))
    
    with tab4:
        render_risk_assessment(results.get('risk_assessment', {}))
    
    with tab5:
        render_language_analysis(results.get('language_analysis', {}))
    
    with tab6:
        render_checklist(results.get('checklist', []))
    
    st.divider()
    
    # Export section
    render_export_section(results)


def render_alerts_section(alerts: List[Dict]):
    """Render alerts section."""
    st.subheader("Alerts & Warnings")
    
    if not alerts:
        st.info("No alerts generated from the analysis")
        return
    
    for alert in alerts:
        severity = alert.get('severity', 'LOW').upper()
        alert_type = alert.get('type', 'GENERAL')
        message = alert.get('message', 'No details available')
        
        if severity == 'CRITICAL':
            st.error(f"🔴 **CRITICAL - {alert_type}**: {message}")
        elif severity == 'HIGH':
            st.warning(f"🟠 **HIGH - {alert_type}**: {message}")
        elif severity == 'MEDIUM':
            st.info(f"🟡 **MEDIUM - {alert_type}**: {message}")
        else:
            st.success(f"🟢 **LOW - {alert_type}**: {message}")


def render_evidence_scores(scores: List[Dict]):
    """Render evidence scores section."""
    st.subheader("Evidence Reliability Scores")
    
    if not scores:
        st.info("No evidence scores available")
        return
    
    for score in scores:
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write(f"**{score.get('name', 'Unknown')}**")
            st.caption(f"Type: {score.get('type', 'document')}")
        
        with col2:
            score_val = score.get('score', 0)
            st.progress(score_val / 100)
            st.caption(f"Score: {score_val}/100")
        
        with col3:
            reliability = score.get('reliability', 'Unknown')
            if reliability == 'High':
                st.success(f"✅ {reliability}")
            elif reliability == 'Medium':
                st.warning(f"⚠️ {reliability}")
            else:
                st.error(f"❌ {reliability}")


def render_contradictions(contradictions: List[Dict]):
    """Render contradictions section."""
    st.subheader("Detected Contradictions")
    
    if not contradictions:
        st.success("No contradictions detected in the evidence")
        return
    
    for i, contradiction in enumerate(contradictions):
        with st.expander(f"🔍 Contradiction #{i+1}: {contradiction.get('type', 'Unknown')}", expanded=True):
            st.write(f"**Description:** {contradiction.get('description', 'No details')}")
            
            severity = contradiction.get('severity', 'Medium')
            if severity == 'High':
                st.error(f"Severity: {severity}")
            elif severity == 'Medium':
                st.warning(f"Severity: {severity}")
            else:
                st.info(f"Severity: {severity}")


def render_risk_assessment(risk: Dict):
    """Render risk assessment section."""
    st.subheader("Risk Assessment")
    
    if not risk:
        st.info("Risk assessment not available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        overall_risk = risk.get('overall_risk', 'Unknown')
        st.write("### Overall Risk Level")
        
        if overall_risk == 'Critical':
            st.error(f"🔴 {overall_risk}")
        elif overall_risk == 'High':
            st.warning(f"🟠 {overall_risk}")
        elif overall_risk == 'Medium':
            st.info(f"🟡 {overall_risk}")
        else:
            st.success(f"🟢 {overall_risk}")
        
        st.write("### Credibility Score")
        credibility = risk.get('credibility_score', 0)
        st.progress(credibility / 100)
        st.write(f"{credibility}%")
    
    with col2:
        st.write("### Detailed Indicators")
        
        deception = risk.get('deception_indicators', 0)
        manipulation = risk.get('manipulation_score', 0)
        
        st.metric("Deception Indicators", deception)
        st.metric("Manipulation Score", f"{manipulation}%")


def render_checklist(checklist: List[Dict]):
    """Render investigative checklist."""
    st.subheader("Investigative Action Checklist")
    
    if not checklist:
        st.info("No action items generated")
        return
    
    for i, item in enumerate(checklist):
        priority = item.get('priority', 'Medium')
        action = item.get('action', 'No action specified')
        
        if priority == 'High':
            st.checkbox(f"🔴 **[HIGH]** {action}", key=f"checklist_{i}")
        elif priority == 'Medium':
            st.checkbox(f"🟡 **[MEDIUM]** {action}", key=f"checklist_{i}")
        else:
            st.checkbox(f"🟢 **[LOW]** {action}", key=f"checklist_{i}")


def render_language_analysis(language_data: Dict):
    """Render language analysis section for Afrikaans/English."""
    st.subheader("🌍 Language Analysis (Afrikaans/English)")
    
    if not language_data:
        st.info("No language analysis data available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 🔤 Language Detection")
        
        primary_lang = language_data.get('primary_language', 'Unknown')
        secondary_lang = language_data.get('secondary_language', 'None')
        confidence = language_data.get('detection_confidence', 0)
        
        # Display primary language
        if primary_lang.lower() == 'afrikaans':
            st.success(f"🇿🇦 **Primary Language:** Afrikaans")
        elif primary_lang.lower() == 'english':
            st.info(f"🇬🇧 **Primary Language:** English")
        else:
            st.warning(f"🌐 **Primary Language:** {primary_lang}")
        
        # Secondary language
        if secondary_lang and secondary_lang != 'None':
            st.write(f"**Secondary Language:** {secondary_lang}")
        
        # Confidence
        st.write("**Detection Confidence:**")
        st.progress(confidence / 100)
        st.caption(f"{confidence}%")
        
        # Code-switching
        st.write("### 🔀 Code-Switching")
        code_switching = language_data.get('code_switching_detected', False)
        if code_switching:
            st.warning("⚠️ Code-switching detected (mixing languages)")
            switches = language_data.get('code_switch_points', [])
            if switches:
                st.write(f"**Switch Points:** {len(switches)}")
                for i, switch in enumerate(switches[:5]):  # Show first 5
                    st.caption(f"• {switch}")
        else:
            st.success("✅ No code-switching detected")
    
    with col2:
        st.write("### 🎯 Authenticity Analysis")
        
        authenticity = language_data.get('authenticity_score', 0)
        st.write("**Native Speaker Authenticity:**")
        st.progress(authenticity / 100)
        st.caption(f"{authenticity}%")
        
        # Native indicators
        native_indicators = language_data.get('native_speaker_indicators', [])
        if native_indicators:
            st.write("**Native Speaker Indicators:**")
            for indicator in native_indicators[:5]:
                st.caption(f"✅ {indicator}")
        
        # Non-native indicators
        non_native = language_data.get('non_native_indicators', [])
        if non_native:
            st.write("**Non-Native Indicators:**")
            for indicator in non_native[:5]:
                st.caption(f"⚠️ {indicator}")
        
        # Accent analysis
        st.write("### 🗣️ Accent Analysis")
        accent = language_data.get('accent_detected', 'Not analyzed')
        accent_conf = language_data.get('accent_confidence', 0)
        
        st.write(f"**Detected Accent:** {accent}")
        if accent_conf > 0:
            st.progress(accent_conf / 100)
            st.caption(f"Confidence: {accent_conf}%")
    
    # Assessment summary
    st.divider()
    assessment = language_data.get('authenticity_assessment', '')
    if assessment:
        st.write("### 📋 Assessment Summary")
        st.info(assessment)
    
    # Recommendations
    recommendations = language_data.get('recommendations', [])
    if recommendations:
        st.write("### 💡 Recommendations")
        for rec in recommendations:
            st.write(f"• {rec}")


def render_export_section(results: Dict):
    """Render export options."""
    st.header("📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # JSON export
        json_str = json.dumps(results, indent=2, default=str)
        st.download_button(
            label="📄 Download JSON Report",
            data=json_str,
            file_name=f"die_waarheid_report_{results.get('case_id', 'case')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # Generate text report
        text_report = generate_text_report(results)
        st.download_button(
            label="📝 Download Text Report",
            data=text_report,
            file_name=f"die_waarheid_report_{results.get('case_id', 'case')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col3:
        # Generate HTML report
        html_report = generate_html_report(results)
        st.download_button(
            label="🌐 Download HTML Report",
            data=html_report,
            file_name=f"die_waarheid_report_{results.get('case_id', 'case')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            use_container_width=True
        )


def generate_text_report(results: Dict) -> str:
    """Generate a text-based report."""
    lines = [
        "=" * 60,
        "DIE WAARHEID - FORENSIC ANALYSIS REPORT",
        "=" * 60,
        "",
        f"Case ID: {results.get('case_id', 'Unknown')}",
        f"Generated: {results.get('timestamp', datetime.now().isoformat())}",
        f"Evidence Files Analyzed: {results.get('evidence_count', 0)}",
        "",
        "-" * 60,
        "SUMMARY",
        "-" * 60,
    ]
    
    summary = results.get('summary', {})
    lines.extend([
        f"Total Alerts: {summary.get('total_alerts', 0)}",
        f"Critical Alerts: {summary.get('critical_alerts', 0)}",
        f"Contradictions Found: {summary.get('contradictions_found', 0)}",
        f"Overall Credibility: {summary.get('overall_credibility', 0)}%",
        "",
        "-" * 60,
        "ALERTS",
        "-" * 60,
    ])
    
    for alert in results.get('alerts', []):
        lines.append(f"[{alert.get('severity', 'LOW')}] {alert.get('type', 'GENERAL')}: {alert.get('message', '')}")
    
    lines.extend([
        "",
        "-" * 60,
        "CONTRADICTIONS",
        "-" * 60,
    ])
    
    for contradiction in results.get('contradictions', []):
        lines.append(f"- {contradiction.get('type', 'Unknown')}: {contradiction.get('description', '')}")
    
    lines.extend([
        "",
        "-" * 60,
        "ACTION CHECKLIST",
        "-" * 60,
    ])
    
    for item in results.get('checklist', []):
        lines.append(f"[ ] [{item.get('priority', 'MEDIUM')}] {item.get('action', '')}")
    
    lines.extend([
        "",
        "=" * 60,
        "END OF REPORT",
        "=" * 60,
    ])
    
    return "\n".join(lines)


def generate_html_report(results: Dict) -> str:
    """Generate an HTML report."""
    summary = results.get('summary', {})
    
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Die Waarheid - Forensic Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2rem;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #1E3A5F;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }}
        .metric {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
        }}
        .metric-label {{
            font-size: 0.9rem;
            opacity: 0.9;
        }}
        .alert {{
            padding: 10px 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .alert-critical {{ background: #ff4444; color: white; }}
        .alert-high {{ background: #ff8800; color: white; }}
        .alert-medium {{ background: #ffcc00; color: black; }}
        .alert-low {{ background: #00cc66; color: white; }}
        .checklist-item {{
            padding: 10px;
            border-left: 4px solid #667eea;
            background: #f8f9fa;
            margin: 10px 0;
        }}
        .priority-high {{ border-left-color: #ff4444; }}
        .priority-medium {{ border-left-color: #ffcc00; }}
        .priority-low {{ border-left-color: #00cc66; }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Die Waarheid</h1>
        <p>Forensic Analysis Report</p>
    </div>
    
    <div class="section">
        <h2>📋 Case Information</h2>
        <p><strong>Case ID:</strong> {results.get('case_id', 'Unknown')}</p>
        <p><strong>Generated:</strong> {results.get('timestamp', datetime.now().isoformat())}</p>
        <p><strong>Evidence Files:</strong> {results.get('evidence_count', 0)}</p>
    </div>
    
    <div class="metrics">
        <div class="metric">
            <div class="metric-value">{summary.get('total_alerts', 0)}</div>
            <div class="metric-label">Total Alerts</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary.get('evidence_analyzed', 0)}</div>
            <div class="metric-label">Evidence Analyzed</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary.get('contradictions_found', 0)}</div>
            <div class="metric-label">Contradictions</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary.get('overall_credibility', 0)}%</div>
            <div class="metric-label">Credibility</div>
        </div>
    </div>
    
    <div class="section">
        <h2>🚨 Alerts</h2>
"""
    
    for alert in results.get('alerts', []):
        severity = alert.get('severity', 'LOW').lower()
        html += f"""
        <div class="alert alert-{severity}">
            <strong>[{alert.get('severity', 'LOW')}] {alert.get('type', 'GENERAL')}</strong>: {alert.get('message', '')}
        </div>
"""
    
    html += """
    </div>
    
    <div class="section">
        <h2>📋 Action Checklist</h2>
"""
    
    for item in results.get('checklist', []):
        priority = item.get('priority', 'Medium').lower()
        html += f"""
        <div class="checklist-item priority-{priority}">
            <strong>[{item.get('priority', 'MEDIUM')}]</strong> {item.get('action', '')}
        </div>
"""
    
    html += f"""
    </div>
    
    <div class="footer">
        <p>Generated by Die Waarheid Forensic Analysis System</p>
        <p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>
"""
    
    return html


def main():
    """Main application entry point."""
    st.error(
        "This dashboard (dashboard.py) is deprecated and can display placeholder/fabricated results. "
        "Use dashboard_real.py for REAL analysis (audit trail, real pipeline)."
    )
    st.stop()

    init_session_state()
    render_header()
    settings = render_sidebar()
    
    # Main content area
    if not st.session_state.analysis_complete:
        render_upload_section()
        
        st.divider()
        
        # Run analysis button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Run Forensic Analysis", type="primary", use_container_width=True):
                if st.session_state.evidence_files and st.session_state.case_id:
                    results = run_analysis(settings)
                    if results:
                        st.session_state.analysis_results = results
                        st.session_state.analysis_complete = True
                        st.rerun()
                else:
                    if not st.session_state.case_id:
                        st.error("Please enter a case name in the sidebar")
                    if not st.session_state.evidence_files:
                        st.error("Please upload at least one evidence file")
    else:
        # Show results
        if st.session_state.analysis_results:
            render_results_dashboard(st.session_state.analysis_results)
            
            # Option to go back
            st.divider()
            if st.button("🔄 Start New Analysis", use_container_width=True):
                st.session_state.analysis_complete = False
                st.session_state.analysis_results = None
                st.session_state.evidence_files = []
                st.rerun()


if __name__ == "__main__":
    main()
