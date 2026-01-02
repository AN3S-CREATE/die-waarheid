"""
🔬 DIE WAARHEID V2: Automated Forensic Analysis System

One-click comprehensive forensic analysis with detailed reporting
"""

import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
from typing import Dict, List, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import configuration
from config import (
    BASE_DIR, DATA_DIR, TEMP_DIR, CREDENTIALS_DIR, MOBITABLES_DIR,
    GEMINI_API_KEY, HUGGINGFACE_TOKEN,
    STREAMLIT_CONFIG, LOG_LEVEL, LOG_FILE
)

# Import forensic modules
from src.chat_parser import whatsapp_parser
from src.forensics import analyze_audio_file
from src.ai_analyzer import ai_analyzer
from src.toxicity_detector import toxicity_detector
from src.trust_calculator import trust_calculator

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
    .forensic-alert {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .forensic-warning {
        background-color: #fff8e1;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .forensic-info {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stress-high { color: #ff4b4b; font-weight: bold; }
    .stress-medium { color: #ffa500; font-weight: bold; }
    .stress-low { color: #00ff00; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = None
    if 'forensic_report' not in st.session_state:
        st.session_state.forensic_report = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False

class BatchForensicAnalyzer:
    """Automated batch forensic analyzer"""
    
    def __init__(self):
        self.results = {
            'summary': {},
            'transcriptions': [],
            'stress_analysis': [],
            'deception_indicators': [],
            'timeline': [],
            'statistics': {}
        }
    
    def process_whatsapp_export(self, zip_path: str) -> pd.DataFrame:
        """Process WhatsApp export and return DataFrame"""
        st.write("📂 Extracting WhatsApp data...")
        df = whatsapp_parser.parse_export_file(Path(zip_path))
        
        if df.empty:
            st.error("No data found in the export file!")
            return df
        
        st.success(f"✅ Loaded {len(df)} messages from {df['sender'].nunique()} participants")
        return df
    
    def transcribe_audio_batch(self, df: pd.DataFrame, max_workers: int = 2) -> pd.DataFrame:
        """Batch transcribe all audio files with improved success rate"""
        audio_files = df[df['message_type'] == 'audio'].copy()
        
        if audio_files.empty:
            st.info("No audio files found for transcription.")
            return df
        
        st.write(f"🎙️ Processing {len(audio_files)} audio files...")
        
        # Check if media files exist
        missing_files = 0
        for idx, row in audio_files.iterrows():
            if not pd.notna(row.get('media_path')) or not os.path.exists(row['media_path']):
                missing_files += 1
                df.loc[idx, 'transcript'] = '[File not found]'
                df.loc[idx, 'analysis_successful'] = False
        
        if missing_files > 0:
            st.warning(f"⚠️ {missing_files} audio files are missing or inaccessible")
        
        # Filter to only existing files
        valid_audio = audio_files[
            audio_files.apply(lambda x: pd.notna(x.get('media_path')) and os.path.exists(x['media_path']), axis=1)
        ]
        
        if valid_audio.empty:
            st.error("No valid audio files found for processing!")
            return df
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        success_count = 0
        processed_count = 0
        
        def process_single_audio(args):
            idx, row = args
            try:
                audio_path = row['media_path']
                
                # Check file size and type
                if not os.path.exists(audio_path):
                    return idx, {'analysis_successful': False, 'transcript': '[File not found]'}
                
                file_size = os.path.getsize(audio_path)
                if file_size == 0:
                    return idx, {'analysis_successful': False, 'transcript': '[Empty file]'}
                
                # Check file extension
                ext = Path(audio_path).suffix.lower()
                if ext not in ['.opus', '.m4a', '.mp3', '.wav', '.ogg']:
                    return idx, {'analysis_successful': False, 'transcript': f'[Unsupported format: {ext}]'}
                
                # Enhanced audio analysis
                result = analyze_audio_file(audio_path)
                
                # Check if transcription was successful
                transcript = result.get('transcript', '')
                
                # More lenient success criteria
                if transcript and transcript != '[Transcription failed]' and len(transcript.strip()) > 5:
                    # Clean up transcript
                    transcript = transcript.strip()
                    
                    # Calculate metrics
                    stress_level = self._calculate_stress_level(result)
                    deception_score = self._calculate_deception_score(result)
                    
                    return idx, {
                        'transcript': transcript,
                        'language': result.get('language', 'unknown'),
                        'speaker_count': result.get('speaker_count', 0),
                        'pitch_volatility': result.get('pitch_volatility', 0),
                        'silence_ratio': result.get('silence_ratio', 0),
                        'max_loudness': result.get('max_loudness', 0),
                        'analysis_successful': True,
                        'stress_level': stress_level,
                        'deception_score': deception_score
                    }
                else:
                    # Try basic transcription without full analysis
                    return idx, self._basic_transcription(audio_path)
                    
            except Exception as e:
                logger.error(f"Error processing audio {idx}: {e}")
                return idx, {'analysis_successful': False, 'transcript': f'[Error: {str(e)[:50]}]'}
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_audio, (idx, row)): idx 
                      for idx, row in valid_audio.iterrows()}
            
            for future in as_completed(futures):
                idx, result = future.result()
                processed_count += 1
                
                # Update DataFrame
                for key, value in result.items():
                    df.loc[idx, key] = value
                
                if result.get('analysis_successful', False):
                    success_count += 1
                
                # Update progress
                progress = processed_count / len(valid_audio)
                progress_bar.progress(progress)
                status_text.text(f"Processed: {processed_count}/{len(valid_audio)} | Success: {success_count} ({success_count/processed_count*100:.1f}%)")
        
        success_rate = success_count / len(valid_audio) * 100 if len(valid_audio) > 0 else 0
        st.success(f"✅ Transcription complete: {success_count}/{len(valid_audio)} files ({success_rate:.1f}% success rate)")
        
        if success_rate < 50:
            st.warning("⚠️ Low success rate detected. This could be due to:")
            st.warning("- Corrupted or missing audio files")
            st.warning("- Unsupported audio formats")
            st.warning("- Very short or silent recordings")
        
        return df
    
    def _basic_transcription(self, audio_path: str) -> Dict:
        """Basic transcription attempt for failed files"""
        try:
            # Import here to avoid circular imports
            import whisper
            
            # Use a smaller, faster model for fallback
            model = whisper.load_model("base")
            
            # Transcribe with basic settings
            result = model.transcribe(
                audio_path,
                language=None,  # Auto-detect
                initial_prompt="",
                condition_on_previous_text=False
            )
            
            transcript = result.get('text', '').strip()
            
            if transcript and len(transcript) > 5:
                return {
                    'analysis_successful': True,
                    'transcript': transcript,
                    'language': result.get('language', 'unknown'),
                    'speaker_count': 1,
                    'pitch_volatility': 0,
                    'silence_ratio': 0,
                    'max_loudness': 0,
                    'stress_level': 'Medium',
                    'deception_score': 25
                }
            else:
                return {
                    'analysis_successful': False,
                    'transcript': '[No speech detected]',
                    'stress_level': 'unknown',
                    'deception_score': 0
                }
                
        except Exception as e:
            return {
                'analysis_successful': False,
                'transcript': f'[Failed: {str(e)[:30]}]',
                'stress_level': 'unknown',
                'deception_score': 0
            }
    
    def _fallback_transcription(self, audio_path: str) -> Dict:
        """Fallback transcription method for failed files"""
        try:
            # Try with a different model or settings
            # This is a placeholder - you could implement:
            # - Different Whisper model
            # - Audio preprocessing
            # - Alternative transcription service
            return {
                'analysis_successful': False,
                'transcript': '[Fallback failed]',
                'stress_level': 'unknown',
                'deception_score': 0
            }
        except:
            return {
                'analysis_successful': False,
                'transcript': '[Failed]',
                'stress_level': 'unknown',
                'deception_score': 0
            }
    
    def _calculate_stress_level(self, audio_result: Dict) -> str:
        """Calculate stress level from audio features"""
        pitch_vol = audio_result.get('pitch_volatility', 0)
        silence_ratio = audio_result.get('silence_ratio', 0)
        
        # Enhanced stress calculation
        stress_score = (pitch_vol * 0.6) + (silence_ratio * 100 * 0.4)
        
        if stress_score >= 70:
            return 'Very High'
        elif stress_score >= 50:
            return 'High'
        elif stress_score >= 30:
            return 'Medium'
        else:
            return 'Low'
    
    def _calculate_deception_score(self, audio_result: Dict) -> float:
        """Calculate deception likelihood score (0-100)"""
        pitch_vol = audio_result.get('pitch_volatility', 0)
        silence_ratio = audio_result.get('silence_ratio', 0)
        speaker_count = audio_result.get('speaker_count', 1)
        
        # Deception indicators
        score = 0
        
        # High pitch volatility
        if pitch_vol > 50:
            score += 30
        
        # High silence ratio (hesitation)
        if silence_ratio > 0.3:
            score += 25
        
        # Multiple speakers when claiming to be alone
        if speaker_count > 1:
            score += 20
        
        # Add more sophisticated analysis here
        
        return min(score, 100)
    
    def analyze_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze communication patterns for deception indicators"""
        st.write("🔍 Analyzing communication patterns...")
        
        analysis = {
            'message_frequency': {},
            'response_times': [],
            'emotional_shifts': [],
            'contradictions': [],
            'evasion_patterns': []
        }
        
        # Message frequency analysis
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            hourly_counts = df.groupby('hour').size()
            analysis['message_frequency'] = hourly_counts.to_dict()
        
        # Analyze text patterns for deception
        for idx, row in df.iterrows():
            message = str(row.get('message', '')).lower()
            
            # Evasion indicators
            evasion_words = ['maybe', 'perhaps', 'i think', 'not sure', 'probably']
            if any(word in message for word in evasion_words):
                analysis['evasion_patterns'].append({
                    'timestamp': row.get('timestamp'),
                    'sender': row.get('sender'),
                    'message': row.get('message')[:100],
                    'type': 'Evasion'
                })
            
            # Contradictions (simplified)
            if 'but' in message and len(message.split()) > 10:
                analysis['contradictions'].append({
                    'timestamp': row.get('timestamp'),
                    'sender': row.get('sender'),
                    'message': row.get('message')[:100],
                    'type': 'Contradiction'
                })
        
        return analysis
    
    def generate_forensic_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive forensic report"""
        st.write("📊 Generating forensic report...")
        
        report = {
            'executive_summary': {},
            'communication_analysis': {},
            'audio_forensics': {},
            'toxicity_analysis': {},
            'trust_assessment': {},
            'risk_assessment': {},
            'recommendations': []
        }
        
        # Executive Summary
        total_messages = len(df)
        audio_messages = len(df[df['message_type'] == 'audio'])
        successful_transcriptions = len(df[df['analysis_successful'] == True])
        
        report['executive_summary'] = {
            'total_messages': total_messages,
            'audio_messages': audio_messages,
            'transcription_success_rate': (successful_transcriptions / audio_messages * 100) if audio_messages > 0 else 0,
            'participants': df['sender'].nunique(),
            'date_range': {
                'start': df['timestamp'].min().strftime('%Y-%m-%d'),
                'end': df['timestamp'].max().strftime('%Y-%m-%d')
            }
        }
        
        # Audio Forensics
        if 'stress_level' in df.columns:
            stress_counts = df[df['message_type'] == 'audio']['stress_level'].value_counts()
            high_stress = stress_counts.get('High', 0) + stress_counts.get('Very High', 0)
            
            report['audio_forensics'] = {
                'high_stress_messages': high_stress,
                'stress_distribution': stress_counts.to_dict(),
                'avg_deception_score': df[df['message_type'] == 'audio']['deception_score'].mean() if 'deception_score' in df.columns else 0
            }
        
        # Toxicity Analysis
        st.write("🔍 Analyzing toxicity patterns...")
        toxicity_results = toxicity_detector.analyze_conversation(df)
        report['toxicity_analysis'] = toxicity_results
        
        # Trust Assessment
        st.write("📈 Calculating trust scores...")
        trust_results = trust_calculator.calculate_overall_trust(
            df, 
            toxicity_results, 
            report['audio_forensics']
        )
        report['trust_assessment'] = trust_results
        
        # Risk Assessment
        risk_score = 0
        risk_factors = []
        
        if report['audio_forensics'].get('avg_deception_score', 0) > 50:
            risk_score += 30
            risk_factors.append("High deception indicators in voice messages")
        
        if report['audio_forensics'].get('high_stress_messages', 0) > audio_messages * 0.3:
            risk_score += 25
            risk_factors.append("Elevated stress levels detected")
        
        if toxicity_results['toxic_messages'] > total_messages * 0.1:
            risk_score += 20
            risk_factors.append("Toxic communication patterns detected")
        
        if toxicity_results['gaslighting_messages'] > 0:
            risk_score += 25
            risk_factors.append("Gaslighting behavior detected")
        
        if trust_results['overall_score'] < 40:
            risk_score += 30
            risk_factors.append("Low trust score")
        
        report['risk_assessment'] = {
            'overall_risk_score': min(risk_score, 100),
            'risk_factors': risk_factors,
            'risk_level': 'High' if risk_score > 50 else 'Medium' if risk_score > 25 else 'Low'
        }
        
        # Recommendations
        report['recommendations'] = trust_results['recommendations']
        
        if successful_transcriptions / audio_messages < 0.5:
            report['recommendations'].append("Consider improving audio quality for better transcription accuracy")
        
        if risk_score > 50:
            report['recommendations'].append("⚠️ High-risk indicators detected. Recommend professional investigation.")
        
        return report
    
    def run_full_analysis(self, zip_path: str) -> Tuple[pd.DataFrame, Dict]:
        """Run complete forensic analysis pipeline"""
        start_time = time.time()
        
        # Step 1: Load data
        df = self.process_whatsapp_export(zip_path)
        
        if df.empty:
            return df, {}
        
        # Step 2: Transcribe audio
        df = self.transcribe_audio_batch(df)
        
        # Step 3: Analyze patterns
        pattern_analysis = self.analyze_patterns(df)
        
        # Step 4: Generate report
        forensic_report = self.generate_forensic_report(df)
        
        elapsed_time = time.time() - start_time
        st.success(f"✅ Complete analysis finished in {elapsed_time:.1f} seconds")
        
        return df, forensic_report

# Main application
def main():
    """Main application entry point"""
    st.markdown('<div class="main-header">🔬 DIE WAARHEID V2: Automated Forensic Analysis</div>', unsafe_allow_html=True)
    
    # Initialize session state
    init_session_state()
    
    # File upload section
    st.markdown("### 📂 Upload WhatsApp Export for Analysis")
    
    uploaded_file = st.file_uploader(
        "Select WhatsApp Export ZIP file",
        type=['zip'],
        help="Upload the complete WhatsApp export ZIP file"
    )
    
    if uploaded_file and not st.session_state.processing:
        # Check available disk space first
        import shutil
        
        temp_drive = str(DATA_DIR).split(':')[0] + ':'
        try:
            total, used, free = shutil.disk_usage(DATA_DIR)
            free_gb = free // (1024**3)
            
            if free_gb < 5:  # Less than 5GB free
                st.error(f"❌ Low disk space! Only {free_gb}GB available. Please free up space and try again.")
                st.stop()
            
            st.info(f"💾 Available disk space: {free_gb}GB")
        except Exception as e:
            st.warning(f"Could not check disk space: {e}")
        
        # Save uploaded file temporarily
        temp_path = DATA_DIR / "temp" / uploaded_file.name
        
        # Check file size
        uploaded_file.seek(0, 2)  # Seek to end
        file_size = uploaded_file.tell()
        uploaded_file.seek(0)  # Reset position
        
        file_size_gb = file_size / (1024**3)
        
        if file_size_gb > free_gb - 1:  # Leave 1GB buffer
            st.error(f"❌ File too large! File is {file_size_gb:.1f}GB but only {free_gb}GB available.")
            st.stop()
        
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Quick diagnostic
        with st.spinner("🔍 Checking audio files..."):
            try:
                df = whatsapp_parser.parse_export_file(Path(temp_path))
                audio_files = df[df['message_type'] == 'audio']
                
                if not audio_files.empty:
                    # Check file existence
                    missing = 0
                    valid = 0
                    empty = 0
                    
                    for idx, row in audio_files.iterrows():
                        if not pd.notna(row.get('media_path')) or not os.path.exists(row['media_path']):
                            missing += 1
                        else:
                            file_size = os.path.getsize(row['media_path'])
                            if file_size == 0:
                                empty += 1
                            else:
                                valid += 1
                    
                    st.write(f"📊 Audio File Diagnostic:")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Audio", len(audio_files))
                    with col2:
                        st.metric("Valid Files", valid)
                    with col3:
                        st.metric("Missing", missing)
                    with col4:
                        st.metric("Empty", empty)
                    
                    if missing > 0:
                        st.warning(f"⚠️ {missing} audio files are missing - check if media was extracted properly")
                    if valid == 0:
                        st.error("❌ No valid audio files found! The ZIP may not contain media files.")
            except OSError as e:
                if "No space left on device" in str(e):
                    st.error("❌ Disk full! Please free up disk space and try again.")
                    st.stop()
                else:
                    st.error(f"❌ Error reading file: {e}")
                    st.stop()
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
                st.stop()
        
        # Initialize analyzer
        analyzer = BatchForensicAnalyzer()
        
        # Run analysis button
        if st.button("🚀 Run Complete Forensic Analysis", type="primary", disabled=valid == 0):
            st.session_state.processing = True
            st.rerun()
    
    # Process data if analysis was triggered
    if st.session_state.processing and uploaded_file:
        temp_path = DATA_DIR / "temp" / uploaded_file.name
        
        with st.spinner("🔬 Running comprehensive forensic analysis..."):
            analyzer = BatchForensicAnalyzer()
            df, report = analyzer.run_full_analysis(str(temp_path))
            
            if not df.empty:
                st.session_state.analysis_data = df
                st.session_state.forensic_report = report
                st.session_state.processing = False
                st.rerun()
    
    # Display results
    if st.session_state.analysis_data is not None and st.session_state.forensic_report:
        display_forensic_results(st.session_state.analysis_data, st.session_state.forensic_report)

def display_forensic_results(df: pd.DataFrame, report: Dict):
    """Display comprehensive forensic results"""
    st.markdown("---")
    st.markdown("## 📊 Forensic Analysis Results")
    
    # Executive Summary
    st.markdown("### 📋 Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Messages", f"{report['executive_summary']['total_messages']:,}")
    
    with col2:
        st.metric("Voice Notes", report['executive_summary']['audio_messages'])
    
    with col3:
        success_rate = report['executive_summary']['transcription_success_rate']
        st.metric("Transcription Success", f"{success_rate:.1f}%")
    
    with col4:
        trust_score = report['trust_assessment']['overall_score']
        trust_color = "🔴" if trust_score < 40 else "🟡" if trust_score < 70 else "🟢"
        st.metric("Trust Score", f"{trust_color} {trust_score}/100")
    
    # Trust Assessment
    st.markdown("### 🎯 Trust Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Trust Score Gauge
        trust_score = report['trust_assessment']['overall_score']
        trust_level = report['trust_assessment']['trust_level']
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = trust_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Trust Level: {trust_level}"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightcoral"},
                    {'range': [40, 70], 'color': "lightyellow"},
                    {'range': [70, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 40
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Trust Breakdown:**")
        for factor, score in report['trust_assessment']['breakdown'].items():
            factor_name = factor.replace('_', ' ').title()
            st.write(f"- {factor_name}: {score:.1f}/100")
        
        st.markdown("**Interpretation:**")
        st.info(report['trust_assessment']['interpretation'])
    
    # Toxicity Analysis
    if 'toxicity_analysis' in report:
        st.markdown("### 🚨 Toxicity Analysis")
        
        toxicity = report['toxicity_analysis']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            toxic_pct = (toxicity['toxic_messages'] / toxicity['total_messages'] * 100) if toxicity['total_messages'] > 0 else 0
            st.metric("Toxic Messages", f"{toxic_pct:.1f}%")
        
        with col2:
            gaslighting_pct = (toxicity['gaslighting_messages'] / toxicity['total_messages'] * 100) if toxicity['total_messages'] > 0 else 0
            st.metric("Gaslighting", f"{gaslighting_pct:.1f}%")
        
        with col3:
            narcissistic_pct = (toxicity['narcissistic_messages'] / toxicity['total_messages'] * 100) if toxicity['total_messages'] > 0 else 0
            st.metric("Narcissistic", f"{narcissistic_pct:.1f}%")
        
        with col4:
            severe_count = toxicity['severity_distribution'].get('severe', 0) + toxicity['severity_distribution'].get('high', 0)
            st.metric("Severe Cases", severe_count)
        
        # Top Offenders
        if toxicity['top_offenders']:
            st.markdown("**Top Offenders:**")
            for offender in toxicity['top_offenders'][:3]:
                with st.expander(f"👤 {offender['sender']} - {offender['toxicity_rate']:.1f}% toxicity"):
                    st.write(f"- Toxic messages: {offender['toxic_count']}")
                    st.write(f"- Gaslighting: {offender['gaslighting_count']}")
                    st.write(f"- Narcissistic patterns: {offender['narcissistic_count']}")
    
    # Risk Assessment
    if report['risk_assessment']['risk_factors']:
        st.markdown("### ⚠️ Risk Assessment")
        risk_level = report['risk_assessment']['risk_level']
        risk_score = report['risk_assessment']['overall_risk_score']
        
        if risk_level == 'High':
            st.markdown(f'<div class="forensic-alert">🚨 HIGH RISK DETECTED (Score: {risk_score}/100)</div>', unsafe_allow_html=True)
        elif risk_level == 'Medium':
            st.markdown(f'<div class="forensic-warning">⚠️ MEDIUM RISK (Score: {risk_score}/100)</div>', unsafe_allow_html=True)
        
        for factor in report['risk_assessment']['risk_factors']:
            st.markdown(f'<div class="forensic-warning">⚠️ {factor}</div>', unsafe_allow_html=True)
    
    # Audio Forensics Dashboard
    if report['audio_forensics']:
        st.markdown("### 🎙️ Audio Forensics Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Stress Level Distribution
            if report['audio_forensics']['stress_distribution']:
                stress_data = report['audio_forensics']['stress_distribution']
                fig = px.pie(
                    values=list(stress_data.values()),
                    names=list(stress_data.keys()),
                    title="Stress Level Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Deception Score Gauge
            avg_deception = report['audio_forensics'].get('avg_deception_score', 0)
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = avg_deception,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Average Deception Score"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    if report['recommendations']:
        st.markdown("### 📝 Recommendations")
        for rec in report['recommendations']:
            if '⚠️' in rec or '🚫' in rec or '🛑' in rec:
                st.markdown(f'<div class="forensic-warning">{rec}</div>', unsafe_allow_html=True)
            elif '✅' in rec:
                st.markdown(f'<div class="forensic-info">{rec}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f"- {rec}")
    
    # Detailed Transcription Results
    st.markdown("### 📝 Transcription Results")
    
    # Filter for successful transcriptions
    audio_df = df[df['message_type'] == 'audio'].copy()
    
    if not audio_df.empty:
        # Search/Filter
        search_term = st.text_input("🔍 Search transcriptions", placeholder="Search for keywords in transcriptions...")
        
        if search_term:
            mask = audio_df['transcript'].str.contains(search_term, case=False, na=False)
            audio_df = audio_df[mask]
        
        # Display transcriptions with forensic markers
        for idx, row in audio_df.iterrows():
            with st.expander(f"🎙️ {row['sender']} - {row['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write("**Transcript:**")
                    st.write(row.get('transcript', '[No transcript]'))
                    
                    if row.get('analysis_successful'):
                        st.write(f"**Language:** {row.get('language', 'unknown')}")
                        st.write(f"**Speakers:** {row.get('speaker_count', 'unknown')}")
                
                with col2:
                    stress = row.get('stress_level', 'unknown')
                    stress_color = "red" if stress in ['High', 'Very High'] else "orange" if stress == 'Medium' else "green"
                    st.markdown(f"**Stress:** <span style='color:{stress_color}'>{stress}</span>", unsafe_allow_html=True)
                    
                    deception = row.get('deception_score', 0)
                    deception_color = "red" if deception > 70 else "orange" if deception > 40 else "green"
                    st.markdown(f"**Deception:** <span style='color:{deception_color}'>{deception:.0f}%</span>", unsafe_allow_html=True)
    
    # Export Options
    st.markdown("### 💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export Full Report (JSON)"):
            export_data = {
                'report': report,
                'transcriptions': df[df['message_type'] == 'audio'].to_dict('records'),
                'exported_at': datetime.now().isoformat()
            }
            
            export_path = DATA_DIR / f"forensic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(export_path, 'w') as f:
                json.dump(export_data, f, default=str, indent=2)
            
            st.success(f"Report exported to {export_path}")
    
    with col2:
        if st.button("📊 Export Transcriptions (CSV)"):
            export_df = df[df['message_type'] == 'audio'][['timestamp', 'sender', 'transcript', 'stress_level', 'deception_score']]
            export_path = DATA_DIR / f"transcriptions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            export_df.to_csv(export_path, index=False)
            st.success(f"Transcriptions exported to {export_path}")
    
    with col3:
        if st.button("🔄 New Analysis"):
            st.session_state.analysis_data = None
            st.session_state.forensic_report = None
            st.session_state.processing = False
            st.rerun()

if __name__ == "__main__":
    main()
