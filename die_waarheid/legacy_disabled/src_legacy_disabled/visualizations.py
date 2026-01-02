"""
Forensic Visualizations Module

Complete set of 21 visualizations for Die Waarheid forensic dashboard
"""

import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import base64
import io

try:
    import streamlit as st
except ImportError:
    st = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    px = None
    go = None
    make_subplots = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import librosa
    import librosa.display
except ImportError:
    librosa = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

try:
    from wordcloud import WordCloud
except ImportError:
    WordCloud = None

try:
    import networkx as nx
except ImportError:
    nx = None

from config import THEME_COLORS, REPORTS_DIR, EXPORTS_DIR

logger = logging.getLogger(__name__)

class ForensicVisualizer:
    """Main class for all forensic visualizations"""
    
    def __init__(self):
        self.colors = THEME_COLORS
        if plt:
            self.setup_matplotlib()
    
    def setup_matplotlib(self):
        """Setup matplotlib for non-interactive backend"""
        if not plt:
            return
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    # ==================== PHASE 1: MVP VISUALIZATIONS ====================
    
    def render_message_volume_chart(self, df: pd.DataFrame):
        """Phase 1: Message Volume Chart"""
        st.markdown("### 📊 Message Volume Over Time")
        
        if 'timestamp' not in df.columns:
            st.warning("No timestamp data available")
            return
        
        # Prepare data
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily_counts = df.groupby(['date', 'sender']).size().reset_index(name='count')
        
        # Create interactive bar chart
        fig = px.bar(
            daily_counts,
            x='date',
            y='count',
            color='sender',
            title="Daily Message Volume by Sender",
            labels={'count': 'Messages', 'date': 'Date'},
            color_discrete_sequence=list(self.colors.values())
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Messages",
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_sentiment_timeline(self, df: pd.DataFrame):
        """Phase 1: Sentiment Polarity Over Time"""
        st.markdown("### 📈 Sentiment Analysis Timeline")
        
        if 'timestamp' not in df.columns:
            st.warning("No timestamp data available")
            return
        
        # Calculate sentiment (placeholder - would use NLP in production)
        df['sentiment'] = self._calculate_sentiment(df)
        
        # Create line chart
        fig = px.line(
            df,
            x='timestamp',
            y='sentiment',
            title="Sentiment Polarity Over Time",
            labels={'sentiment': 'Sentiment Score (-1 to 1)', 'timestamp': 'Time'},
            color_discrete_sequence=[self.colors['primary']]
        )
        
        # Add reference lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
        fig.add_hline(y=0.5, line_dash="dot", line_color="green", annotation_text="Positive")
        fig.add_hline(y=-0.5, line_dash="dot", line_color="red", annotation_text="Negative")
        
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    
    def render_basic_timeline(self, df: pd.DataFrame):
        """Phase 1: Basic Timeline View"""
        st.markdown("### 📅 Communication Timeline")
        
        if 'timestamp' not in df.columns:
            st.warning("No timestamp data available")
            return
        
        # Create scatter plot
        fig = px.scatter(
            df,
            x='timestamp',
            y='sender',
            color='message_type',
            title="Communication Timeline by Message Type",
            labels={'timestamp': 'Time', 'sender': 'Sender'},
            color_discrete_map={
                'text': self.colors['primary'],
                'audio': self.colors['secondary'],
                'image': self.colors['success'],
                'video': self.colors['warning']
            }
        )
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Sender",
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_progress_bar(self, current: int, total: int, stage: str = "Processing"):
        """Phase 1: Simple Progress Bar"""
        progress = current / total if total > 0 else 0
        
        # Progress bar
        st.progress(progress)
        
        # Status text
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Stage:** {stage}")
        with col2:
            st.write(f"**Progress:** {current}/{total} ({progress:.1%})")
        with col3:
            if progress > 0:
                eta = (total - current) / (current / 10) if current > 0 else 0  # Rough ETA
                st.write(f"**ETA:** ~{eta:.0f} min")
    
    # ==================== PHASE 2: CORE FORENSICS ====================
    
    def render_stress_heatmap(self, df: pd.DataFrame):
        """Phase 2: Stress Heat Map (Advanced Plotly Visualization)"""
        st.markdown("### 🌡️ Forensic Stress Heat Map")
        
        if 'timestamp' not in df.columns or 'stress_index' not in df.columns:
            st.warning("Stress analysis data not available. Run audio analysis first.")
            return
        
        # Prepare data
        df_plot = df[df['stress_index'].notna()].copy()
        
        if df_plot.empty:
            st.warning("No stress data available")
            return
        
        # Create scatter plot with multiple dimensions
        fig = px.scatter(
            df_plot,
            x='timestamp',
            y='stress_index',
            size='max_loudness' if 'max_loudness' in df_plot.columns else [10] * len(df_plot),
            color='sender',
            hover_data=[
                'transcript' if 'transcript' in df_plot.columns else 'message',
                'silence_ratio' if 'silence_ratio' in df_plot.columns else None,
                'speaker_count' if 'speaker_count' in df_plot.columns else None
            ],
            title="Forensic Stress Analysis Timeline",
            labels={
                'stress_index': 'Stress Index (0-100)',
                'timestamp': 'Timeline',
                'max_loudness': 'Loudness/Intensity'
            },
            color_discrete_sequence=list(self.colors.values())
        )
        
        # Add stress zones
        fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, layer="below", line_width=0)
        fig.add_hrect(y0=40, y1=70, fillcolor="yellow", opacity=0.1, layer="below", line_width=0)
        fig.add_hrect(y0=0, y1=40, fillcolor="green", opacity=0.1, layer="below", line_width=0)
        
        # Update layout
        fig.update_layout(
            xaxis_title="Timeline (zoomable)",
            yaxis_title="Stress Index",
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Stress statistics
        st.markdown("#### 📊 Stress Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_stress = df_plot['stress_index'].mean()
            st.metric("Average Stress", f"{avg_stress:.1f}")
        with col2:
            high_stress = (df_plot['stress_index'] > 70).sum()
            st.metric("High Stress Events", high_stress)
        with col3:
            stress_spikes = self._detect_stress_spikes(df_plot)
            st.metric("Stress Spikes", stress_spikes)
    
    def render_contradiction_explorer(self, df: pd.DataFrame):
        """Phase 2: Contradiction Explorer (AI-Generated List)"""
        st.markdown("### 🚨 Contradiction Explorer")
        
        # Detect contradictions
        contradictions = self._detect_contradictions(df)
        
        if contradictions.empty:
            st.success("No contradictions detected!")
            return
        
        # Filter controls
        st.markdown("#### Filters")
        col1, col2 = st.columns(2)
        with col1:
            severity_filter = st.multiselect(
                "Severity",
                ['Minor', 'Moderate', 'Severe'],
                default=['Minor', 'Moderate', 'Severe']
            )
        with col2:
            claim_type_filter = st.multiselect(
                "Claim Type",
                contradictions['claim_type'].unique().tolist(),
                default=contradictions['claim_type'].unique().tolist()
            )
        
        # Filter contradictions
        filtered = contradictions[
            contradictions['severity'].isin(severity_filter) &
            contradictions['claim_type'].isin(claim_type_filter)
        ]
        
        # Display contradictions
        for _, row in filtered.iterrows():
            severity_color = {
                'Minor': 'green',
                'Moderate': 'orange',
                'Severe': 'red'
            }.get(row['severity'], 'gray')
            
            st.markdown(f"""
            <div style='padding: 1rem; margin: 0.5rem 0; border-left: 5px solid {severity_color}; 
                        background-color: rgba({severity_color}, 0.1); border-radius: 5px;'>
                <strong>{row['timestamp']}</strong> - {row['claim_type']}<br>
                <strong>Severity:</strong> {row['severity']}<br>
                {row['description']}<br>
                <em>Transcript: "{row['transcript'][:100]}..."</em>
            </div>
            """, unsafe_allow_html=True)
        
        # Export options
        st.markdown("#### Export Options")
        if st.button("Export Contradictions to CSV"):
            csv_path = EXPORTS_DIR / f"contradictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filtered.to_csv(csv_path, index=False)
            st.success(f"Exported to {csv_path}")
    
    def render_mobitable_timeline(self, df: pd.DataFrame):
        """Phase 2: MobiTable Timeline (Markdown Format)"""
        st.markdown("### 📋 MobiTable Timeline")
        
        from src.mobitab_builder import mobitab_builder
        
        # Build MobiTable
        mobitable = mobitab_builder.build_mobitable(df)
        
        if mobitable.empty:
            st.warning("No data available for MobiTable")
            return
        
        # Display options
        view_mode = st.radio("View Mode", ["Interactive Table", "Markdown Preview", "Raw Data"])
        
        if view_mode == "Interactive Table":
            # Show as interactive dataframe
            st.dataframe(mobitable, use_container_width=True, height=400)
        elif view_mode == "Markdown Preview":
            # Show markdown preview
            st.markdown("#### Markdown Preview:")
            markdown_text = self._mobitable_to_markdown(mobitable.head(20))
            st.code(markdown_text, language='markdown')
        else:
            # Show raw data
            st.write(mobitable)
        
        # Export options
        st.markdown("#### Export MobiTable")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Export as Markdown"):
                path = mobitab_builder.export_to_markdown(mobitable)
                st.success(f"Exported to {path}")
        with col2:
            if st.button("Export as CSV"):
                path = mobitab_builder.export_to_csv(mobitable)
                st.success(f"Exported to {path}")
        with col3:
            if st.button("Export as Excel"):
                path = mobitab_builder.export_to_excel(mobitable)
                st.success(f"Exported to {path}")
    
    def render_audio_player(self, df: pd.DataFrame):
        """Phase 2: Audio Player Widget"""
        st.markdown("### 🎵 Audio Player")
        
        # Filter audio files
        audio_files = df[df['message_type'] == 'audio']
        
        if audio_files.empty:
            st.warning("No audio files found")
            return
        
        # File selector
        file_options = [f"{row['timestamp']} - {row['sender']}" for _, row in audio_files.iterrows()]
        selected = st.selectbox("Select audio file:", file_options)
        
        if selected:
            # Get selected file
            idx = file_options.index(selected)
            row = audio_files.iloc[idx]
            
            # Display audio player
            if 'media_path' in row and pd.notna(row['media_path']):
                try:
                    with open(row['media_path'], 'rb') as f:
                        audio_bytes = f.read()
                    st.audio(audio_bytes, format='audio/opus')
                except Exception as e:
                    st.error(f"Could not load audio file: {e}")
            
            # Display transcript
            if 'transcript' in row and pd.notna(row['transcript']):
                st.markdown("#### Transcript:")
                st.write(row['transcript'])
            
            # Display audio analysis
            if 'stress_index' in row and pd.notna(row['stress_index']):
                st.markdown("#### Audio Analysis:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Stress Index", f"{row['stress_index']:.1f}")
                with col2:
                    st.metric("Duration", f"{row.get('duration', 0):.1f}s")
                with col3:
                    st.metric("Speakers", int(row.get('speaker_count', 1)))
    
    # ==================== PHASE 3: ADVANCED FORENSICS ====================
    
    def render_toxicity_dashboard(self, df: pd.DataFrame):
        """Phase 3: Toxicity Dashboard"""
        st.markdown("### ☠️ Toxicity Dashboard")
        
        # Analyze toxicity patterns
        toxicity_metrics = self._analyze_toxicity(df)
        
        # Multi-panel layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Gaslighting Counter
            st.markdown("#### Gaslighting Detection")
            fig_gaslight = px.bar(
                x=['Dismissive', 'Reality Denial', 'Victim Playing', 'Other'],
                y=[toxicity_metrics['gaslighting'][k] for k in ['dismissive', 'denial', 'victim', 'other']],
                title="Gaslighting Tactics Frequency",
                color_discrete_sequence=[self.colors['danger']]
            )
            st.plotly_chart(fig_gaslight, use_container_width=True)
        
        with col2:
            # Narcissistic Pattern Score
            st.markdown("#### Narcissistic Patterns")
            fig_narc = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = toxicity_metrics['narcissism_score'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Narcissism Score"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self.colors['warning']},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            st.plotly_chart(fig_narc, use_container_width=True)
        
        # Emotional Abuse Indicators
        st.markdown("#### Emotional Abuse Indicators")
        fig_abuse = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Intensity Spikes", "Communication Blackouts", 
                          "Sentiment Swings", "Trust Score"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add plots (simplified for example)
        fig_abuse.add_trace(
            go.Scatter(x=df['timestamp'], y=df.get('max_loudness', [0]*len(df)), 
                      name="Intensity", line=dict(color=self.colors['danger'])),
            row=1, col=1
        )
        
        fig_abuse.add_trace(
            go.Scatter(x=df['timestamp'], y=[1]*len(df), 
                      name="Communication", line=dict(color=self.colors['primary'])),
            row=1, col=2
        )
        
        fig_abuse.add_trace(
            go.Scatter(x=df['timestamp'], y=df.get('sentiment', [0]*len(df)), 
                      name="Sentiment", line=dict(color=self.colors['warning'])),
            row=2, col=1
        )
        
        fig_abuse.add_trace(
            go.Scatter(x=df['timestamp'], y=[80]*len(df), 
                      name="Trust", line=dict(color=self.colors['success'])),
            row=2, col=2
        )
        
        fig_abuse.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig_abuse, use_container_width=True)
    
    def render_voice_biometric(self, df: pd.DataFrame):
        """Phase 3: Voice Biometric Fingerprint"""
        st.markdown("### 🎙️ Voice Biometric Fingerprint")
        
        # Select audio file
        audio_files = df[df['message_type'] == 'audio']
        if audio_files.empty:
            st.warning("No audio files for biometric analysis")
            return
        
        file_options = [f"{row['timestamp']} - {row['sender']}" for _, row in audio_files.iterrows()]
        selected = st.selectbox("Select audio file for analysis:", file_options)
        
        if selected:
            idx = file_options.index(selected)
            row = audio_files.iloc[idx]
            
            if 'media_path' in row and pd.notna(row['media_path']):
                # Generate spectrogram
                fig_spectro = self._generate_spectrogram(row['media_path'])
                st.pyplot(fig_spectro)
                
                # Voice metrics
                st.markdown("#### Voice Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Pitch Volatility", f"{row.get('pitch_volatility', 0):.2f}")
                with col2:
                    st.metric("Silence Ratio", f"{row.get('silence_ratio', 0):.2%}")
                with col3:
                    st.metric("Energy", f"{row.get('max_loudness', 0):.3f}")
                with col4:
                    st.metric("Formants", "N/A")  # Would calculate in production
    
    def render_speaker_timeline(self, df: pd.DataFrame):
        """Phase 3: Speaker Timeline View (Parallel Lanes)"""
        st.markdown("### 🗣️ Speaker Timeline View")
        
        if 'speaker_id' not in df.columns:
            st.warning("Speaker diarization data not available")
            return
        
        # Get unique speakers
        speakers = df['speaker_id'].unique()
        
        # Create parallel timeline
        fig = go.Figure()
        
        for i, speaker in enumerate(speakers):
            speaker_data = df[df['speaker_id'] == speaker]
            
            # Color by stress if available
            if 'stress_index' in speaker_data.columns:
                colors = []
                for _, row in speaker_data.iterrows():
                    stress = row.get('stress_index', 0)
                    if stress > 70:
                        colors.append(self.colors['danger'])
                    elif stress > 40:
                        colors.append(self.colors['warning'])
                    else:
                        colors.append(self.colors['success'])
            else:
                colors = [self.colors['primary']] * len(speaker_data)
            
            fig.add_trace(go.Scatter(
                x=speaker_data['timestamp'],
                y=[i] * len(speaker_data),
                mode='markers',
                name=f'Speaker {speaker}',
                marker=dict(color=colors, size=10),
                text=speaker_data.get('message', ''),
                hovertemplate='<b>%{fullData.name}</b><br>%{x}<br>%{text}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Speaker Timeline - Parallel Communication Lanes",
            xaxis_title="Time",
            yaxis_title="Speakers",
            yaxis=dict(tickmode='array', tickvals=list(range(len(speakers))), 
                      ticktext=[f'Speaker {s}' for s in speakers]),
            height=200 * len(speakers)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_live_processing_status(self):
        """Phase 3: Live Processing Status"""
        st.markdown("### ⚡ Live Processing Status")
        
        # Get processing status from session state
        status = st.session_state.get('processing_status', {})
        
        # Overall progress
        progress = status.get('progress', 0)
        stage = status.get('stage', 'Idle')
        current_file = status.get('current_file', '')
        
        # Progress bar with stages
        st.progress(progress / 100)
        
        # Stage indicators
        stages = ['Downloading', 'Extracting', 'Transcribing', 'Analyzing', 'Complete']
        current_stage_idx = min(len(stages) - 1, int(progress / 20))
        
        cols = st.columns(len(stages))
        for i, (col, stage_name) in enumerate(zip(cols, stages)):
            if i <= current_stage_idx:
                col.markdown(f"✅ {stage_name}")
            else:
                col.markdown(f"⏳ {stage_name}")
        
        # Current file info
        if current_file:
            st.info(f"Currently processing: {current_file}")
        
        # Estimated time remaining
        if progress > 0 and progress < 100:
            eta = (100 - progress) / (progress / 10) if progress > 0 else 0
            st.write(f"Estimated time remaining: ~{eta:.0f} minutes")
        
        # Error log (if any)
        errors = status.get('errors', [])
        if errors:
            st.markdown("#### ⚠️ Errors:")
            for error in errors[-5:]:  # Show last 5 errors
                st.error(error)
    
    # ==================== HELPER METHODS ====================
    
    def _calculate_sentiment(self, df: pd.DataFrame) -> pd.Series:
        """Calculate sentiment scores (placeholder implementation)"""
        # In production, would use proper NLP
        sentiments = []
        for message in df.get('message', []):
            if isinstance(message, str):
                # Simple rule-based sentiment
                if any(word in message.lower() for word in ['love', 'happy', 'great']):
                    sentiments.append(0.5)
                elif any(word in message.lower() for word in ['sad', 'angry', 'bad']):
                    sentiments.append(-0.5)
                else:
                    sentiments.append(0.0)
            else:
                sentiments.append(0.0)
        return pd.Series(sentiments)
    
    def _detect_stress_spikes(self, df: pd.DataFrame, threshold: float = 70) -> int:
        """Detect sudden stress spikes"""
        if 'stress_index' not in df.columns:
            return 0
        
        stress_values = df['stress_index'].values
        spikes = 0
        
        for i in range(1, len(stress_values)):
            if stress_values[i] > threshold and stress_values[i-1] < threshold:
                spikes += 1
        
        return spikes
    
    def _detect_contradictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect contradictions in the data"""
        contradictions = []
        
        for _, row in df.iterrows():
            if row.get('message_type') != 'audio':
                continue
            
            transcript = str(row.get('transcript', '')).lower()
            speaker_count = row.get('speaker_count', 1)
            stress_idx = row.get('stress_index', 0)
            
            # Location contradictions
            if any(word in transcript for word in ['alone', 'by myself']) and speaker_count > 1:
                contradictions.append({
                    'timestamp': row['timestamp'],
                    'claim_type': 'Location vs Audio',
                    'description': f"Claims to be alone but {speaker_count} speakers detected",
                    'severity': 'Severe',
                    'transcript': transcript
                })
            
            # Emotional contradictions
            if any(word in transcript for word in ['calm', 'fine']) and stress_idx > 70:
                contradictions.append({
                    'timestamp': row['timestamp'],
                    'claim_type': 'Emotional State vs Bio-signals',
                    'description': f"Claims to be calm but stress index is {stress_idx:.0f}",
                    'severity': 'Moderate',
                    'transcript': transcript
                })
        
        return pd.DataFrame(contradictions)
    
    def _mobitable_to_markdown(self, df: pd.DataFrame) -> str:
        """Convert MobiTable to markdown format"""
        if df.empty:
            return "No data available"
        
        # Header
        markdown = "| " + " | ".join(df.columns) + " |\n"
        markdown += "|" + "|".join(["---"] * len(df.columns)) + "|\n"
        
        # Rows
        for _, row in df.iterrows():
            row_data = []
            for col in df.columns:
                value = str(row[col]) if pd.notna(row[col]) else ''
                value = value.replace('|', '\\|')[:100]  # Escape pipes and limit length
                row_data.append(value)
            markdown += "| " + " | ".join(row_data) + " |\n"
        
        return markdown
    
    def _analyze_toxicity(self, df: pd.DataFrame) -> Dict:
        """Analyze toxicity patterns (placeholder)"""
        return {
            'gaslighting': {
                'dismissive': 5,
                'denial': 3,
                'victim': 2,
                'other': 1
            },
            'narcissism_score': 65
        }
    
    def _generate_spectrogram(self, audio_path: str) -> plt.Figure:
        """Generate spectrogram for audio file"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Generate spectrogram
            D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
            img = librosa.display.specshow(D, x_axis='time', y_axis='hz', ax=ax)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            
            ax.set_title('Voice Biometric Fingerprint - Spectrogram')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error generating spectrogram: {e}")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            return fig

# Global instance for easy importing
forensic_visualizer = ForensicVisualizer()
