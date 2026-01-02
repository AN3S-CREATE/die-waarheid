"""
Visualization Engine for Die Waarheid
Interactive Plotly charts and dashboards for forensic analysis
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import (
    PLOTLY_THEME,
    STRESS_HEATMAP_COLORSCALE,
    SPEAKER_COLORS
)

logger = logging.getLogger(__name__)


class VisualizationEngine:
    """
    Creates interactive visualizations for forensic analysis
    Generates Plotly charts for stress analysis, timelines, and patterns
    """

    def __init__(self):
        self.theme = PLOTLY_THEME
        self.color_scale = STRESS_HEATMAP_COLORSCALE

    def create_stress_timeline(self, timeline_data: pd.DataFrame) -> go.Figure:
        """
        Create stress level timeline chart

        Args:
            timeline_data: DataFrame with timeline entries including stress_level

        Returns:
            Plotly figure object
        """
        try:
            audio_data = timeline_data[timeline_data['Message_Type'] == 'audio'].copy()

            if audio_data.empty:
                logger.warning("No audio data for stress timeline")
                return go.Figure().add_annotation(text="No audio data available")

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=audio_data['Recorded_At'],
                y=audio_data['Stress_Level'],
                mode='lines+markers',
                name='Stress Level',
                line=dict(color='#FF6B6B', width=2),
                marker=dict(size=8),
                hovertemplate='<b>%{x}</b><br>Stress: %{y:.2f}<extra></extra>'
            ))

            fig.add_hline(y=50, line_dash="dash", line_color="orange", 
                         annotation_text="High Stress Threshold")

            fig.update_layout(
                title='Stress Level Timeline',
                xaxis_title='Time',
                yaxis_title='Stress Level (0-100)',
                template=self.theme,
                hovermode='x unified',
                height=500
            )

            logger.info("Created stress timeline chart")
            return fig

        except Exception as e:
            logger.error(f"Error creating stress timeline: {str(e)}")
            return go.Figure().add_annotation(text=f"Error: {str(e)}")

    def create_speaker_distribution(self, timeline_data: pd.DataFrame) -> go.Figure:
        """
        Create speaker message distribution pie chart

        Args:
            timeline_data: DataFrame with timeline entries

        Returns:
            Plotly figure object
        """
        try:
            speaker_counts = timeline_data['Sender'].value_counts()

            fig = go.Figure(data=[go.Pie(
                labels=speaker_counts.index,
                values=speaker_counts.values,
                marker=dict(colors=[SPEAKER_COLORS.get(f"SPEAKER_{i:02d}", f"#{i*100:06x}") 
                                    for i in range(len(speaker_counts))]),
                hovertemplate='<b>%{label}</b><br>Messages: %{value}<extra></extra>'
            )])

            fig.update_layout(
                title='Message Distribution by Speaker',
                template=self.theme,
                height=500
            )

            logger.info("Created speaker distribution chart")
            return fig

        except Exception as e:
            logger.error(f"Error creating speaker distribution: {str(e)}")
            return go.Figure().add_annotation(text=f"Error: {str(e)}")

    def create_message_timeline(self, timeline_data: pd.DataFrame) -> go.Figure:
        """
        Create message frequency timeline

        Args:
            timeline_data: DataFrame with timeline entries

        Returns:
            Plotly figure object
        """
        try:
            timeline_data['Date'] = pd.to_datetime(timeline_data['Recorded_At']).dt.date
            daily_counts = timeline_data.groupby('Date').size()

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=daily_counts.index,
                y=daily_counts.values,
                name='Messages',
                marker=dict(color='#4ECDC4'),
                hovertemplate='<b>%{x}</b><br>Messages: %{y}<extra></extra>'
            ))

            fig.update_layout(
                title='Daily Message Frequency',
                xaxis_title='Date',
                yaxis_title='Number of Messages',
                template=self.theme,
                hovermode='x unified',
                height=500
            )

            logger.info("Created message timeline chart")
            return fig

        except Exception as e:
            logger.error(f"Error creating message timeline: {str(e)}")
            return go.Figure().add_annotation(text=f"Error: {str(e)}")

    def create_bio_signal_heatmap(self, timeline_data: pd.DataFrame) -> go.Figure:
        """
        Create bio-signal heatmap for audio messages

        Args:
            timeline_data: DataFrame with timeline entries

        Returns:
            Plotly figure object
        """
        try:
            audio_data = timeline_data[timeline_data['Message_Type'] == 'audio'].copy()

            if audio_data.empty:
                logger.warning("No audio data for bio-signal heatmap")
                return go.Figure().add_annotation(text="No audio data available")

            heatmap_data = audio_data[['Pitch_Volatility', 'Silence_Ratio', 'Intensity_Max']].T

            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=range(len(audio_data)),
                y=['Pitch Volatility', 'Silence Ratio', 'Intensity (Max)'],
                colorscale=self.color_scale,
                hovertemplate='Signal: %{y}<br>Message %{x}<br>Value: %{z:.2f}<extra></extra>'
            ))

            fig.update_layout(
                title='Bio-Signal Analysis Heatmap',
                xaxis_title='Message Index',
                yaxis_title='Bio-Signal',
                template=self.theme,
                height=400
            )

            logger.info("Created bio-signal heatmap")
            return fig

        except Exception as e:
            logger.error(f"Error creating bio-signal heatmap: {str(e)}")
            return go.Figure().add_annotation(text=f"Error: {str(e)}")

    def create_forensic_flags_chart(self, timeline_data: pd.DataFrame) -> go.Figure:
        """
        Create forensic flags distribution chart

        Args:
            timeline_data: DataFrame with timeline entries

        Returns:
            Plotly figure object
        """
        try:
            all_flags = []
            for flags_str in timeline_data['Forensic_Flag']:
                if flags_str and isinstance(flags_str, str):
                    flags = flags_str.split('|')
                    all_flags.extend(flags)

            if not all_flags:
                logger.warning("No forensic flags found")
                return go.Figure().add_annotation(text="No forensic flags detected")

            flag_counts = pd.Series(all_flags).value_counts()

            fig = go.Figure(data=[go.Bar(
                x=flag_counts.index,
                y=flag_counts.values,
                marker=dict(color='#FF6B6B'),
                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
            )])

            fig.update_layout(
                title='Forensic Flags Distribution',
                xaxis_title='Flag Type',
                yaxis_title='Frequency',
                template=self.theme,
                height=500,
                xaxis_tickangle=-45
            )

            logger.info("Created forensic flags chart")
            return fig

        except Exception as e:
            logger.error(f"Error creating forensic flags chart: {str(e)}")
            return go.Figure().add_annotation(text=f"Error: {str(e)}")

    def create_emotion_distribution(self, timeline_data: pd.DataFrame) -> go.Figure:
        """
        Create emotion distribution chart

        Args:
            timeline_data: DataFrame with timeline entries

        Returns:
            Plotly figure object
        """
        try:
            emotions = timeline_data['Tone_Emotion'].value_counts()

            if emotions.empty:
                logger.warning("No emotion data found")
                return go.Figure().add_annotation(text="No emotion data available")

            emotion_colors = {
                'Calm': '#2ECC71',
                'Concerned': '#F39C12',
                'Stressed': '#E74C3C',
                'Highly Stressed': '#C0392B',
                'Anxious': '#E67E22',
                'Unknown': '#95A5A6'
            }

            colors = [emotion_colors.get(emotion, '#95A5A6') for emotion in emotions.index]

            fig = go.Figure(data=[go.Bar(
                x=emotions.index,
                y=emotions.values,
                marker=dict(color=colors),
                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
            )])

            fig.update_layout(
                title='Emotion Distribution',
                xaxis_title='Emotion',
                yaxis_title='Frequency',
                template=self.theme,
                height=500
            )

            logger.info("Created emotion distribution chart")
            return fig

        except Exception as e:
            logger.error(f"Error creating emotion distribution: {str(e)}")
            return go.Figure().add_annotation(text=f"Error: {str(e)}")

    def create_multi_signal_chart(self, timeline_data: pd.DataFrame) -> go.Figure:
        """
        Create multi-signal comparison chart

        Args:
            timeline_data: DataFrame with timeline entries

        Returns:
            Plotly figure object
        """
        try:
            audio_data = timeline_data[timeline_data['Message_Type'] == 'audio'].copy()

            if audio_data.empty:
                logger.warning("No audio data for multi-signal chart")
                return go.Figure().add_annotation(text="No audio data available")

            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Pitch Volatility', 'Silence Ratio', 'Intensity (Max)'),
                shared_xaxes=True,
                vertical_spacing=0.1
            )

            fig.add_trace(
                go.Scatter(x=range(len(audio_data)), y=audio_data['Pitch_Volatility'],
                          name='Pitch Volatility', line=dict(color='#FF6B6B')),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(x=range(len(audio_data)), y=audio_data['Silence_Ratio'],
                          name='Silence Ratio', line=dict(color='#4ECDC4')),
                row=2, col=1
            )

            fig.add_trace(
                go.Scatter(x=range(len(audio_data)), y=audio_data['Intensity_Max'],
                          name='Intensity (Max)', line=dict(color='#FFE66D')),
                row=3, col=1
            )

            fig.update_yaxes(title_text="Volatility", row=1, col=1)
            fig.update_yaxes(title_text="Ratio", row=2, col=1)
            fig.update_yaxes(title_text="Intensity", row=3, col=1)
            fig.update_xaxes(title_text="Message Index", row=3, col=1)

            fig.update_layout(
                title='Multi-Signal Bio-Analysis',
                template=self.theme,
                height=800,
                hovermode='x unified'
            )

            logger.info("Created multi-signal chart")
            return fig

        except Exception as e:
            logger.error(f"Error creating multi-signal chart: {str(e)}")
            return go.Figure().add_annotation(text=f"Error: {str(e)}")

    def create_conversation_flow(self, timeline_data: pd.DataFrame) -> go.Figure:
        """
        Create conversation flow diagram

        Args:
            timeline_data: DataFrame with timeline entries

        Returns:
            Plotly figure object
        """
        try:
            senders = timeline_data['Sender'].unique()
            sender_to_idx = {sender: i for i, sender in enumerate(senders)}

            y_positions = timeline_data['Sender'].map(sender_to_idx)

            fig = go.Figure()

            for sender in senders:
                sender_data = timeline_data[timeline_data['Sender'] == sender]
                fig.add_trace(go.Scatter(
                    x=sender_data['Recorded_At'],
                    y=[sender_to_idx[sender]] * len(sender_data),
                    mode='markers',
                    name=sender,
                    marker=dict(size=10),
                    hovertemplate=f'<b>{sender}</b><br>%{{x}}<extra></extra>'
                ))

            fig.update_layout(
                title='Conversation Flow',
                xaxis_title='Time',
                yaxis_title='Participant',
                yaxis=dict(tickvals=list(range(len(senders))), ticktext=list(senders)),
                template=self.theme,
                height=400,
                hovermode='closest'
            )

            logger.info("Created conversation flow chart")
            return fig

        except Exception as e:
            logger.error(f"Error creating conversation flow: {str(e)}")
            return go.Figure().add_annotation(text=f"Error: {str(e)}")

    def create_dashboard(self, timeline_data: pd.DataFrame) -> go.Figure:
        """
        Create comprehensive dashboard with multiple visualizations

        Args:
            timeline_data: DataFrame with timeline entries

        Returns:
            Plotly figure object with subplots
        """
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Stress Timeline', 'Speaker Distribution', 
                               'Daily Messages', 'Emotion Distribution'),
                specs=[[{'type': 'scatter'}, {'type': 'pie'}],
                       [{'type': 'bar'}, {'type': 'bar'}]]
            )

            audio_data = timeline_data[timeline_data['Message_Type'] == 'audio']
            if not audio_data.empty:
                fig.add_trace(
                    go.Scatter(x=audio_data['Recorded_At'], y=audio_data['Stress_Level'],
                              name='Stress', line=dict(color='#FF6B6B')),
                    row=1, col=1
                )

            speaker_counts = timeline_data['Sender'].value_counts()
            fig.add_trace(
                go.Pie(labels=speaker_counts.index, values=speaker_counts.values,
                       name='Speakers'),
                row=1, col=2
            )

            timeline_data['Date'] = pd.to_datetime(timeline_data['Recorded_At']).dt.date
            daily_counts = timeline_data.groupby('Date').size()
            fig.add_trace(
                go.Bar(x=daily_counts.index, y=daily_counts.values, name='Messages',
                      marker=dict(color='#4ECDC4')),
                row=2, col=1
            )

            emotions = timeline_data['Tone_Emotion'].value_counts()
            if not emotions.empty:
                fig.add_trace(
                    go.Bar(x=emotions.index, y=emotions.values, name='Emotions',
                          marker=dict(color='#FFE66D')),
                    row=2, col=2
                )

            fig.update_layout(
                title='Forensic Analysis Dashboard',
                template=self.theme,
                height=800,
                showlegend=False
            )

            logger.info("Created comprehensive dashboard")
            return fig

        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            return go.Figure().add_annotation(text=f"Error: {str(e)}")


if __name__ == "__main__":
    engine = VisualizationEngine()
    print("Visualization Engine initialized")
