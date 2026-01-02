"""
Timeline Visualization for Die Waarheid
Interactive visualizations and exports for reconstructed timelines
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class TimelineVisualizer:
    """
    Create interactive visualizations for reconstructed timelines
    """

    def __init__(self):
        """Initialize timeline visualizer"""
        self.color_map = {
            'audio': '#3498db',
            'image': '#2ecc71',
            'video': '#9b59b6',
            'document': '#e74c3c',
            'unknown': '#95a5a6'
        }
        
        self.confidence_colors = {
            'certain': '#27ae60',
            'high': '#2ecc71',
            'medium': '#f1c40f',
            'low': '#e67e22',
            'uncertain': '#e74c3c'
        }

    def create_plotly_timeline(self, entries: List[Dict]) -> Any:
        """
        Create interactive Plotly timeline

        Args:
            entries: List of timeline entry dictionaries

        Returns:
            Plotly figure
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.error("Plotly not available")
            return None

        if not entries:
            return None

        # Prepare data
        timestamps = []
        labels = []
        colors = []
        hover_texts = []
        sizes = []

        for entry in entries:
            ts = datetime.fromisoformat(entry['timestamp']) if isinstance(entry['timestamp'], str) else entry['timestamp']
            timestamps.append(ts)
            
            label = entry.get('speaker_id', 'Unknown')
            labels.append(label)
            
            colors.append(self.color_map.get(entry.get('content_type', 'unknown'), '#95a5a6'))
            
            hover = f"<b>{entry.get('entry_id', '')}</b><br>"
            hover += f"Time: {ts.strftime('%Y-%m-%d %H:%M:%S')}<br>"
            hover += f"Type: {entry.get('content_type', 'unknown')}<br>"
            hover += f"Speaker: {entry.get('speaker_id', 'Unknown')}<br>"
            hover += f"Confidence: {entry.get('timestamp_confidence', 'unknown')}<br>"
            if entry.get('transcription'):
                hover += f"Text: {entry['transcription'][:50]}..."
            hover_texts.append(hover)
            
            # Size based on duration or default
            duration = entry.get('duration', 10)
            sizes.append(max(10, min(30, (duration or 10) / 2)))

        # Create figure
        fig = go.Figure()

        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=labels,
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            text=hover_texts,
            hoverinfo='text',
            name='Timeline Events'
        ))

        # Update layout
        fig.update_layout(
            title='Reconstructed Timeline',
            xaxis_title='Date/Time',
            yaxis_title='Speaker',
            height=500,
            showlegend=True,
            hovermode='closest'
        )

        return fig

    def create_gantt_chart(self, entries: List[Dict]) -> Any:
        """
        Create Gantt-style chart for timeline

        Args:
            entries: List of timeline entry dictionaries

        Returns:
            Plotly figure
        """
        try:
            import plotly.express as px
            import pandas as pd
        except ImportError:
            logger.error("Plotly or pandas not available")
            return None

        if not entries:
            return None

        # Prepare data for Gantt chart
        gantt_data = []
        for entry in entries:
            ts = datetime.fromisoformat(entry['timestamp']) if isinstance(entry['timestamp'], str) else entry['timestamp']
            duration = entry.get('duration', 5)  # Default 5 seconds
            
            gantt_data.append({
                'Task': entry.get('speaker_id', 'Unknown'),
                'Start': ts,
                'Finish': ts + timedelta(seconds=duration or 5),
                'Type': entry.get('content_type', 'unknown'),
                'Confidence': entry.get('timestamp_confidence', 'unknown')
            })

        df = pd.DataFrame(gantt_data)

        fig = px.timeline(
            df,
            x_start='Start',
            x_end='Finish',
            y='Task',
            color='Type',
            title='Timeline Gantt Chart',
            color_discrete_map=self.color_map
        )

        fig.update_layout(height=400)
        return fig

    def create_confidence_chart(self, entries: List[Dict]) -> Any:
        """
        Create chart showing confidence distribution

        Args:
            entries: List of timeline entry dictionaries

        Returns:
            Plotly figure
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        if not entries:
            return None

        # Count confidence levels
        confidence_counts = {}
        for entry in entries:
            conf = entry.get('timestamp_confidence', 'unknown')
            confidence_counts[conf] = confidence_counts.get(conf, 0) + 1

        labels = list(confidence_counts.keys())
        values = list(confidence_counts.values())
        colors = [self.confidence_colors.get(l, '#95a5a6') for l in labels]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            hole=0.4
        )])

        fig.update_layout(
            title='Timestamp Confidence Distribution',
            height=350
        )

        return fig

    def create_hourly_distribution(self, entries: List[Dict]) -> Any:
        """
        Create hourly distribution chart

        Args:
            entries: List of timeline entry dictionaries

        Returns:
            Plotly figure
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        if not entries:
            return None

        # Count by hour
        hourly_counts = {h: 0 for h in range(24)}
        for entry in entries:
            ts = datetime.fromisoformat(entry['timestamp']) if isinstance(entry['timestamp'], str) else entry['timestamp']
            hourly_counts[ts.hour] += 1

        hours = list(range(24))
        counts = [hourly_counts[h] for h in hours]

        fig = go.Figure(data=[go.Bar(
            x=[f"{h:02d}:00" for h in hours],
            y=counts,
            marker_color='#3498db'
        )])

        fig.update_layout(
            title='Activity by Hour of Day',
            xaxis_title='Hour',
            yaxis_title='Number of Events',
            height=300
        )

        return fig

    def create_daily_heatmap(self, entries: List[Dict]) -> Any:
        """
        Create daily activity heatmap

        Args:
            entries: List of timeline entry dictionaries

        Returns:
            Plotly figure
        """
        try:
            import plotly.graph_objects as go
            import numpy as np
        except ImportError:
            return None

        if not entries:
            return None

        # Get date range
        timestamps = [
            datetime.fromisoformat(e['timestamp']) if isinstance(e['timestamp'], str) else e['timestamp']
            for e in entries
        ]
        min_date = min(timestamps).date()
        max_date = max(timestamps).date()

        # Create matrix (days x hours)
        days = (max_date - min_date).days + 1
        matrix = np.zeros((days, 24))

        for ts in timestamps:
            day_idx = (ts.date() - min_date).days
            matrix[day_idx][ts.hour] += 1

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=[f"{h:02d}:00" for h in range(24)],
            y=[(min_date + timedelta(days=d)).strftime('%Y-%m-%d') for d in range(days)],
            colorscale='Blues'
        ))

        fig.update_layout(
            title='Activity Heatmap (Days x Hours)',
            xaxis_title='Hour of Day',
            yaxis_title='Date',
            height=max(300, days * 20)
        )

        return fig

    def create_speaker_timeline(self, entries: List[Dict]) -> Any:
        """
        Create speaker-specific timeline

        Args:
            entries: List of timeline entry dictionaries

        Returns:
            Plotly figure
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            return None

        if not entries:
            return None

        # Group by speaker
        speakers = {}
        for entry in entries:
            speaker = entry.get('speaker_id', 'Unknown')
            if speaker not in speakers:
                speakers[speaker] = []
            speakers[speaker].append(entry)

        # Create subplots
        fig = make_subplots(
            rows=len(speakers),
            cols=1,
            shared_xaxes=True,
            subplot_titles=list(speakers.keys()),
            vertical_spacing=0.05
        )

        colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f1c40f']

        for idx, (speaker, speaker_entries) in enumerate(speakers.items()):
            timestamps = [
                datetime.fromisoformat(e['timestamp']) if isinstance(e['timestamp'], str) else e['timestamp']
                for e in speaker_entries
            ]
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=[1] * len(timestamps),
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=colors[idx % len(colors)]
                    ),
                    name=speaker
                ),
                row=idx + 1,
                col=1
            )

        fig.update_layout(
            height=150 * len(speakers) + 100,
            title='Timeline by Speaker',
            showlegend=True
        )

        return fig

    def export_html_report(
        self,
        entries: List[Dict],
        output_path: Path,
        title: str = "Timeline Report"
    ) -> bool:
        """
        Export complete HTML report with all visualizations

        Args:
            entries: List of timeline entry dictionaries
            output_path: Output file path
            title: Report title

        Returns:
            True if successful
        """
        try:
            # Create visualizations
            timeline_fig = self.create_plotly_timeline(entries)
            confidence_fig = self.create_confidence_chart(entries)
            hourly_fig = self.create_hourly_distribution(entries)

            # Build HTML
            html_parts = [
                f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; }}
        .chart {{ margin: 20px 0; border: 1px solid #ddd; padding: 10px; }}
        .stats {{ background: #f8f9fa; padding: 15px; border-radius: 5px; }}
        .entry {{ border-left: 3px solid #3498db; padding-left: 10px; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>Total Entries: {len(entries)}</p>
"""
            ]

            # Add charts
            if timeline_fig:
                html_parts.append('<div class="chart">')
                html_parts.append(timeline_fig.to_html(full_html=False, include_plotlyjs=False))
                html_parts.append('</div>')

            if confidence_fig:
                html_parts.append('<div class="chart">')
                html_parts.append(confidence_fig.to_html(full_html=False, include_plotlyjs=False))
                html_parts.append('</div>')

            if hourly_fig:
                html_parts.append('<div class="chart">')
                html_parts.append(hourly_fig.to_html(full_html=False, include_plotlyjs=False))
                html_parts.append('</div>')

            # Add timeline entries
            html_parts.append('<h2>Timeline Entries</h2>')
            for entry in entries:
                ts = entry.get('timestamp', 'Unknown')
                html_parts.append(f"""
<div class="entry">
    <strong>{ts}</strong> [{entry.get('timestamp_confidence', 'unknown')}]<br>
    Type: {entry.get('content_type', 'unknown')} | 
    Speaker: {entry.get('speaker_id', 'Unknown')}<br>
    {f"<em>{entry.get('transcription', '')}</em>" if entry.get('transcription') else ''}
</div>
""")

            html_parts.append('</body></html>')

            # Write file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(html_parts))

            logger.info(f"Exported HTML report to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting HTML report: {e}")
            return False


class TimelineAnalyzer:
    """
    Analyze timeline for patterns and insights
    """

    def analyze_communication_patterns(self, entries: List[Dict]) -> Dict[str, Any]:
        """
        Analyze communication patterns in timeline

        Args:
            entries: List of timeline entry dictionaries

        Returns:
            Pattern analysis results
        """
        if not entries:
            return {}

        # Sort by timestamp
        sorted_entries = sorted(
            entries,
            key=lambda x: datetime.fromisoformat(x['timestamp']) if isinstance(x['timestamp'], str) else x['timestamp']
        )

        # Analyze patterns
        patterns = {
            'response_times': [],
            'active_hours': {},
            'speaker_patterns': {},
            'conversation_gaps': []
        }

        prev_entry = None
        for entry in sorted_entries:
            ts = datetime.fromisoformat(entry['timestamp']) if isinstance(entry['timestamp'], str) else entry['timestamp']
            speaker = entry.get('speaker_id', 'Unknown')

            # Track active hours
            hour = ts.hour
            patterns['active_hours'][hour] = patterns['active_hours'].get(hour, 0) + 1

            # Track speaker patterns
            if speaker not in patterns['speaker_patterns']:
                patterns['speaker_patterns'][speaker] = {
                    'count': 0,
                    'total_duration': 0,
                    'first_seen': ts,
                    'last_seen': ts
                }
            
            patterns['speaker_patterns'][speaker]['count'] += 1
            patterns['speaker_patterns'][speaker]['last_seen'] = ts
            if entry.get('duration'):
                patterns['speaker_patterns'][speaker]['total_duration'] += entry['duration']

            # Calculate response times between different speakers
            if prev_entry:
                prev_ts = datetime.fromisoformat(prev_entry['timestamp']) if isinstance(prev_entry['timestamp'], str) else prev_entry['timestamp']
                prev_speaker = prev_entry.get('speaker_id', 'Unknown')
                
                gap = (ts - prev_ts).total_seconds()
                
                if prev_speaker != speaker:
                    patterns['response_times'].append({
                        'from': prev_speaker,
                        'to': speaker,
                        'gap_seconds': gap
                    })
                
                # Flag large gaps
                if gap > 3600:  # More than 1 hour
                    patterns['conversation_gaps'].append({
                        'start': prev_ts.isoformat(),
                        'end': ts.isoformat(),
                        'gap_hours': gap / 3600
                    })

            prev_entry = entry

        # Calculate summary statistics
        if patterns['response_times']:
            avg_response = sum(r['gap_seconds'] for r in patterns['response_times']) / len(patterns['response_times'])
            patterns['avg_response_time_seconds'] = avg_response

        # Find peak hours
        if patterns['active_hours']:
            peak_hour = max(patterns['active_hours'], key=patterns['active_hours'].get)
            patterns['peak_activity_hour'] = peak_hour

        return patterns

    def detect_anomalies(self, entries: List[Dict]) -> List[Dict]:
        """
        Detect anomalies in timeline

        Args:
            entries: List of timeline entry dictionaries

        Returns:
            List of detected anomalies
        """
        anomalies = []

        if len(entries) < 2:
            return anomalies

        sorted_entries = sorted(
            entries,
            key=lambda x: datetime.fromisoformat(x['timestamp']) if isinstance(x['timestamp'], str) else x['timestamp']
        )

        # Calculate average gap
        gaps = []
        for i in range(1, len(sorted_entries)):
            ts1 = datetime.fromisoformat(sorted_entries[i-1]['timestamp']) if isinstance(sorted_entries[i-1]['timestamp'], str) else sorted_entries[i-1]['timestamp']
            ts2 = datetime.fromisoformat(sorted_entries[i]['timestamp']) if isinstance(sorted_entries[i]['timestamp'], str) else sorted_entries[i]['timestamp']
            gaps.append((ts2 - ts1).total_seconds())

        if not gaps:
            return anomalies

        avg_gap = sum(gaps) / len(gaps)
        
        # Detect unusual gaps
        for i, gap in enumerate(gaps):
            if gap > avg_gap * 5:
                anomalies.append({
                    'type': 'large_gap',
                    'entry_before': sorted_entries[i]['entry_id'],
                    'entry_after': sorted_entries[i+1]['entry_id'],
                    'gap_hours': gap / 3600,
                    'description': f'Unusually large gap of {gap/3600:.1f} hours'
                })

        # Detect timestamp clustering (potential batch upload)
        timestamp_minutes = {}
        for entry in sorted_entries:
            ts = datetime.fromisoformat(entry['timestamp']) if isinstance(entry['timestamp'], str) else entry['timestamp']
            minute_key = ts.strftime('%Y-%m-%d %H:%M')
            if minute_key not in timestamp_minutes:
                timestamp_minutes[minute_key] = []
            timestamp_minutes[minute_key].append(entry['entry_id'])

        for minute, entry_ids in timestamp_minutes.items():
            if len(entry_ids) > 3:
                anomalies.append({
                    'type': 'timestamp_cluster',
                    'minute': minute,
                    'count': len(entry_ids),
                    'entries': entry_ids,
                    'description': f'{len(entry_ids)} entries at same minute - possible batch upload'
                })

        # Detect low confidence entries
        low_confidence = [e for e in entries if e.get('timestamp_confidence') in ['low', 'uncertain']]
        if low_confidence:
            anomalies.append({
                'type': 'low_confidence_timestamps',
                'count': len(low_confidence),
                'entries': [e['entry_id'] for e in low_confidence],
                'description': f'{len(low_confidence)} entries with low timestamp confidence'
            })

        return anomalies


if __name__ == "__main__":
    visualizer = TimelineVisualizer()
    analyzer = TimelineAnalyzer()
    print("Timeline Visualizer and Analyzer initialized")
