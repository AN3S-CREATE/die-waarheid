"""
MobiTable Builder

Generates forensic timeline tables in Markdown format for export and analysis
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import re

from config import MOBITABLES_DIR, TIMESTAMP_FORMAT

logger = logging.getLogger(__name__)

class MobiTableBuilder:
    """Builds forensic timeline tables (MobiTables) for analysis"""
    
    def __init__(self):
        self.columns = [
            'Index',
            'Msg ID',
            'Sender',
            'Recorded At',
            'Speaker',
            'Transcript',
            'Tone/Emotion',
            'Stress Level',
            'Media Type',
            'Duration',
            'Speaker Count',
            'Contradiction',
            'Evidence Link'
        ]
    
    def build_mobitable(self, df: pd.DataFrame, include_audio_analysis: bool = True) -> pd.DataFrame:
        """Build a comprehensive MobiTable from conversation data"""
        if df.empty:
            return pd.DataFrame(columns=self.columns)
        
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        
        # Initialize MobiTable
        mobitable_data = []
        
        for idx, row in df_sorted.iterrows():
            mobi_row = self._convert_row_to_mobi(row, idx + 1, include_audio_analysis)
            mobitable_data.append(mobi_row)
        
        mobitable_df = pd.DataFrame(mobitable_data, columns=self.columns)
        
        # Add forensic annotations
        mobitable_df = self._add_forensic_annotations(mobitable_df, df_sorted)
        
        return mobitable_df
    
    def _convert_row_to_mobi(self, row: pd.Series, index: int, include_audio: bool) -> Dict:
        """Convert a DataFrame row to MobiTable format"""
        # Basic fields
        mobi_row = {
            'Index': index,
            'Msg ID': self._generate_message_id(row),
            'Sender': row.get('sender', 'Unknown'),
            'Recorded At': row['timestamp'].strftime(TIMESTAMP_FORMAT),
            'Speaker': self._identify_speaker(row),
            'Transcript': self._get_transcript(row),
            'Tone/Emotion': self._detect_tone_emotion(row),
            'Stress Level': self._get_stress_level(row) if include_audio else 'N/A',
            'Media Type': row.get('message_type', 'text'),
            'Duration': self._get_duration(row) if include_audio else 'N/A',
            'Speaker Count': self._get_speaker_count(row) if include_audio else 'N/A',
            'Contradiction': self._detect_contradiction(row),
            'Evidence Link': self._get_evidence_link(row)
        }
        
        return mobi_row
    
    def _generate_message_id(self, row: pd.Series) -> str:
        """Generate unique message ID"""
        timestamp = row['timestamp'].strftime('%Y%m%d_%H%M%S')
        sender = re.sub(r'[^a-zA-Z0-9]', '_', str(row.get('sender', 'unknown')))[:10]
        return f"MSG_{timestamp}_{sender}"
    
    def _identify_speaker(self, row: pd.Series) -> str:
        """Identify speaker with diarization data if available"""
        if 'speaker_id' in row and pd.notna(row['speaker_id']):
            return f"Speaker {row['speaker_id']}"
        return row.get('sender', 'Unknown')
    
    def _get_transcript(self, row: pd.Series) -> str:
        """Get transcript or message text"""
        if row.get('message_type') == 'audio' and 'transcript' in row:
            return row['transcript'] or '[Transcription failed]'
        return row.get('message', '')
    
    def _detect_tone_emotion(self, row: pd.Series) -> str:
        """Detect tone and emotion from message and audio analysis"""
        # Check emotional state from audio analysis
        if 'emotional_state' in row and pd.notna(row['emotional_state']):
            return row['emotional_state']
        
        # Simple text-based tone detection
        message = str(row.get('message', '')).lower()
        
        if any(word in message for word in ['!', '!!!', 'angry', 'mad', 'furious']):
            return 'angry'
        elif any(word in message for word in ['sad', 'cry', 'upset', 'hurt']):
            return 'sad'
        elif any(word in message for word in ['love', 'happy', 'glad', 'excited']):
            return 'happy'
        elif any(word in message for word in ['sorry', 'apologize', 'my bad']):
            return 'apologetic'
        elif any(word in message for word in ['?', '??', 'what', 'how', 'why']):
            return 'questioning'
        elif len(message) == 0:
            return 'neutral'
        else:
            return 'neutral'
    
    def _get_stress_level(self, row: pd.Series) -> str:
        """Get stress level from bio-signal analysis"""
        if 'stress_index' in row and pd.notna(row['stress_index']):
            stress = row['stress_index']
            if stress >= 80:
                return 'Very High'
            elif stress >= 60:
                return 'High'
            elif stress >= 40:
                return 'Medium'
            elif stress >= 20:
                return 'Low'
            else:
                return 'Very Low'
        return 'N/A'
    
    def _get_duration(self, row: pd.Series) -> str:
        """Get audio duration"""
        if 'duration' in row and pd.notna(row['duration']):
            return f"{row['duration']:.1f}s"
        return 'N/A'
    
    def _get_speaker_count(self, row: pd.Series) -> str:
        """Get speaker count from diarization"""
        if 'speaker_count' in row and pd.notna(row['speaker_count']):
            return str(int(row['speaker_count']))
        return 'N/A'
    
    def _detect_contradiction(self, row: pd.Series) -> str:
        """Detect contradictions in the message"""
        message = str(row.get('message', '')).lower()
        transcript = str(row.get('transcript', '')).lower()
        
        # Check for location contradictions
        if any(word in message or word in transcript for word in ['alone', 'by myself', 'solo']):
            if row.get('speaker_count', 1) > 1:
                return 'Location: Claims alone, multiple speakers detected'
        
        # Check for emotional contradictions
        if any(word in message or word in transcript for word in ['calm', 'fine', 'okay']):
            if row.get('stress_index', 0) > 70:
                return f"Emotional: Claims calm, stress index {row.get('stress_index', 0):.0f}"
        
        # Check for activity contradictions
        if any(word in message or word in transcript for word in ['busy', 'working', 'in meeting']):
            # This would need external verification
            pass
        
        return 'None'
    
    def _get_evidence_link(self, row: pd.Series) -> str:
        """Get link to evidence file"""
        if 'media_path' in row and pd.notna(row['media_path']):
            return str(row['media_path'])
        return 'N/A'
    
    def _add_forensic_annotations(self, mobitable_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
        """Add forensic annotations and highlights"""
        # Add warning flags for high stress
        if 'Stress Level' in mobitable_df.columns:
            high_stress_mask = mobitable_df['Stress Level'].isin(['High', 'Very High'])
            mobitable_df.loc[high_stress_mask, 'Index'] = '⚠️ ' + mobitable_df.loc[high_stress_mask, 'Index'].astype(str)
        
        # Add contradiction flags
        contradiction_mask = mobitable_df['Contradiction'] != 'None'
        mobitable_df.loc[contradiction_mask, 'Index'] = '🚨 ' + mobitable_df.loc[contradiction_mask, 'Index'].astype(str)
        
        return mobitable_df
    
    def export_to_markdown(self, mobitable_df: pd.DataFrame, filename: str = None) -> Path:
        """Export MobiTable to Markdown format"""
        if filename is None:
            filename = f"mobitable_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        output_path = MOBITABLES_DIR / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("# 🔬 DIE WAARHEID MOBITABLE\n\n")
            f.write(f"Generated: {datetime.now().strftime(TIMESTAMP_FORMAT)}\n\n")
            
            # Table header
            f.write("| " + " | ".join(self.columns) + " |\n")
            f.write("| " + " | ".join(['---'] * len(self.columns)) + " |\n")
            
            # Table rows
            for _, row in mobitable_df.iterrows():
                # Clean and format each cell
                row_data = []
                for col in self.columns:
                    value = str(row[col]) if pd.notna(row[col]) else ''
                    # Escape pipe characters in Markdown
                    value = value.replace('|', '\\|')
                    # Truncate long content
                    if col in ['Transcript'] and len(value) > 100:
                        value = value[:97] + '...'
                    row_data.append(value)
                
                f.write("| " + " | ".join(row_data) + " |\n")
            
            # Footer
            f.write("\n---\n")
            f.write("*Generated by Die Waarheid Forensic Analysis System*\n")
        
        logger.info(f"MobiTable exported to {output_path}")
        return output_path
    
    def export_to_csv(self, mobitable_df: pd.DataFrame, filename: str = None) -> Path:
        """Export MobiTable to CSV format"""
        if filename is None:
            filename = f"mobitable_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        output_path = MOBITABLES_DIR / filename
        mobitable_df.to_csv(output_path, index=False)
        
        logger.info(f"MobiTable exported to {output_path}")
        return output_path
    
    def export_to_excel(self, mobitable_df: pd.DataFrame, filename: str = None) -> Path:
        """Export MobiTable to Excel format with formatting"""
        if filename is None:
            filename = f"mobitable_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        output_path = MOBITABLES_DIR / filename
        
        # Create Excel writer with formatting
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            mobitable_df.to_excel(writer, sheet_name='MobiTable', index=False)
            
            # Get the workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets['MobiTable']
            
            # Format headers
            for col_num, column in enumerate(mobitable_df.columns, 1):
                cell = worksheet.cell(row=1, column=col_num)
                cell.font = cell.font.copy(bold=True)
            
            # Auto-adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        logger.info(f"MobiTable exported to {output_path}")
        return output_path
    
    def create_summary_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a summary statistics table"""
        if df.empty:
            return pd.DataFrame(columns=['Metric', 'Value'])
        
        # Build mobitable first to get proper columns
        mobitable_df = self.build_mobitable(df)
        
        # Check if 'Media Type' column exists
        audio_count = 0
        if 'Media Type' in mobitable_df.columns:
            audio_count = len(mobitable_df[mobitable_df['Media Type'] == 'audio'])
        
        # Check if 'Stress Level' column exists
        high_stress_count = 0
        if 'Stress Level' in mobitable_df.columns:
            high_stress_count = len(mobitable_df[mobitable_df['Stress Level'].isin(['High', 'Very High'])])
        
        # Check if 'Contradiction' column exists
        contradiction_count = 0
        if 'Contradiction' in mobitable_df.columns:
            contradiction_count = len(mobitable_df[mobitable_df['Contradiction'] != 'None'])
        
        # Check if 'Speaker' column exists
        unique_speakers = 0
        if 'Speaker' in mobitable_df.columns:
            unique_speakers = mobitable_df['Speaker'].nunique()
        
        summary_data = {
            'Metric': [
                'Total Messages',
                'Audio Messages',
                'High Stress Messages',
                'Contradictions Detected',
                'Unique Speakers',
                'Time Span',
                'Average Message Length'
            ],
            'Value': [
                len(mobitable_df),
                audio_count,
                high_stress_count,
                contradiction_count,
                unique_speakers,
                self._calculate_time_span(mobitable_df),
                self._calculate_avg_message_length(mobitable_df)
            ]
        }
        
        return pd.DataFrame(summary_data)
    
    def _calculate_time_span(self, mobitable_df: pd.DataFrame) -> str:
        """Calculate time span of conversation"""
        if len(mobitable_df) < 2:
            return 'N/A'
        
        try:
            first_date = pd.to_datetime(mobitable_df['Recorded At'].iloc[0])
            last_date = pd.to_datetime(mobitable_df['Recorded At'].iloc[-1])
            delta = last_date - first_date
            
            if delta.days > 0:
                return f"{delta.days} days"
            else:
                hours = delta.seconds // 3600
                return f"{hours} hours"
        except:
            return 'N/A'
    
    def _calculate_avg_message_length(self, mobitable_df: pd.DataFrame) -> str:
        """Calculate average message length"""
        transcripts = mobitable_df['Transcript'].dropna()
        if len(transcripts) == 0:
            return 'N/A'
        
        avg_length = transcripts.str.len().mean()
        return f"{avg_length:.0f} characters"

# Global instance for easy importing
mobitab_builder = MobiTableBuilder()
