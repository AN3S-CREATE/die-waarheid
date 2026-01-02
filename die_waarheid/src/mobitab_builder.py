"""
Mobitab Builder for Die Waarheid
Generates forensic timelines with integrated audio and chat analysis
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

import pandas as pd

from config import (
    MOBITAB_COLUMNS,
    MOBITAB_FILENAME,
    MOBITAB_DIR
)

logger = logging.getLogger(__name__)


class MobitabBuilder:
    """
    Builds forensic timelines (mobitabs) combining chat and audio analysis
    Generates structured markdown reports with integrated findings
    """

    def __init__(self):
        self.timeline_data = []
        self.metadata = {}
        self.forensic_findings = {}

    def add_chat_message(
        self,
        msg_id: str,
        timestamp: datetime,
        sender: str,
        text: str,
        message_type: str = "text"
    ) -> None:
        """
        Add chat message to timeline

        Args:
            msg_id: Unique message ID
            timestamp: Message timestamp
            sender: Sender name
            text: Message text
            message_type: Type of message (text, image, audio, video, etc.)
        """
        entry = {
            'Index': len(self.timeline_data) + 1,
            'Msg_ID': msg_id,
            'Sender': sender,
            'Recorded_At': timestamp,
            'Speaker_Count': 1,
            'Transcript': text,
            'Tone_Emotion': '',
            'Pitch_Volatility': 0.0,
            'Silence_Ratio': 0.0,
            'Intensity_Max': 0.0,
            'Forensic_Flag': '',
            'Message_Type': message_type
        }

        self.timeline_data.append(entry)
        logger.debug(f"Added chat message: {msg_id} from {sender}")

    def add_audio_analysis(
        self,
        msg_id: str,
        timestamp: datetime,
        sender: str,
        transcript: str,
        forensics_result: Dict,
        speaker_count: int = 1
    ) -> None:
        """
        Add audio analysis result to timeline

        Args:
            msg_id: Unique message ID
            timestamp: Message timestamp
            sender: Sender name
            transcript: Transcribed text
            forensics_result: Dictionary with forensic analysis results
            speaker_count: Number of speakers detected
        """
        stress_level = forensics_result.get('stress_level', 0.0)
        pitch_volatility = forensics_result.get('pitch_volatility', 0.0)
        silence_ratio = forensics_result.get('silence_ratio', 0.0)
        intensity_max = forensics_result.get('intensity', {}).get('max', 0.0)

        forensic_flags = []

        if stress_level > 50:
            forensic_flags.append("HIGH_STRESS")

        if silence_ratio > 0.4:
            forensic_flags.append("HIGH_COGNITIVE_LOAD")

        if pitch_volatility > 50:
            forensic_flags.append("PITCH_VOLATILITY")

        entry = {
            'Index': len(self.timeline_data) + 1,
            'Msg_ID': msg_id,
            'Sender': sender,
            'Recorded_At': timestamp,
            'Speaker_Count': speaker_count,
            'Transcript': transcript,
            'Tone_Emotion': self._infer_emotion(stress_level, pitch_volatility),
            'Pitch_Volatility': round(pitch_volatility, 2),
            'Silence_Ratio': round(silence_ratio, 2),
            'Intensity_Max': round(intensity_max, 2),
            'Forensic_Flag': '|'.join(forensic_flags) if forensic_flags else '',
            'Message_Type': 'audio',
            'Stress_Level': round(stress_level, 2),
            'Duration': forensics_result.get('duration', 0.0)
        }

        self.timeline_data.append(entry)
        logger.debug(f"Added audio analysis: {msg_id} from {sender} (stress: {stress_level:.2f})")

    def _infer_emotion(self, stress_level: float, pitch_volatility: float) -> str:
        """
        Infer emotion from bio-signals

        Args:
            stress_level: Stress level (0-100)
            pitch_volatility: Pitch volatility (0-100)

        Returns:
            Emotion label
        """
        if stress_level > 70:
            return "Highly Stressed"
        elif stress_level > 50:
            return "Stressed"
        elif pitch_volatility > 60:
            return "Anxious"
        elif stress_level > 30:
            return "Concerned"
        else:
            return "Calm"

    def add_ai_analysis(
        self,
        msg_id: str,
        ai_result: Dict
    ) -> None:
        """
        Add AI analysis results to existing timeline entry

        Args:
            msg_id: Message ID to update
            ai_result: Dictionary with AI analysis results
        """
        for entry in self.timeline_data:
            if entry['Msg_ID'] == msg_id:
                entry['Tone_Emotion'] = ai_result.get('emotion', entry.get('Tone_Emotion', ''))
                entry['AI_Toxicity'] = ai_result.get('toxicity_score', 0.0)
                entry['AI_Gaslighting'] = ai_result.get('gaslighting_detected', False)
                entry['AI_Contradiction'] = ai_result.get('contradiction_detected', False)
                logger.debug(f"Updated AI analysis for {msg_id}")
                break

    def set_metadata(
        self,
        case_id: str,
        chat_name: str,
        participants: List[str],
        date_range: Tuple[datetime, datetime]
    ) -> None:
        """
        Set timeline metadata

        Args:
            case_id: Case identifier
            chat_name: Name of chat/conversation
            participants: List of participant names
            date_range: Tuple of (start_date, end_date)
        """
        self.metadata = {
            'case_id': case_id,
            'chat_name': chat_name,
            'participants': participants,
            'start_date': date_range[0],
            'end_date': date_range[1],
            'generated_at': datetime.now(),
            'total_entries': len(self.timeline_data)
        }

        logger.info(f"Set metadata for case {case_id}: {len(self.timeline_data)} entries")

    def get_timeline_dataframe(self) -> pd.DataFrame:
        """
        Get timeline as pandas DataFrame

        Returns:
            DataFrame with timeline data
        """
        if not self.timeline_data:
            return pd.DataFrame()

        df = pd.DataFrame(self.timeline_data)

        if 'Recorded_At' in df.columns:
            df['Recorded_At'] = pd.to_datetime(df['Recorded_At'])
            df = df.sort_values('Recorded_At')
            df['Index'] = range(1, len(df) + 1)

        return df

    def generate_markdown_report(self) -> str:
        """
        Generate markdown timeline report

        Returns:
            Markdown formatted timeline
        """
        if not self.timeline_data:
            return "# Timeline\n\nNo entries to display.\n"

        df = self.get_timeline_dataframe()

        markdown = "# 📋 Forensic Timeline\n\n"

        if self.metadata:
            markdown += "## Case Information\n\n"
            markdown += f"- **Case ID**: {self.metadata.get('case_id', 'N/A')}\n"
            markdown += f"- **Chat**: {self.metadata.get('chat_name', 'N/A')}\n"
            markdown += f"- **Participants**: {', '.join(self.metadata.get('participants', []))}\n"
            markdown += f"- **Period**: {self.metadata.get('start_date', 'N/A')} to {self.metadata.get('end_date', 'N/A')}\n"
            markdown += f"- **Generated**: {self.metadata.get('generated_at', 'N/A')}\n\n"

        markdown += "## Timeline Entries\n\n"

        for idx, row in df.iterrows():
            markdown += self._format_entry_markdown(row)
            markdown += "\n---\n\n"

        return markdown

    def _format_entry_markdown(self, row: pd.Series) -> str:
        """
        Format a single timeline entry as markdown

        Args:
            row: DataFrame row

        Returns:
            Formatted markdown string
        """
        entry_md = f"### Entry {row.get('Index', 'N/A')}\n\n"

        entry_md += f"**Time**: {row.get('Recorded_At', 'N/A')}\n\n"
        entry_md += f"**Sender**: {row.get('Sender', 'N/A')}\n\n"

        if row.get('Message_Type') == 'audio':
            entry_md += f"**Type**: 🎙️ Voice Note\n\n"
            entry_md += f"**Duration**: {row.get('Duration', 0):.2f}s\n\n"
            entry_md += f"**Stress Level**: {row.get('Stress_Level', 0):.2f}/100\n\n"
            entry_md += f"**Emotion**: {row.get('Tone_Emotion', 'N/A')}\n\n"

            if row.get('Forensic_Flag'):
                entry_md += f"**⚠️ Flags**: {row.get('Forensic_Flag')}\n\n"

            entry_md += "**Bio-Signals**:\n"
            entry_md += f"- Pitch Volatility: {row.get('Pitch_Volatility', 0):.2f}\n"
            entry_md += f"- Silence Ratio: {row.get('Silence_Ratio', 0):.2f}\n"
            entry_md += f"- Intensity (Max): {row.get('Intensity_Max', 0):.2f}\n\n"

        else:
            entry_md += f"**Type**: 💬 Text Message\n\n"

        entry_md += f"**Content**:\n\n> {row.get('Transcript', 'N/A')}\n\n"

        if row.get('AI_Toxicity', 0) > 0.5:
            entry_md += f"🚨 **Toxicity Detected**: {row.get('AI_Toxicity', 0):.2f}\n\n"

        if row.get('AI_Gaslighting'):
            entry_md += "🚨 **Gaslighting Pattern Detected**\n\n"

        if row.get('AI_Contradiction'):
            entry_md += "⚠️ **Contradiction Detected**\n\n"

        return entry_md

    def export_to_csv(self, output_path: Optional[Path] = None) -> Tuple[bool, str]:
        """
        Export timeline to CSV

        Args:
            output_path: Path to save CSV (uses default if None)

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            if output_path is None:
                output_path = MOBITAB_DIR / "timeline.csv"

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            df = self.get_timeline_dataframe()

            if df.empty:
                return False, "No timeline data to export"

            df.to_csv(output_path, index=False, encoding='utf-8')

            logger.info(f"Exported timeline to {output_path}")
            return True, f"Exported to {output_path}"

        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")
            return False, f"Export error: {str(e)}"

    def export_to_markdown(self, output_path: Optional[Path] = None) -> Tuple[bool, str]:
        """
        Export timeline to markdown

        Args:
            output_path: Path to save markdown (uses default if None)

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            if output_path is None:
                output_path = MOBITAB_DIR / MOBITAB_FILENAME

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            markdown = self.generate_markdown_report()

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown)

            logger.info(f"Exported timeline to {output_path}")
            return True, f"Exported to {output_path}"

        except Exception as e:
            logger.error(f"Error exporting to markdown: {str(e)}")
            return False, f"Export error: {str(e)}"

    def export_to_json(self, output_path: Optional[Path] = None) -> Tuple[bool, str]:
        """
        Export timeline to JSON

        Args:
            output_path: Path to save JSON (uses default if None)

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            if output_path is None:
                output_path = MOBITAB_DIR / "timeline.json"

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            df = self.get_timeline_dataframe()

            if df.empty:
                return False, "No timeline data to export"

            df_dict = df.to_dict('records')

            metadata_dict = {
                'case_id': self.metadata.get('case_id'),
                'chat_name': self.metadata.get('chat_name'),
                'participants': self.metadata.get('participants'),
                'start_date': str(self.metadata.get('start_date')),
                'end_date': str(self.metadata.get('end_date')),
                'generated_at': str(self.metadata.get('generated_at')),
                'total_entries': len(df_dict)
            }

            export_data = {
                'metadata': metadata_dict,
                'timeline': df_dict
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Exported timeline to {output_path}")
            return True, f"Exported to {output_path}"

        except Exception as e:
            logger.error(f"Error exporting to JSON: {str(e)}")
            return False, f"Export error: {str(e)}"

    def get_statistics(self) -> Dict:
        """
        Get timeline statistics

        Returns:
            Dictionary with statistics
        """
        if not self.timeline_data:
            return {}

        df = self.get_timeline_dataframe()

        audio_entries = df[df['Message_Type'] == 'audio']
        text_entries = df[df['Message_Type'] == 'text']

        stats = {
            'total_entries': len(df),
            'audio_entries': len(audio_entries),
            'text_entries': len(text_entries),
            'messages_per_sender': df['Sender'].value_counts().to_dict(),
            'average_stress_level': audio_entries['Stress_Level'].mean() if len(audio_entries) > 0 else 0,
            'high_stress_entries': len(audio_entries[audio_entries['Stress_Level'] > 50]),
            'entries_with_flags': len(df[df['Forensic_Flag'] != '']),
            'toxicity_detected': len(df[df.get('AI_Toxicity', 0) > 0.5]),
            'gaslighting_detected': len(df[df.get('AI_Gaslighting', False) == True]),
        }

        logger.info(f"Generated statistics for {len(df)} timeline entries")
        return stats


if __name__ == "__main__":
    builder = MobitabBuilder()
    builder.set_metadata(
        case_id="CASE_001",
        chat_name="Investigation Chat",
        participants=["Person A", "Person B"],
        date_range=(datetime(2024, 1, 1), datetime(2024, 12, 31))
    )

    print("Mobitab Builder initialized")
    print(f"Metadata: {builder.metadata}")
