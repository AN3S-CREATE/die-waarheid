"""
WhatsApp Chat Parser for Die Waarheid
Parses WhatsApp exports and extracts structured message data
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


class WhatsAppParser:
    """
    Parses WhatsApp chat exports in standard text format
    Supports both individual and group chats
    """

    TIMESTAMP_PATTERNS = [
        r'^\[?(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s*(?:AM|PM|am|pm))?)\]?\s*[-–]\s*',
        r'^(\d{1,2}/\d{1,2}/\d{2,4})\s+(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–]\s*',
        r'^(\d{4}-\d{2}-\d{2})\s+(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–]\s*',
    ]

    SYSTEM_MESSAGE_PATTERNS = [
        r'Messages and calls are encrypted',
        r'You created this group',
        r'You added',
        r'removed',
        r'left',
        r'joined',
        r'changed the subject',
        r'changed this group\'s icon',
        r'Media omitted',
        r'<Media omitted>',
    ]

    def __init__(self):
        self.messages = []
        self.participants = set()
        self.chat_metadata = {}

    def parse_file(self, file_path: Path) -> Tuple[bool, str]:
        """
        Parse WhatsApp export file

        Args:
            file_path: Path to WhatsApp export text file

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return False, f"File not found: {file_path}"

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            self.messages = []
            self.participants = set()

            lines = content.split('\n')
            current_message = None

            for line in lines:
                if not line.strip():
                    continue

                parsed = self._parse_line(line)

                if parsed and parsed.get('is_new_message'):
                    if current_message:
                        self.messages.append(current_message)

                    current_message = parsed
                    if parsed.get('sender'):
                        self.participants.add(parsed['sender'])

                elif current_message and not parsed.get('is_new_message'):
                    current_message['text'] += '\n' + line

            if current_message:
                self.messages.append(current_message)

            self._extract_metadata()

            logger.info(f"Parsed {len(self.messages)} messages from {file_path.name}")
            logger.info(f"Found {len(self.participants)} participants")

            return True, f"Successfully parsed {len(self.messages)} messages"

        except Exception as e:
            logger.error(f"Error parsing file: {str(e)}")
            return False, f"Error parsing file: {str(e)}"

    def _parse_line(self, line: str) -> Optional[Dict]:
        """
        Parse a single line from WhatsApp export

        Args:
            line: Line from chat export

        Returns:
            Dictionary with message data or None
        """
        for pattern in self.TIMESTAMP_PATTERNS:
            match = re.match(pattern, line)

            if match:
                try:
                    timestamp_str = f"{match.group(1)} {match.group(2)}"
                    timestamp = self._parse_timestamp(timestamp_str)

                    rest = re.sub(pattern, '', line).strip()

                    if ':' in rest:
                        sender, text = rest.split(':', 1)
                        sender = sender.strip()
                        text = text.strip()
                    else:
                        sender = "System"
                        text = rest

                    is_system = self._is_system_message(text)

                    return {
                        'is_new_message': True,
                        'timestamp': timestamp,
                        'timestamp_str': timestamp_str,
                        'sender': sender,
                        'text': text,
                        'is_system': is_system,
                        'message_type': self._detect_message_type(text)
                    }

                except Exception as e:
                    logger.debug(f"Error parsing line: {str(e)}")
                    return None

        return {'is_new_message': False}

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """
        Parse timestamp string to datetime object

        Args:
            timestamp_str: Timestamp string

        Returns:
            datetime object
        """
        formats = [
            '%m/%d/%Y, %I:%M:%S %p',
            '%m/%d/%Y, %I:%M %p',
            '%m/%d/%Y %I:%M:%S %p',
            '%m/%d/%Y %I:%M %p',
            '%d/%m/%Y, %H:%M:%S',
            '%d/%m/%Y, %H:%M',
            '%d/%m/%Y %H:%M:%S',
            '%d/%m/%Y %H:%M',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M',
        ]

        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue

        logger.warning(f"Could not parse timestamp: {timestamp_str}")
        return datetime.now()

    def _is_system_message(self, text: str) -> bool:
        """
        Check if message is a system message

        Args:
            text: Message text

        Returns:
            True if system message
        """
        for pattern in self.SYSTEM_MESSAGE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def _detect_message_type(self, text: str) -> str:
        """
        Detect message type (text, image, audio, video, etc.)

        Args:
            text: Message text

        Returns:
            Message type string
        """
        if '<Media omitted>' in text or 'Media omitted' in text:
            if 'image' in text.lower():
                return 'image'
            elif 'audio' in text.lower() or 'voice' in text.lower():
                return 'audio'
            elif 'video' in text.lower():
                return 'video'
            else:
                return 'media'

        if text.startswith('http') or 'http' in text:
            return 'link'

        return 'text'

    def _extract_metadata(self):
        """Extract metadata from parsed messages"""
        if not self.messages:
            self.chat_metadata = {}
            return

        timestamps = [m['timestamp'] for m in self.messages if m.get('timestamp')]

        self.chat_metadata = {
            'total_messages': len(self.messages),
            'total_participants': len(self.participants),
            'participants': sorted(list(self.participants)),
            'start_date': min(timestamps) if timestamps else None,
            'end_date': max(timestamps) if timestamps else None,
            'duration_days': (max(timestamps) - min(timestamps)).days if len(timestamps) > 1 else 0,
            'system_messages': sum(1 for m in self.messages if m.get('is_system')),
            'user_messages': sum(1 for m in self.messages if not m.get('is_system')),
        }

    def get_messages(self) -> List[Dict]:
        """
        Get all parsed messages

        Returns:
            List of message dictionaries
        """
        return self.messages

    def get_messages_by_sender(self, sender: str) -> List[Dict]:
        """
        Get all messages from a specific sender

        Args:
            sender: Sender name

        Returns:
            List of messages from sender
        """
        return [m for m in self.messages if m.get('sender') == sender]

    def get_messages_in_range(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """
        Get messages within a date range

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            List of messages in range
        """
        return [
            m for m in self.messages
            if start_date <= m.get('timestamp', datetime.now()) <= end_date
        ]

    def get_metadata(self) -> Dict:
        """
        Get chat metadata

        Returns:
            Dictionary with chat metadata
        """
        return self.chat_metadata

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert messages to pandas DataFrame

        Returns:
            DataFrame with message data
        """
        if not self.messages:
            return pd.DataFrame()

        df = pd.DataFrame(self.messages)

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            df['time'] = df['timestamp'].dt.time
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.day_name()

        logger.info(f"Created DataFrame with {len(df)} rows")
        return df

    def export_to_csv(self, output_path: Path) -> Tuple[bool, str]:
        """
        Export messages to CSV file

        Args:
            output_path: Path to save CSV

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            df = self.to_dataframe()

            if df.empty:
                return False, "No messages to export"

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            df.to_csv(output_path, index=False, encoding='utf-8')

            logger.info(f"Exported {len(df)} messages to {output_path}")
            return True, f"Exported to {output_path}"

        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")
            return False, f"Export error: {str(e)}"

    def get_statistics(self) -> Dict:
        """
        Get chat statistics

        Returns:
            Dictionary with statistics
        """
        if not self.messages:
            return {}

        df = self.to_dataframe()

        if df.empty:
            return {}

        stats = {
            'total_messages': len(df),
            'total_participants': len(self.participants),
            'messages_per_participant': df['sender'].value_counts().to_dict(),
            'messages_per_day': df['date'].value_counts().to_dict(),
            'messages_per_hour': df['hour'].value_counts().to_dict(),
            'message_types': df['message_type'].value_counts().to_dict(),
            'average_message_length': df['text'].str.len().mean(),
            'longest_message': df['text'].str.len().max(),
            'shortest_message': df['text'].str.len().min(),
        }

        logger.info(f"Generated statistics for {len(df)} messages")
        return stats


if __name__ == "__main__":
    parser = WhatsAppParser()
    test_file = Path("data/text/chat_export.txt")

    if test_file.exists():
        success, message = parser.parse_file(test_file)
        print(f"Parse result: {message}")

        if success:
            print(f"\nMetadata: {parser.get_metadata()}")
            print(f"\nStatistics: {parser.get_statistics()}")
    else:
        print(f"Test file not found: {test_file}")
