"""
WhatsApp Chat Parser

Robust parsing of WhatsApp export files with support for multiple formats,
languages, and media file association
"""

import re
import os
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Iterator
import logging

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from chardet import detect
except ImportError:
    detect = None

try:
    import emoji
except ImportError:
    emoji = None

from config import TEXT_DIR, TEMP_DIR, TIMESTAMP_FORMAT

logger = logging.getLogger(__name__)

class WhatsAppChatParser:
    """Parse WhatsApp chat exports with forensic-grade accuracy"""
    
    # WhatsApp timestamp patterns for different locales
    TIMESTAMP_PATTERNS = [
        # English formats
        r'\[(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?::\d{2})?(?:\s[AP]M)?)\]',
        r'(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?::\d{2})?(?:\s[AP]M)?)\s-\s',
        # South African format (DD/MM/YYYY)
        r'\[(\d{1,2}/\d{1,2}/\d{4}),\s(\d{1,2}:\d{2}:\d{2})\]',
        # US format (MM/DD/YYYY)
        r'\[(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?::\d{2})?\s[AP]M)\]',
        # 24-hour formats
        r'(\d{1,2}\.\d{1,2}\.\d{2,4}),\s(\d{1,2}:\d{2})',
        r'(\d{1,2}-\d{1,2}-\d{2,4}),\s(\d{1,2}:\d{2})',
        # WhatsApp Web format
        r'\[(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}:\d{2})(?::\d{2})?\]',
    ]
    
    # Date format patterns to try
    DATE_FORMATS = [
        '%d/%m/%Y, %H:%M',
        '%d/%m/%y, %H:%M',
        '%m/%d/%Y, %H:%M',
        '%m/%d/%y, %H:%M',
        '%d/%m/%Y, %H:%M:%S',
        '%d/%m/%y, %H:%M:%S',
        '%m/%d/%Y, %H:%M:%S',
        '%m/%d/%y, %H:%M:%S',
        '%d/%m/%Y, %I:%M %p',
        '%d/%m/%y, %I:%M %p',
        '%m/%d/%Y, %I:%M %p',
        '%m/%d/%y, %I:%M %p',
        '%d/%m/%Y, %I:%M:%S %p',
        '%d/%m/%y, %I:%M:%S %p',
        '%m/%d/%Y, %I:%M:%S %p',
        '%m/%d/%y, %I:%M:%S %p',
        '%d.%m.%Y, %H:%M',
        '%d.%m.%y, %H:%M',
        '%d-%m-%Y, %H:%M',
        '%d-%m-%y, %H:%M',
    ]
    
    # Media file patterns
    MEDIA_PATTERNS = {
        'audio': [
            r'(?i)<attached:\s*([^>]+\.(?:opus|m4a|mp3|wav|ogg))>',
            r'(?i)PTT-(\d{8}-WA\d{4}\.(?:opus|m4a|mp3))',
            r'(?i)AUD-(\d{8}-WA\d{4}\.(?:opus|m4a|mp3))',
        ],
        'image': [
            r'(?i)<attached:\s*([^>]+\.(?:jpg|jpeg|png|gif|webp))>',
            r'(?i)IMG-(\d{8}-WA\d{4}\.(?:jpg|jpeg|png|gif))',
        ],
        'video': [
            r'(?i)<attached:\s*([^>]+\.(?:mp4|mov|avi|mkv|webm))>',
            r'(?i)VID-(\d{8}-WA\d{4}\.(?:mp4|mov|avi))',
        ],
        'document': [
            r'(?i)<attached:\s*([^>]+\.(?:pdf|doc|docx|xls|xlsx|ppt|pptx|txt))>',
            r'(?i)DOC-(\d{8}-WA\d{4}\.(?:pdf|doc|docx|xls|xlsx))',
        ],
        'sticker': [
            r'(?i)<attached:\s*([^>]+\.(?:webp|tgs))>',
            r'(?i)STK-(\d{8}-WA\d{4}\.(?:webp|tgs))',
        ]
    }
    
    # System message patterns
    SYSTEM_PATTERNS = [
        r'(?i)^(.+ added .+)$',
        r'(?i)^(.+ left)$',
        r'(?i)^(.+ changed the subject to .+)$',
        r'(?i)^(.+ changed this group\'s icon)$',
        r'(?i)^(.+ created group .+)$',
        r'(?i)^(.+ removed .+)$',
        r'(?i)Messages and calls are end-to-end encrypted',
        r'(?i)^\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}:\d{2}\s-\s.+$',
        r'(?i)<Media omitted>',
    ]
    
    def __init__(self):
        self.messages = []
        self.current_date = None
        
    def parse_export_file(self, file_path: Path) -> pd.DataFrame:
        """Parse a WhatsApp export file (ZIP or TXT)"""
        self.messages = []
        
        if file_path.suffix.lower() == '.zip':
            return self._parse_zip_file(file_path)
        elif file_path.suffix.lower() == '.txt':
            return self._parse_txt_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _parse_zip_file(self, zip_path: Path) -> pd.DataFrame:
        """Parse a WhatsApp export ZIP file"""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract to temp directory
            extract_dir = TEMP_DIR / f"extract_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            zip_ref.extractall(extract_dir)
            
            try:
                # Find the chat file
                chat_file = self._find_chat_file(extract_dir)
                if not chat_file:
                    logger.warning("No chat file found. Creating timeline from media files only.")
                    df = self._create_media_only_timeline(extract_dir)
                else:
                    # Parse the chat file
                    df = self._parse_txt_file(chat_file)
                    # Associate media files
                    df = self._associate_media_files(df, extract_dir)
                
                return df
                
            finally:
                # Clean up temp directory
                import shutil
                shutil.rmtree(extract_dir, ignore_errors=True)
    
    def _create_media_only_timeline(self, directory: Path) -> pd.DataFrame:
        """Create a timeline from media files when no chat file is available"""
        messages = []
        
        # Find all media files
        media_files = []
        for ext in ['*.opus', '*.m4a', '*.mp3', '*.wav', '*.ogg',  # audio
                    '*.jpg', '*.jpeg', '*.png', '*.gif', '*.webp',  # images
                    '*.mp4', '*.mov', '*.avi', '*.mkv', '*.webm']:  # video
            media_files.extend(directory.rglob(ext))
        
        for media_file in media_files:
            # Extract timestamp from filename
            file_timestamp = self.extract_date_from_media_filename(media_file.name)
            
            if file_timestamp:
                # Determine message type from extension
                ext = media_file.suffix.lower()
                if ext in ['.opus', '.m4a', '.mp3', '.wav', '.ogg']:
                    message_type = 'audio'
                elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                    message_type = 'image'
                elif ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
                    message_type = 'video'
                else:
                    message_type = 'document'
                
                messages.append({
                    'timestamp': file_timestamp,
                    'sender': 'Unknown',
                    'message': f'<Media file: {media_file.name}>',
                    'message_type': message_type,
                    'media_file': media_file.name,
                    'has_media': True,
                    'media_path': str(media_file)
                })
        
        # Sort by timestamp
        messages.sort(key=lambda x: x['timestamp'])
        
        # Create DataFrame
        df = pd.DataFrame(messages)
        
        if not df.empty:
            logger.info(f"Created timeline with {len(df)} media files from {df['timestamp'].min()} to {df['timestamp'].max()}")
        else:
            logger.warning("No media files with valid timestamps found")
        
        return df
    
    def _find_chat_file(self, directory: Path) -> Optional[Path]:
        # Common chat file names
        possible_names = [
            'WhatsApp Chat with .txt',
            'WhatsApp Chat.txt',
            'chat.txt',
            '_chat.txt',
        ]
        
        # Try exact matches first
        for name in possible_names:
            file_path = directory / name
            if file_path.exists():
                return file_path
        
        # Search for any .txt file that looks like a chat
        for txt_file in directory.rglob('*.txt'):
            if self._is_chat_file(txt_file):
                return txt_file
        
        return None
    
    def _is_chat_file(self, file_path: Path) -> bool:
        """Check if a file is likely a WhatsApp chat file"""
        try:
            # Read first few lines to check for WhatsApp patterns
            with open(file_path, 'rb') as f:
                raw_data = f.read(1024)
                encoding = detect(raw_data)['encoding'] or 'utf-8'
            
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= 10:  # Check first 10 lines
                        break
                    for pattern in self.TIMESTAMP_PATTERNS:
                        if re.search(pattern, line):
                            return True
            return False
            
        except Exception:
            return False
    
    def _parse_txt_file(self, file_path: Path) -> pd.DataFrame:
        """Parse a WhatsApp chat text file"""
        # Detect encoding
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            encoding = detect(raw_data)['encoding'] or 'utf-8'
        
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            lines = f.readlines()
        
        self.messages = []
        self.current_date = None
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                self._parse_line(line)
            except Exception as e:
                # Safely encode the line for logging to avoid Unicode errors
                safe_line = line.encode('ascii', errors='replace').decode('ascii')
                logger.warning(f"Error parsing line {line_num}: {safe_line}\nError: {e}")
                continue
        
        return pd.DataFrame(self.messages)
    
    def _parse_line(self, line: str):
        """Parse a single line from the chat"""
        # Check for timestamp
        timestamp_match = None
        for pattern in self.TIMESTAMP_PATTERNS:
            timestamp_match = re.match(pattern, line)
            if timestamp_match:
                break
        
        if timestamp_match:
            # Extract timestamp
            date_str = timestamp_match.group(1)
            time_str = timestamp_match.group(2)
            full_timestamp = f"{date_str}, {time_str}"
            
            # Parse timestamp
            for fmt in self.DATE_FORMATS:
                try:
                    self.current_date = datetime.strptime(full_timestamp, fmt)
                    break
                except ValueError as e:
                    # Log the specific format that failed for debugging
                    logger.debug(f"Date format '{fmt}' failed for '{full_timestamp}': {e}")
                    continue
            
            if not self.current_date:
                logger.warning(f"Could not parse timestamp: {full_timestamp} - skipping line")
                return
            
            # Extract message content
            message_content = line[timestamp_match.end():].strip()
            
            # Check for system message
            if self._is_system_message(message_content):
                self.messages.append({
                    'timestamp': self.current_date,
                    'sender': 'System',
                    'message': message_content,
                    'message_type': 'system',
                    'media_file': None,
                    'has_media': False
                })
                return
            
            # Extract sender and message
            if ': ' in message_content:
                sender, message = message_content.split(':', 1)
                sender = sender.strip()
                message = message.strip()
            else:
                sender = "Unknown"
                message = message_content
            
            # Check for media attachments
            media_file, media_type = self._extract_media_info(message)
            if media_file:
                # Clean up message
                message = re.sub(r'<attached:[^>]+>', '', message).strip()
            
            # Detect message type
            message_type = self._detect_message_type(message, media_type)
            
            self.messages.append({
                'timestamp': self.current_date,
                'sender': sender,
                'message': message,
                'message_type': message_type,
                'media_file': media_file,
                'has_media': bool(media_file)
            })
            
        elif self.messages and self.current_date:
            # Continuation of previous message
            self.messages[-1]['message'] += '\n' + line
    
    def _is_system_message(self, message: str) -> bool:
        """Check if message is a system message"""
        for pattern in self.SYSTEM_PATTERNS:
            if re.match(pattern, message):
                return True
        return False
    
    def _extract_media_info(self, message: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract media file information from message"""
        for media_type, patterns in self.MEDIA_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, message)
                if match:
                    return match.group(1), media_type
        return None, None
    
    def _detect_message_type(self, message: str, media_type: str = None) -> str:
        """Detect the type of message based on content"""
        if media_type:
            return media_type
        
        # Check for different message types
        if message.lower() in ['<media omitted>', 'image omitted', 'video omitted', 'audio omitted', 'document omitted']:
            return 'media_omitted'
        
        # Count emojis using the new API
        try:
            emoji_count = emoji.emoji_count(message)
        except:
            emoji_count = 0
        
        # Check for common patterns
        if any(word in message.lower() for word in ['http://', 'https://', 'www.']):
            return 'link'
        elif emoji_count > 0 and len(message.strip()) <= 10:
            return 'emoji'
        elif message.strip() == '':
            return 'empty'
        else:
            return 'text'
    
    def _associate_media_files(self, df: pd.DataFrame, extract_dir: Path) -> pd.DataFrame:
        """Associate media files with messages"""
        if 'media_file' not in df.columns:
            return df
        
        df = df.copy()
        df['media_path'] = None
        
        # Get all media files in extracted directory
        media_files = list(extract_dir.rglob('*'))
        media_files = [f for f in media_files if f.is_file()]
        
        for idx, row in df[df['has_media']].iterrows():
            media_file = row['media_file']
            if not media_file:
                continue
            
            # Search for the media file
            for file_path in media_files:
                if file_path.name == media_file:
                    df.at[idx, 'media_path'] = str(file_path)
                    break
        
        return df
    
    def extract_date_from_media_filename(self, filename: str) -> Optional[datetime]:
        """Extract date from WhatsApp media filename"""
        # Pattern: IMG-YYYYMMDD-WAxxxx.jpg
        match = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
        if match:
            year, month, day = map(int, match.groups())
            try:
                return datetime(year, month, day)
            except ValueError as e:
                logger.warning(f"Invalid date in filename {filename}: {e}")
                return None
        return None
    
    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """Get forensic statistics about the chat"""
        stats = {
            'total_messages': len(df),
            'date_range': {
                'start': df['timestamp'].min().isoformat() if not df.empty else None,
                'end': df['timestamp'].max().isoformat() if not df.empty else None
            },
            'participants': df['sender'].nunique(),
            'message_types': df['message_type'].value_counts().to_dict(),
            'media_count': df['has_media'].sum(),
            'most_active': df['sender'].value_counts().head().to_dict(),
            'peak_hours': self._get_peak_hours(df),
            'average_message_length': df['message'].str.len().mean() if 'message' in df.columns else 0
        }
        return stats
    
    def _get_peak_hours(self, df: pd.DataFrame) -> List[int]:
        """Get the hours with most activity"""
        if df.empty or 'timestamp' not in df.columns:
            return []
        
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        hourly_counts = df['hour'].value_counts().head(3)
        return hourly_counts.index.tolist()

# Global instance for easy importing
whatsapp_parser = WhatsAppChatParser()
