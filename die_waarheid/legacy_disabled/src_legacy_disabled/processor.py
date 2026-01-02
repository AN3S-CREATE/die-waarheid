import os
import re
import zipfile
from datetime import datetime
from typing import Dict, List, Optional, Union
import logging
import hashlib

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from .forensics import analyze_audio_file
except ImportError:
    analyze_audio_file = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhatsAppProcessor:
    """Processes WhatsApp export files and extracts conversation data."""
    
    def __init__(self, temp_dir: str = "temp_export"):
        """
        Initialize the WhatsApp processor.
        
        Args:
            temp_dir: Directory to extract temporary files
        """
        default_dir = temp_dir
        if os.path.isdir('Q:\\'):
            default_dir = os.path.join('Q:\\', 'die-waarheid', 'temp_export')
        self.temp_dir = os.getenv('DW_TEMP_DIR', default_dir)
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir, exist_ok=True)
    
    def process_export(self, zip_path: Union[str, bytes]) -> pd.DataFrame:
        """
        Process a WhatsApp export ZIP file.
        
        Args:
            zip_path: Path to the ZIP file or file-like object
            
        Returns:
            DataFrame containing the processed chat data
        """
        try:
            # Extract the ZIP file
            extract_path = self._extract_zip(zip_path)
            
            # Find and parse the chat file
            chat_file = self._find_chat_file(extract_path)
            if chat_file:
                chat_data = self._parse_chat_file(chat_file)
            else:
                chat_data = self._build_media_only_timeline(extract_path)
            
            # Process media files
            self._process_media_files(chat_data, extract_path)

            if not chat_data.empty:
                chat_data['audio_path'] = None
                audio_mask = (chat_data.get('message_type') == 'audio') & chat_data.get('media_path').notna()
                chat_data.loc[audio_mask, 'audio_path'] = chat_data.loc[audio_mask, 'media_path']

            return chat_data
            
        except Exception as e:
            logger.error(f"Error processing WhatsApp export: {e}", exc_info=True)
            raise
    
    def _extract_zip(self, zip_path: Union[str, bytes]) -> str:
        """Extract a ZIP file to the temporary directory."""
        try:
            target_dir = self.temp_dir
            if isinstance(zip_path, str):
                digest = hashlib.md5(zip_path.encode('utf-8', errors='ignore')).hexdigest()[:10]
                base = os.path.splitext(os.path.basename(zip_path))[0]
                safe_base = re.sub(r'[^A-Za-z0-9._-]+', '_', base)[:60]
                target_dir = os.path.join(self.temp_dir, f"{safe_base}_{digest}")
            os.makedirs(target_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            return target_dir
        except Exception as e:
            logger.error(f"Error extracting ZIP file: {e}")
            raise

    def process_exports_in_folder(self, folder_path: str) -> pd.DataFrame:
        if not folder_path or not os.path.isdir(folder_path):
            raise FileNotFoundError("Folder not found")

        zip_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith('.zip')
        ]

        if not zip_files:
            return pd.DataFrame()

        frames = []
        for zip_file in sorted(zip_files):
            df = self.process_export(zip_file)
            if df is None or df.empty:
                continue
            df = df.copy()
            df['source_zip'] = os.path.basename(zip_file)
            frames.append(df)

        if not frames:
            return pd.DataFrame()

        merged = pd.concat(frames, ignore_index=True)
        if 'event_id' in merged.columns:
            merged = merged.drop_duplicates(subset=['event_id'], keep='first')
        if 'timestamp' in merged.columns:
            merged = merged.sort_values('timestamp', kind='stable').reset_index(drop=True)
        return merged
    
    def _find_chat_file(self, directory: str) -> Optional[str]:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower() == "_chat.txt":
                    return os.path.join(root, file)

        candidates: List[str] = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.txt'):
                    candidates.append(os.path.join(root, file))

        if not candidates:
            return None

        def score(path: str) -> int:
            base = os.path.basename(path).lower()
            s = 0
            if 'chat' in base:
                s += 5
            if 'whatsapp' in base:
                s += 3
            return s

        for path in sorted(candidates, key=score, reverse=True):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for _ in range(30):
                        line = f.readline()
                        if not line:
                            break
                        if re.match(r'^\[?(\d{1,2}/\d{1,2}/\d{2,4}),?\s(\d{1,2}:\d{2})', line.strip()):
                            return path
            except Exception:
                continue

        return None

    def _build_media_only_timeline(self, extract_path: str) -> pd.DataFrame:
        media_rows: List[Dict] = []
        for root, _, files in os.walk(extract_path):
            for file in files:
                if file.lower().endswith('.txt'):
                    continue
                full_path = os.path.join(root, file)
                try:
                    mtime_ts = datetime.fromtimestamp(os.path.getmtime(full_path))
                except Exception:
                    continue
                ts = self._timestamp_from_media_filename(file, mtime_ts)
                ext = os.path.splitext(file)[1].lower()
                is_audio = ext in ['.opus', '.mp3', '.m4a', '.ogg', '.wav']
                is_image = ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']
                is_video = ext in ['.mp4', '.mov', '.avi', '.mkv']
                is_doc = ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx']
                if not (is_audio or is_image or is_video or is_doc):
                    continue

                message_type = 'other_media'
                if is_audio:
                    message_type = 'audio'
                elif is_image:
                    message_type = 'image'
                elif is_video:
                    message_type = 'video'
                elif is_doc:
                    message_type = 'document'

                media_rows.append({
                    'timestamp': ts,
                    'sender': 'Unknown',
                    'message': '',
                    'media_file': file,
                    'has_media': True,
                    'message_length': 0,
                    'word_count': 0,
                    'message_type': message_type,
                    'media_path': full_path,
                    'event_id': hashlib.sha1(f"{file}|{int(ts.timestamp())}|{os.path.getsize(full_path)}".encode('utf-8', errors='ignore')).hexdigest(),
                })

        df = pd.DataFrame(media_rows)
        if not df.empty:
            df = df.sort_values('timestamp', kind='stable').reset_index(drop=True)
            df['audio_path'] = None
            audio_mask = (df.get('message_type') == 'audio') & df.get('media_path').notna()
            df.loc[audio_mask, 'audio_path'] = df.loc[audio_mask, 'media_path']
        return df

    def _timestamp_from_media_filename(self, filename: str, fallback: datetime) -> datetime:
        base = os.path.basename(filename)
        m = re.match(r'^(AUD|PTT|IMG|VID|STK|DOC)-(\d{8})-WA\d+\.[A-Za-z0-9]+$', base, flags=re.IGNORECASE)
        if not m:
            return fallback

        datestr = m.group(2)
        try:
            date_only = datetime.strptime(datestr, '%Y%m%d')
            return datetime(
                date_only.year,
                date_only.month,
                date_only.day,
                fallback.hour,
                fallback.minute,
                fallback.second,
            )
        except Exception:
            return fallback

    def _build_event_id_from_row(self, row: pd.Series) -> str:
        ts = row.get('timestamp')
        ts_str = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
        sender = str(row.get('sender', ''))
        message = str(row.get('message', ''))
        media_file = str(row.get('media_file', ''))
        key = f"{ts_str}|{sender}|{message}|{media_file}"
        return hashlib.sha1(key.encode('utf-8', errors='ignore')).hexdigest()
    
    def _parse_chat_file(self, file_path: str) -> pd.DataFrame:
        """Parse the WhatsApp chat file into a structured format."""
        messages = []
        current_date = None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Try to match the date pattern at the start of a message
            date_match = re.match(
                r'^\[?(\d{1,2}/\d{1,2}/\d{2,4}),?\s(\d{1,2}:\d{2}(?::\d{2})?(?:\s[AP]M)?)\]?\s-?\s?',
                line,
            )
            if date_match:
                try:
                    date_str = f"{date_match.group(1)}, {date_match.group(2)}"
                    # Try different date formats
                    for fmt in (
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
                    ):
                        try:
                            current_date = datetime.strptime(date_str, fmt)
                            break
                        except ValueError:
                            continue
                    
                    # Extract the rest of the message
                    message_content = line[date_match.end():].strip()
                    
                    # Check if it's a system message or has a sender
                    if ': ' in message_content:
                        sender, message = message_content.split(':', 1)
                        sender = sender.strip()
                        message = message.strip()
                    else:
                        sender = "System"
                        message = message_content
                    
                    # Check for media attachments
                    media_file = None
                    if '<attached: ' in message:
                        media_match = re.search(r'<attached:\s*([^>]+)>', message)
                        if media_match:
                            media_file = media_match.group(1)
                            # Clean up the message
                            message = re.sub(r'<attached:[^>]+>', '', message).strip()
                    
                    messages.append({
                        'timestamp': current_date,
                        'sender': sender,
                        'message': message,
                        'media_file': media_file,
                        'has_media': bool(media_file)
                    })
                    
                except Exception as e:
                    logger.warning(f"Error parsing line: {line}\nError: {e}")
                    continue
            
            # Handle multi-line messages
            elif messages and current_date:
                messages[-1]['message'] += '\n' + line
        
        # Convert to DataFrame
        df = pd.DataFrame(messages)
        
        # Add additional metadata
        if not df.empty:
            df['message_length'] = df['message'].str.len()
            df['word_count'] = df['message'].str.split().str.len()
            
            # Extract message type
            df['message_type'] = df.apply(self._classify_message_type, axis=1)

            df['event_id'] = df.apply(self._build_event_id_from_row, axis=1)
        
        return df
    
    def _classify_message_type(self, row: pd.Series) -> str:
        """Classify the type of message."""
        if row['has_media']:
            media_ext = os.path.splitext(row['media_file'])[1].lower()
            if media_ext in ['.opus', '.mp3', '.m4a', '.ogg', '.wav']:
                return 'audio'
            elif media_ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                return 'image'
            elif media_ext in ['.mp4', '.mov', '.avi', '.mkv']:
                return 'video'
            elif media_ext in ['.pdf', '.doc', '.docx', '.xls', '.xlsx']:
                return 'document'
            else:
                return 'other_media'
        return 'text'
    
    def _process_media_files(self, df: pd.DataFrame, base_path: str) -> None:
        """Process media files and update the DataFrame with file paths."""
        if 'media_file' not in df.columns:
            return
        
        media_paths = []
        
        for _, row in df[df['has_media']].iterrows():
            media_file = row['media_file']
            if not media_file:
                media_paths.append(None)
                continue
                
            # Search for the media file in the extracted directory
            found = False
            for root, _, files in os.walk(base_path):
                if media_file in files:
                    media_path = os.path.join(root, media_file)
                    media_paths.append(media_path)
                    found = True
                    break
            
            if not found:
                media_paths.append(None)
                logger.warning(f"Media file not found: {media_file}")
        
        # Update the DataFrame
        df.loc[df['has_media'], 'media_path'] = media_paths

    def enrich_audio_forensics(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        if 'message_type' not in df.columns:
            return df

        if 'audio_path' not in df.columns:
            if 'media_path' in df.columns:
                df = df.copy()
                df['audio_path'] = None
                audio_mask = (df['message_type'] == 'audio') & df['media_path'].notna()
                df.loc[audio_mask, 'audio_path'] = df.loc[audio_mask, 'media_path']
            else:
                return df

        df = df.copy()

        for col in (
            'duration',
            'transcript',
            'language',
            'speaker_count',
            'pitch_volatility',
            'silence_ratio',
            'max_loudness',
            'analysis_successful',
            'audio_error',
        ):
            if col not in df.columns:
                df[col] = None

        audio_rows = df[(df['message_type'] == 'audio') & df['audio_path'].notna()].index
        total_audio = len(audio_rows)
        logger.info(f"Starting audio forensics enrichment for {total_audio} audio files...")
        if total_audio == 0:
            logger.warning("No audio files found to analyze.")
            return df

        for i, idx in enumerate(audio_rows, 1):
            audio_path = df.at[idx, 'audio_path']
            logger.info(f"[{i}/{total_audio}] Analyzing: {audio_path}")
            try:
                result = analyze_audio_file(audio_path)
                df.at[idx, 'duration'] = result.get('duration')
                df.at[idx, 'transcript'] = result.get('transcript')
                df.at[idx, 'language'] = result.get('language')
                df.at[idx, 'speaker_count'] = result.get('speaker_count')
                df.at[idx, 'pitch_volatility'] = result.get('pitch_volatility')
                df.at[idx, 'silence_ratio'] = result.get('silence_ratio')
                df.at[idx, 'max_loudness'] = result.get('max_loudness')
                df.at[idx, 'analysis_successful'] = result.get('analysis_successful')
                df.at[idx, 'audio_error'] = result.get('error')
                if result.get('analysis_successful'):
                    logger.info(f"  ✓ Success: {result.get('language', 'unknown')} | {result.get('duration', 0)}s")
                else:
                    logger.warning(f"  ✗ Failed: {result.get('error', 'unknown error')}")
            except Exception as e:
                df.at[idx, 'analysis_successful'] = False
                df.at[idx, 'audio_error'] = str(e)
                logger.error(f"  ✗ Exception: {e}")

        logger.info("Audio forensics enrichment completed.")
        return df

# Global instance for easy importing
whatsapp_processor = WhatsAppProcessor()

def enrich_audio_forensics(df: pd.DataFrame) -> pd.DataFrame:
    return whatsapp_processor.enrich_audio_forensics(df)

def process_whatsapp_folder(folder_path: str) -> pd.DataFrame:
    return whatsapp_processor.process_exports_in_folder(folder_path)

def process_whatsapp_export(zip_path: Union[str, bytes]) -> pd.DataFrame:
    """
    Process a WhatsApp export ZIP file.
    
    Args:
        zip_path: Path to the ZIP file or file-like object
        
    Returns:
        DataFrame containing the processed chat data
    """
    return whatsapp_processor.process_export(zip_path)
