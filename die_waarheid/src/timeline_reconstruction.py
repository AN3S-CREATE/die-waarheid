"""
Chronological Timeline Reconstruction for Die Waarheid
Builds accurate timelines from multiple sources WITHOUT requiring main chat file

SOURCES FOR TIMELINE RECONSTRUCTION:
1. Audio file metadata (creation date, modification date)
2. Voice note filenames (often contain timestamps)
3. EXIF data from media files
4. File system timestamps
5. Audio content analysis (references to time, dates)
6. Cross-referencing multiple sources
7. WhatsApp media naming conventions
8. Google Drive metadata
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)


class TimestampSource(Enum):
    """Source of timestamp"""
    FILE_CREATION = "file_creation"
    FILE_MODIFICATION = "file_modification"
    FILENAME_PATTERN = "filename_pattern"
    EXIF_DATA = "exif_data"
    AUDIO_METADATA = "audio_metadata"
    CONTENT_REFERENCE = "content_reference"
    WHATSAPP_FORMAT = "whatsapp_format"
    GDRIVE_METADATA = "gdrive_metadata"
    USER_PROVIDED = "user_provided"
    CROSS_REFERENCE = "cross_reference"
    INFERRED = "inferred"


class TimelineConfidence(Enum):
    """Confidence level for timeline entry"""
    CERTAIN = "certain"           # 100% confident (multiple sources agree)
    HIGH = "high"                 # 90%+ (primary source + confirmation)
    MEDIUM = "medium"             # 70-90% (single reliable source)
    LOW = "low"                   # 50-70% (inferred or uncertain)
    UNCERTAIN = "uncertain"       # <50% (best guess)


@dataclass
class TimestampExtraction:
    """Extracted timestamp from a source"""
    timestamp: datetime
    source: TimestampSource
    confidence: float
    raw_value: str
    notes: str = ""


@dataclass
class TimelineEntry:
    """Single entry in the reconstructed timeline"""
    entry_id: str
    timestamp: datetime
    timestamp_confidence: TimelineConfidence
    sources: List[TimestampExtraction]
    content_type: str  # audio, image, video, document
    file_path: Optional[str] = None
    speaker_id: Optional[str] = None
    transcription: Optional[str] = None
    translation: Optional[str] = None
    duration: Optional[float] = None
    forensic_flags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'entry_id': self.entry_id,
            'timestamp': self.timestamp.isoformat(),
            'timestamp_confidence': self.timestamp_confidence.value,
            'sources': [s.source.value for s in self.sources],
            'content_type': self.content_type,
            'file_path': self.file_path,
            'speaker_id': self.speaker_id,
            'transcription': self.transcription,
            'translation': self.translation,
            'duration': self.duration,
            'forensic_flags': self.forensic_flags,
            'metadata': self.metadata
        }


class TimestampExtractor:
    """
    Extract timestamps from multiple sources
    """

    # WhatsApp voice note patterns
    WHATSAPP_PATTERNS = [
        # PTT-YYYYMMDD-WAXXXX.opus
        r'PTT-(\d{8})-WA\d+\.(opus|ogg|mp3)',
        # AUD-YYYYMMDD-WAXXXX.opus
        r'AUD-(\d{8})-WA\d+\.(opus|ogg|mp3|m4a)',
        # VN-YYYYMMDD-WAXXXX.opus
        r'VN-(\d{8})-WA\d+\.(opus|ogg|mp3)',
        # IMG-YYYYMMDD-WAXXXX.jpg
        r'IMG-(\d{8})-WA\d+\.(jpg|jpeg|png)',
        # VID-YYYYMMDD-WAXXXX.mp4
        r'VID-(\d{8})-WA\d+\.(mp4|3gp)',
        # WhatsApp Audio YYYY-MM-DD at HH.MM.SS
        r'WhatsApp Audio (\d{4}-\d{2}-\d{2}) at (\d{2}\.\d{2}\.\d{2})',
        # WhatsApp Voice Note YYYY-MM-DD
        r'WhatsApp Voice (\d{4}-\d{2}-\d{2})',
    ]

    # General date patterns in filenames
    DATE_PATTERNS = [
        # YYYYMMDD
        (r'(\d{4})(\d{2})(\d{2})', '%Y%m%d'),
        # YYYY-MM-DD
        (r'(\d{4})-(\d{2})-(\d{2})', '%Y-%m-%d'),
        # DD-MM-YYYY
        (r'(\d{2})-(\d{2})-(\d{4})', '%d-%m-%Y'),
        # YYYY_MM_DD
        (r'(\d{4})_(\d{2})_(\d{2})', '%Y_%m_%d'),
        # YYYYMMDD_HHMMSS
        (r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})', '%Y%m%d_%H%M%S'),
    ]

    def __init__(self):
        """Initialize timestamp extractor"""
        pass

    def extract_from_filename(self, filename: str) -> List[TimestampExtraction]:
        """
        Extract timestamp from filename patterns

        Args:
            filename: Filename to analyze

        Returns:
            List of extracted timestamps
        """
        extractions = []
        
        # Try WhatsApp patterns
        for pattern in self.WHATSAPP_PATTERNS:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                try:
                    date_str = match.group(1)
                    
                    # Handle different formats
                    if len(date_str) == 8:  # YYYYMMDD
                        dt = datetime.strptime(date_str, '%Y%m%d')
                    elif '-' in date_str:  # YYYY-MM-DD
                        dt = datetime.strptime(date_str, '%Y-%m-%d')
                    else:
                        continue

                    # Check for time component
                    if len(match.groups()) > 1 and match.group(2):
                        time_str = match.group(2)
                        if '.' in time_str:
                            time_parts = time_str.split('.')
                            if len(time_parts) >= 2:
                                dt = dt.replace(
                                    hour=int(time_parts[0]),
                                    minute=int(time_parts[1]),
                                    second=int(time_parts[2]) if len(time_parts) > 2 else 0
                                )

                    extractions.append(TimestampExtraction(
                        timestamp=dt,
                        source=TimestampSource.WHATSAPP_FORMAT,
                        confidence=0.95,
                        raw_value=match.group(0),
                        notes=f"WhatsApp format detected: {pattern}"
                    ))
                except (ValueError, IndexError) as e:
                    logger.debug(f"Failed to parse WhatsApp pattern: {e}")

        # Try general date patterns
        for pattern, date_format in self.DATE_PATTERNS:
            match = re.search(pattern, filename)
            if match:
                try:
                    date_str = match.group(0)
                    dt = datetime.strptime(date_str, date_format)
                    
                    # Validate date is reasonable
                    if 2000 <= dt.year <= datetime.now().year + 1:
                        extractions.append(TimestampExtraction(
                            timestamp=dt,
                            source=TimestampSource.FILENAME_PATTERN,
                            confidence=0.80,
                            raw_value=date_str,
                            notes=f"Date pattern: {date_format}"
                        ))
                except ValueError:
                    pass

        return extractions

    def extract_from_file_stats(self, file_path: Path) -> List[TimestampExtraction]:
        """
        Extract timestamps from file system metadata

        Args:
            file_path: Path to file

        Returns:
            List of extracted timestamps
        """
        extractions = []
        
        try:
            stat = file_path.stat()
            
            # Creation time
            if hasattr(stat, 'st_birthtime'):
                ctime = datetime.fromtimestamp(stat.st_birthtime)
            else:
                ctime = datetime.fromtimestamp(stat.st_ctime)
            
            extractions.append(TimestampExtraction(
                timestamp=ctime,
                source=TimestampSource.FILE_CREATION,
                confidence=0.70,
                raw_value=str(stat.st_ctime),
                notes="File system creation time"
            ))
            
            # Modification time
            mtime = datetime.fromtimestamp(stat.st_mtime)
            extractions.append(TimestampExtraction(
                timestamp=mtime,
                source=TimestampSource.FILE_MODIFICATION,
                confidence=0.60,
                raw_value=str(stat.st_mtime),
                notes="File system modification time"
            ))

        except Exception as e:
            logger.error(f"Error extracting file stats: {e}")

        return extractions

    def extract_from_audio_metadata(self, file_path: Path) -> List[TimestampExtraction]:
        """
        Extract timestamps from audio file metadata

        Args:
            file_path: Path to audio file

        Returns:
            List of extracted timestamps
        """
        extractions = []
        
        try:
            from mutagen import File as MutagenFile
            
            audio = MutagenFile(str(file_path))
            if audio is None:
                return extractions

            # Check various metadata fields
            date_fields = ['date', 'TDRC', 'TYER', 'creation_time', 'DATE']
            
            for field in date_fields:
                if field in audio:
                    try:
                        date_str = str(audio[field][0]) if isinstance(audio[field], list) else str(audio[field])
                        
                        # Try various date formats
                        for fmt in ['%Y-%m-%d', '%Y%m%d', '%Y', '%Y-%m-%dT%H:%M:%S']:
                            try:
                                dt = datetime.strptime(date_str[:len(fmt.replace('%', ''))], fmt)
                                extractions.append(TimestampExtraction(
                                    timestamp=dt,
                                    source=TimestampSource.AUDIO_METADATA,
                                    confidence=0.85,
                                    raw_value=date_str,
                                    notes=f"Audio metadata field: {field}"
                                ))
                                break
                            except ValueError:
                                continue
                    except Exception:
                        pass

        except ImportError:
            logger.debug("mutagen not available for audio metadata extraction")
        except Exception as e:
            logger.error(f"Error extracting audio metadata: {e}")

        return extractions

    def extract_from_exif(self, file_path: Path) -> List[TimestampExtraction]:
        """
        Extract timestamps from EXIF data (images/videos)

        Args:
            file_path: Path to media file

        Returns:
            List of extracted timestamps
        """
        extractions = []
        
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS
            
            with Image.open(file_path) as img:
                exif_data = img._getexif()
                if exif_data:
                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)
                        
                        if tag in ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']:
                            try:
                                dt = datetime.strptime(value, '%Y:%m:%d %H:%M:%S')
                                extractions.append(TimestampExtraction(
                                    timestamp=dt,
                                    source=TimestampSource.EXIF_DATA,
                                    confidence=0.95,
                                    raw_value=value,
                                    notes=f"EXIF field: {tag}"
                                ))
                            except ValueError:
                                pass

        except ImportError:
            logger.debug("PIL not available for EXIF extraction")
        except Exception as e:
            logger.debug(f"No EXIF data found or error: {e}")

        return extractions

    def extract_all(self, file_path: Path) -> List[TimestampExtraction]:
        """
        Extract timestamps from all available sources

        Args:
            file_path: Path to file

        Returns:
            List of all extracted timestamps, sorted by confidence
        """
        all_extractions = []
        
        # Filename patterns
        all_extractions.extend(self.extract_from_filename(file_path.name))
        
        # File system stats
        all_extractions.extend(self.extract_from_file_stats(file_path))
        
        # Audio metadata (if applicable)
        if file_path.suffix.lower() in ['.mp3', '.wav', '.ogg', '.opus', '.m4a', '.flac']:
            all_extractions.extend(self.extract_from_audio_metadata(file_path))
        
        # EXIF data (if applicable)
        if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.heic', '.mp4', '.mov']:
            all_extractions.extend(self.extract_from_exif(file_path))
        
        # Sort by confidence
        all_extractions.sort(key=lambda x: x.confidence, reverse=True)
        
        return all_extractions


class TimelineReconstructor:
    """
    Reconstruct chronological timeline from multiple sources
    WITHOUT requiring main chat text file
    """

    def __init__(self):
        """Initialize timeline reconstructor"""
        self.extractor = TimestampExtractor()
        self.entries: List[TimelineEntry] = []
        self.entry_counter = 0

    def add_file(
        self,
        file_path: Path,
        speaker_id: Optional[str] = None,
        transcription: Optional[str] = None,
        translation: Optional[str] = None
    ) -> TimelineEntry:
        """
        Add file to timeline

        Args:
            file_path: Path to file
            speaker_id: Speaker identifier
            transcription: Transcription text
            translation: Translation text

        Returns:
            Created timeline entry
        """
        self.entry_counter += 1
        entry_id = f"TL_{self.entry_counter:06d}"
        
        # Extract timestamps
        extractions = self.extractor.extract_all(file_path)
        
        # Determine best timestamp
        if extractions:
            best_timestamp = extractions[0].timestamp
            confidence = self._calculate_confidence(extractions)
        else:
            best_timestamp = datetime.now()
            confidence = TimelineConfidence.UNCERTAIN
            extractions = [TimestampExtraction(
                timestamp=best_timestamp,
                source=TimestampSource.INFERRED,
                confidence=0.1,
                raw_value="",
                notes="No timestamp source found"
            )]

        # Determine content type
        suffix = file_path.suffix.lower()
        if suffix in ['.mp3', '.wav', '.ogg', '.opus', '.m4a', '.flac']:
            content_type = 'audio'
        elif suffix in ['.jpg', '.jpeg', '.png', '.heic', '.gif']:
            content_type = 'image'
        elif suffix in ['.mp4', '.mov', '.avi', '.3gp']:
            content_type = 'video'
        else:
            content_type = 'document'

        # Get duration for audio/video
        duration = self._get_media_duration(file_path) if content_type in ['audio', 'video'] else None

        entry = TimelineEntry(
            entry_id=entry_id,
            timestamp=best_timestamp,
            timestamp_confidence=confidence,
            sources=extractions,
            content_type=content_type,
            file_path=str(file_path),
            speaker_id=speaker_id,
            transcription=transcription,
            translation=translation,
            duration=duration,
            metadata={
                'filename': file_path.name,
                'extension': file_path.suffix,
                'size_bytes': file_path.stat().st_size if file_path.exists() else 0
            }
        )

        self.entries.append(entry)
        return entry

    def add_files_from_directory(
        self,
        directory: Path,
        extensions: List[str] = None
    ) -> List[TimelineEntry]:
        """
        Add all files from directory to timeline

        Args:
            directory: Directory to scan
            extensions: File extensions to include

        Returns:
            List of created timeline entries
        """
        if extensions is None:
            extensions = ['.mp3', '.wav', '.ogg', '.opus', '.m4a', '.jpg', '.jpeg', '.png', '.mp4']

        entries = []
        
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                entry = self.add_file(file_path)
                entries.append(entry)

        return entries

    def _calculate_confidence(self, extractions: List[TimestampExtraction]) -> TimelineConfidence:
        """Calculate overall timestamp confidence"""
        if not extractions:
            return TimelineConfidence.UNCERTAIN

        # Check for agreement between sources
        if len(extractions) >= 2:
            # Check if top sources agree (within 1 day)
            if abs((extractions[0].timestamp - extractions[1].timestamp).days) <= 1:
                if extractions[0].confidence >= 0.9:
                    return TimelineConfidence.CERTAIN
                else:
                    return TimelineConfidence.HIGH

        # Single source confidence
        top_confidence = extractions[0].confidence
        if top_confidence >= 0.90:
            return TimelineConfidence.HIGH
        elif top_confidence >= 0.70:
            return TimelineConfidence.MEDIUM
        elif top_confidence >= 0.50:
            return TimelineConfidence.LOW
        else:
            return TimelineConfidence.UNCERTAIN

    def _get_media_duration(self, file_path: Path) -> Optional[float]:
        """Get duration of media file"""
        try:
            from mutagen import File as MutagenFile
            audio = MutagenFile(str(file_path))
            if audio and hasattr(audio, 'info') and hasattr(audio.info, 'length'):
                return audio.info.length
        except Exception:
            pass
        
        try:
            import librosa
            duration = librosa.get_duration(path=str(file_path))
            return duration
        except Exception:
            pass

        return None

    def get_sorted_timeline(self) -> List[TimelineEntry]:
        """Get timeline sorted chronologically"""
        return sorted(self.entries, key=lambda x: x.timestamp)

    def get_timeline_by_speaker(self, speaker_id: str) -> List[TimelineEntry]:
        """Get timeline entries for specific speaker"""
        return sorted(
            [e for e in self.entries if e.speaker_id == speaker_id],
            key=lambda x: x.timestamp
        )

    def get_timeline_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[TimelineEntry]:
        """Get timeline entries within date range"""
        return sorted(
            [e for e in self.entries if start_date <= e.timestamp <= end_date],
            key=lambda x: x.timestamp
        )

    def cross_reference_timestamps(self) -> Dict[str, Any]:
        """
        Cross-reference timestamps to improve accuracy

        Returns:
            Analysis of timestamp patterns and adjustments
        """
        sorted_entries = self.get_sorted_timeline()
        
        if len(sorted_entries) < 2:
            return {'adjustments': [], 'patterns': []}

        adjustments = []
        patterns = []

        # Look for patterns in timing
        gaps = []
        for i in range(1, len(sorted_entries)):
            gap = (sorted_entries[i].timestamp - sorted_entries[i-1].timestamp).total_seconds()
            gaps.append({
                'from': sorted_entries[i-1].entry_id,
                'to': sorted_entries[i].entry_id,
                'gap_seconds': gap
            })

        # Identify unusual gaps
        if gaps:
            avg_gap = sum(g['gap_seconds'] for g in gaps) / len(gaps)
            for gap in gaps:
                if gap['gap_seconds'] > avg_gap * 10:
                    patterns.append({
                        'type': 'large_gap',
                        'gap': gap,
                        'note': 'Unusually large time gap detected'
                    })

        # Look for timestamp clustering (multiple files at same time)
        timestamp_groups = {}
        for entry in sorted_entries:
            date_key = entry.timestamp.strftime('%Y-%m-%d %H')
            if date_key not in timestamp_groups:
                timestamp_groups[date_key] = []
            timestamp_groups[date_key].append(entry.entry_id)

        for date_key, entries in timestamp_groups.items():
            if len(entries) > 5:
                patterns.append({
                    'type': 'cluster',
                    'date': date_key,
                    'count': len(entries),
                    'entries': entries,
                    'note': 'Multiple files from same hour'
                })

        return {
            'adjustments': adjustments,
            'patterns': patterns,
            'gaps': gaps,
            'total_entries': len(sorted_entries)
        }

    def infer_missing_timestamps(self) -> int:
        """
        Infer timestamps for entries with low confidence

        Returns:
            Number of entries updated
        """
        sorted_entries = self.get_sorted_timeline()
        updated = 0

        for i, entry in enumerate(sorted_entries):
            if entry.timestamp_confidence in [TimelineConfidence.UNCERTAIN, TimelineConfidence.LOW]:
                # Try to infer from neighbors
                prev_entry = sorted_entries[i-1] if i > 0 else None
                next_entry = sorted_entries[i+1] if i < len(sorted_entries) - 1 else None

                if prev_entry and next_entry:
                    # Interpolate between neighbors
                    prev_ts = prev_entry.timestamp
                    next_ts = next_entry.timestamp
                    
                    if (next_ts - prev_ts).total_seconds() < 86400:  # Within 1 day
                        # Place in middle
                        new_ts = prev_ts + (next_ts - prev_ts) / 2
                        entry.timestamp = new_ts
                        entry.sources.append(TimestampExtraction(
                            timestamp=new_ts,
                            source=TimestampSource.INFERRED,
                            confidence=0.5,
                            raw_value="",
                            notes="Inferred from neighboring entries"
                        ))
                        entry.timestamp_confidence = TimelineConfidence.LOW
                        updated += 1

        return updated

    def export_timeline(self, format: str = 'json') -> str:
        """
        Export timeline in specified format

        Args:
            format: Export format (json, csv, markdown)

        Returns:
            Exported timeline string
        """
        sorted_entries = self.get_sorted_timeline()

        if format == 'json':
            return json.dumps(
                [e.to_dict() for e in sorted_entries],
                indent=2,
                default=str
            )

        elif format == 'csv':
            lines = ['entry_id,timestamp,confidence,content_type,speaker_id,file_path,transcription']
            for entry in sorted_entries:
                lines.append(
                    f'{entry.entry_id},{entry.timestamp.isoformat()},'
                    f'{entry.timestamp_confidence.value},{entry.content_type},'
                    f'{entry.speaker_id or ""},'
                    f'"{entry.file_path or ""}",'
                    f'"{(entry.transcription or "").replace(chr(34), chr(39))}"'
                )
            return '\n'.join(lines)

        elif format == 'markdown':
            lines = ['# Reconstructed Timeline\n']
            lines.append(f'Generated: {datetime.now().isoformat()}\n')
            lines.append(f'Total Entries: {len(sorted_entries)}\n')
            lines.append('---\n')

            current_date = None
            for entry in sorted_entries:
                entry_date = entry.timestamp.strftime('%Y-%m-%d')
                
                if entry_date != current_date:
                    lines.append(f'\n## {entry_date}\n')
                    current_date = entry_date

                time_str = entry.timestamp.strftime('%H:%M:%S')
                confidence = entry.timestamp_confidence.value
                
                lines.append(f'### {time_str} [{confidence}]')
                lines.append(f'- **Type**: {entry.content_type}')
                if entry.speaker_id:
                    lines.append(f'- **Speaker**: {entry.speaker_id}')
                if entry.duration:
                    lines.append(f'- **Duration**: {entry.duration:.1f}s')
                if entry.transcription:
                    lines.append(f'- **Transcription**: {entry.transcription}')
                if entry.translation:
                    lines.append(f'- **Translation**: {entry.translation}')
                lines.append('')

            return '\n'.join(lines)

        return ""

    def get_timeline_statistics(self) -> Dict[str, Any]:
        """Get statistics about the timeline"""
        if not self.entries:
            return {'total_entries': 0}

        sorted_entries = self.get_sorted_timeline()
        
        # Calculate statistics
        by_type = {}
        by_speaker = {}
        by_confidence = {}
        total_duration = 0

        for entry in sorted_entries:
            # By content type
            by_type[entry.content_type] = by_type.get(entry.content_type, 0) + 1
            
            # By speaker
            speaker = entry.speaker_id or 'unknown'
            by_speaker[speaker] = by_speaker.get(speaker, 0) + 1
            
            # By confidence
            conf = entry.timestamp_confidence.value
            by_confidence[conf] = by_confidence.get(conf, 0) + 1
            
            # Duration
            if entry.duration:
                total_duration += entry.duration

        return {
            'total_entries': len(sorted_entries),
            'date_range': {
                'start': sorted_entries[0].timestamp.isoformat(),
                'end': sorted_entries[-1].timestamp.isoformat(),
                'span_days': (sorted_entries[-1].timestamp - sorted_entries[0].timestamp).days
            },
            'by_content_type': by_type,
            'by_speaker': by_speaker,
            'by_confidence': by_confidence,
            'total_duration_seconds': total_duration,
            'total_duration_formatted': str(timedelta(seconds=int(total_duration)))
        }


if __name__ == "__main__":
    reconstructor = TimelineReconstructor()
    print("Timeline Reconstructor initialized")
    print("Use reconstructor.add_file(path) or reconstructor.add_files_from_directory(path)")
