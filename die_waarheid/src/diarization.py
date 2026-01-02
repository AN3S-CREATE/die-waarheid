"""
Speaker Diarization for Die Waarheid
Identifies and separates different speakers in audio files
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


class SpeakerSegment:
    """Represents a speaker segment in audio"""

    def __init__(self, start: float, end: float, speaker: str, confidence: float = 1.0):
        """
        Initialize speaker segment

        Args:
            start: Start time in seconds
            end: End time in seconds
            speaker: Speaker identifier
            confidence: Confidence score (0-1)
        """
        self.start = start
        self.end = end
        self.speaker = speaker
        self.confidence = confidence
        self.duration = end - start

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'start': self.start,
            'end': self.end,
            'duration': self.duration,
            'speaker': self.speaker,
            'confidence': self.confidence
        }


class SimpleDiarizer:
    """
    Simple speaker diarization using audio features
    Fallback implementation when pyannote is not available
    """

    def __init__(self, sample_rate: int = 16000):
        """
        Initialize diarizer

        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.segments = []

    def detect_speaker_changes(self, audio: np.ndarray, threshold: float = 0.5) -> List[int]:
        """
        Detect speaker changes using energy-based segmentation

        Args:
            audio: Audio signal
            threshold: Energy threshold for change detection

        Returns:
            List of frame indices where speaker changes occur
        """
        try:
            frame_length = int(0.025 * self.sample_rate)
            hop_length = int(0.010 * self.sample_rate)

            energy = []
            for i in range(0, len(audio) - frame_length, hop_length):
                frame = audio[i:i + frame_length]
                energy.append(np.sum(frame ** 2))

            energy = np.array(energy)
            if len(energy) == 0:
                return []

            energy_normalized = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-10)

            changes = []
            for i in range(1, len(energy_normalized)):
                if abs(energy_normalized[i] - energy_normalized[i-1]) > threshold:
                    changes.append(i)

            return changes

        except Exception as e:
            logger.error(f"Error detecting speaker changes: {str(e)}")
            return []

    def diarize(self, audio: np.ndarray, num_speakers: int = 2) -> List[SpeakerSegment]:
        """
        Perform speaker diarization using simple energy-based segmentation

        Args:
            audio: Audio signal
            num_speakers: Expected number of speakers

        Returns:
            List of speaker segments
        """
        try:
            changes = self.detect_speaker_changes(audio)

            if not changes:
                logger.warning("No speaker changes detected, assuming single speaker")
                duration = len(audio) / self.sample_rate
                return [SpeakerSegment(0, duration, "SPEAKER_00", 1.0)]

            frame_length = int(0.025 * self.sample_rate)
            hop_length = int(0.010 * self.sample_rate)

            segments = []
            current_speaker = 0
            start_time = 0.0

            for change_idx in changes:
                end_time = (change_idx * hop_length) / self.sample_rate
                speaker_id = f"SPEAKER_{current_speaker:02d}"
                segments.append(SpeakerSegment(start_time, end_time, speaker_id))

                current_speaker = (current_speaker + 1) % num_speakers
                start_time = end_time

            duration = len(audio) / self.sample_rate
            if start_time < duration:
                speaker_id = f"SPEAKER_{current_speaker:02d}"
                segments.append(SpeakerSegment(start_time, duration, speaker_id))

            self.segments = segments
            logger.info(f"Diarization complete: {len(segments)} segments detected")
            return segments

        except Exception as e:
            logger.error(f"Error during diarization: {str(e)}")
            return []

    def get_segments_for_time(self, time: float) -> Optional[SpeakerSegment]:
        """
        Get speaker segment for a specific time

        Args:
            time: Time in seconds

        Returns:
            Speaker segment or None
        """
        for segment in self.segments:
            if segment.start <= time <= segment.end:
                return segment
        return None

    def get_speaker_timeline(self) -> List[Dict]:
        """
        Get speaker timeline

        Returns:
            List of speaker segments as dictionaries
        """
        return [seg.to_dict() for seg in self.segments]

    def merge_adjacent_speakers(self, min_duration: float = 0.5) -> List[SpeakerSegment]:
        """
        Merge adjacent segments from same speaker

        Args:
            min_duration: Minimum segment duration to keep

        Returns:
            Merged segments
        """
        if not self.segments:
            return []

        merged = []
        current_segment = self.segments[0]

        for next_segment in self.segments[1:]:
            if next_segment.speaker == current_segment.speaker:
                current_segment.end = next_segment.end
                current_segment.duration = current_segment.end - current_segment.start
            else:
                if current_segment.duration >= min_duration:
                    merged.append(current_segment)
                current_segment = next_segment

        if current_segment.duration >= min_duration:
            merged.append(current_segment)

        self.segments = merged
        logger.info(f"Merged segments: {len(self.segments)} remaining")
        return merged


class DiarizationPipeline:
    """
    Complete diarization pipeline
    Handles diarization with fallback to simple method
    """

    def __init__(self, sample_rate: int = 16000, use_advanced: bool = False):
        """
        Initialize pipeline

        Args:
            sample_rate: Audio sample rate
            use_advanced: Try to use advanced diarization (pyannote)
        """
        self.sample_rate = sample_rate
        self.use_advanced = use_advanced
        self.diarizer = None
        self.initialize()

    def initialize(self) -> bool:
        """
        Initialize diarization backend

        Returns:
            True if successful
        """
        try:
            if self.use_advanced:
                try:
                    from pyannote.audio import Pipeline
                    self.diarizer = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1"
                    )
                    logger.info("Initialized advanced diarization (pyannote)")
                    return True
                except ImportError:
                    logger.warning("pyannote not available, using simple diarization")
                    self.diarizer = SimpleDiarizer(self.sample_rate)
                    return True
            else:
                self.diarizer = SimpleDiarizer(self.sample_rate)
                logger.info("Initialized simple diarization")
                return True

        except Exception as e:
            logger.error(f"Error initializing diarization: {str(e)}")
            self.diarizer = SimpleDiarizer(self.sample_rate)
            return True

    def diarize(self, audio: np.ndarray, num_speakers: int = 2) -> List[Dict]:
        """
        Perform diarization

        Args:
            audio: Audio signal
            num_speakers: Expected number of speakers

        Returns:
            List of speaker segments
        """
        try:
            if self.diarizer is None:
                self.initialize()

            if isinstance(self.diarizer, SimpleDiarizer):
                segments = self.diarizer.diarize(audio, num_speakers)
            else:
                segments = self.diarizer(audio)

            return [seg.to_dict() if hasattr(seg, 'to_dict') else seg for seg in segments]

        except Exception as e:
            logger.error(f"Error during diarization: {str(e)}")
            duration = len(audio) / self.sample_rate
            return [SpeakerSegment(0, duration, "SPEAKER_00").to_dict()]

    def get_speaker_count(self, segments: List[Dict]) -> int:
        """
        Get number of unique speakers

        Args:
            segments: List of speaker segments

        Returns:
            Number of unique speakers
        """
        speakers = set(seg.get('speaker') for seg in segments)
        return len(speakers)

    def get_speaker_statistics(self, segments: List[Dict]) -> Dict:
        """
        Get speaker statistics

        Args:
            segments: List of speaker segments

        Returns:
            Dictionary with speaker statistics
        """
        stats = {}
        total_duration = 0

        for segment in segments:
            speaker = segment.get('speaker')
            duration = segment.get('duration', 0)
            total_duration += duration

            if speaker not in stats:
                stats[speaker] = {
                    'total_duration': 0,
                    'segment_count': 0,
                    'average_segment_duration': 0
                }

            stats[speaker]['total_duration'] += duration
            stats[speaker]['segment_count'] += 1

        for speaker in stats:
            if stats[speaker]['segment_count'] > 0:
                stats[speaker]['average_segment_duration'] = (
                    stats[speaker]['total_duration'] / stats[speaker]['segment_count']
                )
            stats[speaker]['percentage'] = (
                stats[speaker]['total_duration'] / total_duration * 100
                if total_duration > 0 else 0
            )

        return {
            'total_duration': total_duration,
            'speaker_count': len(stats),
            'speakers': stats
        }


if __name__ == "__main__":
    diarizer = SimpleDiarizer()
    print("Diarization module initialized")
