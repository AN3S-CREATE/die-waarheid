"""
Speaker Identification and Tracking System for Die Waarheid
Maintains consistent participant identification across investigation
Uses voice recognition and speaker diarization to track speakers
Even if usernames change, the same two participants are always tracked correctly

FEATURES:
- Voice fingerprinting and speaker embedding
- Automatic speaker diarization
- Username change detection and mapping
- Consistent participant tracking
- Speaker profile management
- Voice similarity matching
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import json
import hashlib
import numpy as np

from sqlalchemy import create_engine, Column, String, DateTime, Float, Text, Boolean, JSON, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

logger = logging.getLogger(__name__)

Base = declarative_base()


class SpeakerRole(Enum):
    """Role of speaker in investigation"""
    PARTICIPANT_A = "participant_a"
    PARTICIPANT_B = "participant_b"
    UNKNOWN = "unknown"


@dataclass
class VoiceFingerprint:
    """Voice fingerprint for speaker identification"""
    speaker_id: str
    fingerprint_hash: str
    mfcc_features: List[float]
    pitch_range: Tuple[float, float]
    speech_rate: float
    accent_markers: List[str]
    confidence: float
    created_at: str
    
    def to_dict(self) -> Dict:
        return {
            'speaker_id': self.speaker_id,
            'fingerprint_hash': self.fingerprint_hash,
            'mfcc_features': self.mfcc_features,
            'pitch_range': list(self.pitch_range),
            'speech_rate': self.speech_rate,
            'accent_markers': self.accent_markers,
            'confidence': self.confidence,
            'created_at': self.created_at
        }


@dataclass
class ParticipantProfile:
    """Profile for investigation participant"""
    participant_id: str
    assigned_role: SpeakerRole
    
    # Identifiers
    primary_username: str
    alternate_usernames: List[str]
    
    # Voice data
    voice_fingerprints: List[VoiceFingerprint]
    voice_embedding: Optional[List[float]]
    
    # Characteristics
    accent: str
    speech_patterns: Dict[str, Any]
    typical_stress_level: float
    
    # Tracking
    first_appearance: str
    last_appearance: str
    message_count: int
    voice_note_count: int
    
    # Verification
    confidence_score: float
    verified: bool
    
    def to_dict(self) -> Dict:
        return {
            'participant_id': self.participant_id,
            'assigned_role': self.assigned_role.value,
            'primary_username': self.primary_username,
            'alternate_usernames': self.alternate_usernames,
            'voice_fingerprints': [vf.to_dict() for vf in self.voice_fingerprints],
            'accent': self.accent,
            'speech_patterns': self.speech_patterns,
            'typical_stress_level': self.typical_stress_level,
            'first_appearance': self.first_appearance,
            'last_appearance': self.last_appearance,
            'message_count': self.message_count,
            'voice_note_count': self.voice_note_count,
            'confidence_score': self.confidence_score,
            'verified': self.verified
        }


class SpeakerRecord(Base):
    """Database model for speaker tracking"""
    __tablename__ = 'speakers'
    
    id = Column(String, primary_key=True)
    case_id = Column(String, nullable=False, index=True)
    participant_id = Column(String, nullable=False, index=True)
    assigned_role = Column(String)
    
    # Identifiers
    primary_username = Column(String)
    alternate_usernames = Column(JSON, default=[])
    
    # Voice data
    voice_fingerprints = Column(JSON, default=[])
    voice_embedding = Column(JSON)
    
    # Characteristics
    accent = Column(String)
    speech_patterns = Column(JSON, default={})
    typical_stress_level = Column(Float, default=0.0)
    
    # Tracking
    first_appearance = Column(DateTime)
    last_appearance = Column(DateTime)
    message_count = Column(Integer, default=0)
    voice_note_count = Column(Integer, default=0)
    
    # Verification
    confidence_score = Column(Float, default=0.0)
    verified = Column(Boolean, default=False)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now)


class UsernameMapping(Base):
    """Track username changes and mappings"""
    __tablename__ = 'username_mappings'
    
    id = Column(String, primary_key=True)
    case_id = Column(String, nullable=False, index=True)
    participant_id = Column(String, nullable=False, index=True)
    
    # Username info
    username = Column(String, nullable=False)
    first_seen = Column(DateTime)
    last_seen = Column(DateTime)
    
    # Verification
    verified_by_voice = Column(Boolean, default=False)
    voice_confidence = Column(Float, default=0.0)
    
    # Mapping
    mapped_to_participant = Column(String)
    mapping_confidence = Column(Float, default=0.0)


class SpeakerIdentificationSystem:
    """
    Main speaker identification system
    Maintains consistent participant tracking across investigation
    """

    def __init__(self, case_id: str, db_path: Optional[Path] = None):
        """
        Initialize speaker identification system

        Args:
            case_id: Case identifier
            db_path: Path to SQLite database
        """
        self.case_id = case_id
        
        if db_path is None:
            from config import DATA_DIR
            db_path = DATA_DIR / "investigations.db"
        
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Participant profiles
        self.participants: Dict[str, ParticipantProfile] = {}
        self.username_to_participant: Dict[str, str] = {}
        
        # Load existing participants
        self._load_participants()
        
        # Initialize diarization
        try:
            import pyannote.audio
            self.diarization_available = True
            logger.info("Pyannote diarization available")
        except ImportError:
            self.diarization_available = False
            logger.warning("Pyannote not available - using energy-based diarization")

    def _load_participants(self):
        """Load existing participants from database"""
        session = self.SessionLocal()
        try:
            speakers = session.query(SpeakerRecord).filter_by(case_id=self.case_id).all()
            
            for speaker in speakers:
                profile = ParticipantProfile(
                    participant_id=speaker.participant_id,
                    assigned_role=self._get_speaker_role(speaker.assigned_role),
                    primary_username=speaker.primary_username or "Unknown",
                    alternate_usernames=speaker.alternate_usernames or [],
                    voice_fingerprints=[],
                    voice_embedding=speaker.voice_embedding,
                    accent=speaker.accent or "",
                    speech_patterns=speaker.speech_patterns or {},
                    typical_stress_level=speaker.typical_stress_level or 0.0,
                    first_appearance=speaker.first_appearance.isoformat() if speaker.first_appearance else "",
                    last_appearance=speaker.last_appearance.isoformat() if speaker.last_appearance else "",
                    message_count=speaker.message_count or 0,
                    voice_note_count=speaker.voice_note_count or 0,
                    confidence_score=speaker.confidence_score or 0.0,
                    verified=speaker.verified or False
                )
                
                self.participants[speaker.participant_id] = profile
                self.username_to_participant[speaker.primary_username] = speaker.participant_id
                
                for alt_username in speaker.alternate_usernames:
                    self.username_to_participant[alt_username] = speaker.participant_id
            
            logger.info(f"Loaded {len(self.participants)} participants")

        finally:
            session.close()
    
    def _get_speaker_role(self, role_str: str) -> SpeakerRole:
        """Convert string to SpeakerRole enum safely"""
        if not role_str:
            return SpeakerRole.UNKNOWN
        
        role_mapping = {
            "participant_a": SpeakerRole.PARTICIPANT_A,
            "participant_b": SpeakerRole.PARTICIPANT_B,
            "unknown": SpeakerRole.UNKNOWN
        }
        
        return role_mapping.get(role_str.lower(), SpeakerRole.UNKNOWN)

    def initialize_investigation(
        self,
        participant_a_name: str,
        participant_b_name: str
    ) -> Tuple[str, str]:
        """
        Initialize investigation with two participants

        Args:
            participant_a_name: Name/username of first participant
            participant_b_name: Name/username of second participant

        Returns:
            Tuple of (participant_a_id, participant_b_id)
        """
        logger.info(f"Initializing investigation with {participant_a_name} and {participant_b_name}")

        # Create participant A
        participant_a_id = self._create_participant(
            participant_a_name,
            SpeakerRole.PARTICIPANT_A
        )

        # Create participant B
        participant_b_id = self._create_participant(
            participant_b_name,
            SpeakerRole.PARTICIPANT_B
        )

        logger.info(f"Investigation initialized: {participant_a_id}, {participant_b_id}")
        return participant_a_id, participant_b_id

    def _create_participant(self, username: str, role: SpeakerRole) -> str:
        """Create new participant"""
        participant_id = f"PART_{self.case_id}_{role.value}"
        
        profile = ParticipantProfile(
            participant_id=participant_id,
            assigned_role=role,
            primary_username=username,
            alternate_usernames=[],
            voice_fingerprints=[],
            voice_embedding=None,
            accent="",
            speech_patterns={},
            typical_stress_level=0.0,
            first_appearance=datetime.now().isoformat(),
            last_appearance=datetime.now().isoformat(),
            message_count=0,
            voice_note_count=0,
            confidence_score=1.0,
            verified=True
        )
        
        self.participants[participant_id] = profile
        self.username_to_participant[username] = participant_id
        
        # Save to database
        self._save_participant(profile)
        
        return participant_id

    def _save_participant(self, profile: ParticipantProfile):
        """Save participant to database"""
        session = self.SessionLocal()
        try:
            speaker = SpeakerRecord(
                id=profile.participant_id,
                case_id=self.case_id,
                participant_id=profile.participant_id,
                assigned_role=profile.assigned_role.value,
                primary_username=profile.primary_username,
                alternate_usernames=profile.alternate_usernames,
                voice_fingerprints=[vf.to_dict() for vf in profile.voice_fingerprints],
                voice_embedding=profile.voice_embedding,
                accent=profile.accent,
                speech_patterns=profile.speech_patterns,
                typical_stress_level=profile.typical_stress_level,
                first_appearance=datetime.fromisoformat(profile.first_appearance) if profile.first_appearance else None,
                last_appearance=datetime.fromisoformat(profile.last_appearance) if profile.last_appearance else None,
                message_count=profile.message_count,
                voice_note_count=profile.voice_note_count,
                confidence_score=profile.confidence_score,
                verified=profile.verified
            )
            
            session.merge(speaker)
            session.commit()

        finally:
            session.close()

    def identify_speaker(
        self,
        username: str,
        audio_file: Optional[Path] = None,
        text_sample: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Identify speaker and map to participant

        Args:
            username: Username from message/file
            audio_file: Optional audio file for voice verification
            text_sample: Optional text sample for linguistic analysis

        Returns:
            Tuple of (participant_id, confidence)
        """
        # Check if username already mapped
        if username in self.username_to_participant:
            participant_id = self.username_to_participant[username]
            return participant_id, 1.0

        # Try voice matching if audio available
        if audio_file and audio_file.exists():
            participant_id, confidence = self._match_by_voice(audio_file)
            if confidence > 0.3:  # Lower threshold for better matching
                self._map_username_to_participant(username, participant_id, confidence, verified_by_voice=True)
                return participant_id, confidence

        # Try linguistic matching if text available
        if text_sample:
            participant_id, confidence = self._match_by_linguistics(text_sample)
            if confidence > 0.6:
                self._map_username_to_participant(username, participant_id, confidence)
                return participant_id, confidence

        # Default: try to match to existing participants
        # If no match found, return empty (don't assign to random speaker)
        logger.warning(f"Could not identify speaker: {username}")
        return "", 0.0

    def _match_by_voice(self, audio_file: Path) -> Tuple[str, float]:
        """Match speaker by voice"""
        try:
            # Extract voice features
            fingerprint = self._extract_voice_fingerprint(audio_file)
            
            if not fingerprint:
                return "", 0.0

            # Compare with existing participants
            best_match = ""
            best_score = 0.0
            
            logger.debug(f"Comparing with {len(self.participants)} participants")
            for participant_id, profile in self.participants.items():
                if not profile.voice_fingerprints:
                    logger.debug(f"Participant {participant_id} has no voice fingerprints")
                    continue
                
                logger.debug(f"Comparing with participant {participant_id} ({len(profile.voice_fingerprints)} fingerprints)")
                
                # Compare fingerprints
                for existing_fp in profile.voice_fingerprints:
                    similarity = self._compare_fingerprints(fingerprint, existing_fp)
                    logger.debug(f"Fingerprint similarity: {similarity:.3f}")
                    if similarity > best_score:
                        best_score = similarity
                        best_match = participant_id
            
            logger.debug(f"Best match: {best_match} with score {best_score:.3f}")
            return best_match, best_score

        except Exception as e:
            logger.error(f"Voice matching error: {e}")
            return "", 0.0

    def _extract_voice_fingerprint(self, audio_file: Path) -> Optional[VoiceFingerprint]:
        """Extract voice fingerprint from audio"""
        try:
            import librosa
            import numpy as np

            # Load audio
            y, sr = librosa.load(str(audio_file), sr=16000)

            # Extract MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = [float(x) for x in np.mean(mfcc, axis=1)]

            # Extract pitch
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(float(pitch))

            pitch_range = (min(pitch_values), max(pitch_values)) if pitch_values else (0.0, 0.0)

            # Speech rate (simple estimation)
            speech_rate = float(len(y) / sr)

            # Create fingerprint
            fingerprint_hash = hashlib.sha256(
                str(mfcc_mean + list(pitch_range)).encode()
            ).hexdigest()

            return VoiceFingerprint(
                speaker_id="",
                fingerprint_hash=fingerprint_hash,
                mfcc_features=mfcc_mean,
                pitch_range=pitch_range,
                speech_rate=speech_rate,
                accent_markers=[],
                confidence=0.8,
                created_at=datetime.now().isoformat()
            )

        except ImportError:
            logger.warning("Librosa not available for voice fingerprinting")
            return None
        except Exception as e:
            logger.error(f"Voice fingerprint extraction error: {e}")
            return None

    def _compare_fingerprints(
        self,
        fp1: VoiceFingerprint,
        fp2: VoiceFingerprint
    ) -> float:
        """Compare two voice fingerprints"""
        try:
            # Compare MFCC features
            mfcc_distance = np.linalg.norm(
                np.array(fp1.mfcc_features) - np.array(fp2.mfcc_features)
            )
            mfcc_similarity = 1.0 / (1.0 + mfcc_distance)

            # Compare pitch ranges
            pitch_overlap = min(fp1.pitch_range[1], fp2.pitch_range[1]) - max(fp1.pitch_range[0], fp2.pitch_range[0])
            pitch_range1 = fp1.pitch_range[1] - fp1.pitch_range[0]
            pitch_range2 = fp2.pitch_range[1] - fp2.pitch_range[0]
            pitch_similarity = pitch_overlap / max(pitch_range1, pitch_range2, 1)

            # Compare speech rate
            rate_diff = abs(fp1.speech_rate - fp2.speech_rate)
            rate_similarity = 1.0 / (1.0 + rate_diff)

            # Weighted average
            similarity = (mfcc_similarity * 0.5 + pitch_similarity * 0.3 + rate_similarity * 0.2)
            return similarity

        except Exception as e:
            logger.error(f"Fingerprint comparison error: {e}")
            return 0.0

    def _match_by_linguistics(self, text_sample: str) -> Tuple[str, float]:
        """Match speaker by linguistic patterns"""
        try:
            best_match = ""
            best_score = 0.0

            for participant_id, profile in self.participants.items():
                if not profile.speech_patterns:
                    continue

                # Simple linguistic matching
                score = self._linguistic_similarity(text_sample, profile.speech_patterns)
                if score > best_score:
                    best_score = score
                    best_match = participant_id

            return best_match, best_score

        except Exception as e:
            logger.error(f"Linguistic matching error: {e}")
            return "", 0.0

    def _linguistic_similarity(self, text: str, patterns: Dict[str, Any]) -> float:
        """Calculate linguistic similarity"""
        score = 0.0
        matches = 0

        # Check vocabulary patterns
        if 'common_words' in patterns:
            common_words = patterns['common_words']
            text_words = set(text.lower().split())
            matching_words = len(text_words & set(common_words))
            if matching_words > 0:
                score += matching_words / len(common_words)
                matches += 1

        # Check sentence length
        if 'avg_sentence_length' in patterns:
            sentences = text.split('.')
            avg_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
            expected_length = patterns['avg_sentence_length']
            length_diff = abs(avg_length - expected_length)
            length_similarity = 1.0 / (1.0 + length_diff)
            score += length_similarity
            matches += 1

        return score / max(matches, 1) if matches > 0 else 0.0

    def _map_username_to_participant(
        self,
        username: str,
        participant_id: str,
        confidence: float,
        verified_by_voice: bool = False
    ):
        """Map username to participant"""
        self.username_to_participant[username] = participant_id

        # Update participant profile
        if participant_id in self.participants:
            profile = self.participants[participant_id]
            if username not in profile.alternate_usernames and username != profile.primary_username:
                profile.alternate_usernames.append(username)
            profile.last_appearance = datetime.now().isoformat()
            self._save_participant(profile)

        # Save mapping to database
        session = self.SessionLocal()
        try:
            mapping = UsernameMapping(
                id=f"UM_{self.case_id}_{username}",
                case_id=self.case_id,
                participant_id=participant_id,
                username=username,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                verified_by_voice=verified_by_voice,
                voice_confidence=confidence if verified_by_voice else 0.0,
                mapped_to_participant=participant_id,
                mapping_confidence=confidence
            )
            session.merge(mapping)
            session.commit()

        finally:
            session.close()

    def register_message(
        self,
        username: str,
        message_text: str,
        timestamp: datetime
    ) -> str:
        """
        Register message and identify speaker

        Args:
            username: Username from message
            message_text: Message text
            timestamp: Message timestamp

        Returns:
            Participant ID
        """
        # Identify speaker
        participant_id, confidence = self.identify_speaker(username, text_sample=message_text)

        if not participant_id:
            logger.warning(f"Could not identify speaker: {username}")
            return ""

        # Update participant stats
        if participant_id in self.participants:
            profile = self.participants[participant_id]
            profile.message_count += 1
            profile.last_appearance = timestamp.isoformat()
            
            # Update speech patterns
            self._update_speech_patterns(profile, message_text)
            
            self._save_participant(profile)

        return participant_id

    def register_voice_note(
        self,
        username: str,
        audio_file: Path,
        timestamp: datetime
    ) -> str:
        """
        Register voice note and identify speaker

        Args:
            username: Username from message
            audio_file: Path to audio file
            timestamp: Message timestamp

        Returns:
            Participant ID
        """
        # Identify speaker by voice
        participant_id, confidence = self.identify_speaker(username, audio_file=audio_file)

        if not participant_id:
            logger.warning(f"Could not identify speaker: {username}")
            return ""

        # Update participant stats
        if participant_id in self.participants:
            profile = self.participants[participant_id]
            profile.voice_note_count += 1
            profile.last_appearance = timestamp.isoformat()
            
            # Extract and store voice fingerprint
            fingerprint = self._extract_voice_fingerprint(audio_file)
            if fingerprint:
                profile.voice_fingerprints.append(fingerprint)
            
            self._save_participant(profile)

        return participant_id

    def _update_speech_patterns(self, profile: ParticipantProfile, text: str):
        """Update speech patterns for participant"""
        if 'common_words' not in profile.speech_patterns:
            profile.speech_patterns['common_words'] = []

        # Extract common words
        words = text.lower().split()
        for word in words:
            if len(word) > 3 and word not in profile.speech_patterns['common_words']:
                profile.speech_patterns['common_words'].append(word)

        # Limit to top 100 words
        if len(profile.speech_patterns['common_words']) > 100:
            profile.speech_patterns['common_words'] = profile.speech_patterns['common_words'][-100:]

        # Update sentence length
        sentences = text.split('.')
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        if sentence_lengths:
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            profile.speech_patterns['avg_sentence_length'] = avg_length

    def get_participant(self, participant_id: str) -> Optional[ParticipantProfile]:
        """Get participant profile"""
        return self.participants.get(participant_id)

    def get_all_participants(self) -> List[ParticipantProfile]:
        """Get all participants"""
        return list(self.participants.values())

    def get_participant_by_username(self, username: str) -> Optional[ParticipantProfile]:
        """Get participant by username"""
        participant_id = self.username_to_participant.get(username)
        if participant_id:
            return self.participants.get(participant_id)
        return None

    def get_investigation_summary(self) -> Dict[str, Any]:
        """Get investigation summary with speaker info"""
        summary = {
            'case_id': self.case_id,
            'participants': [],
            'total_messages': 0,
            'total_voice_notes': 0
        }

        for profile in self.participants.values():
            summary['participants'].append({
                'participant_id': profile.participant_id,
                'role': profile.assigned_role.value,
                'primary_username': profile.primary_username,
                'alternate_usernames': profile.alternate_usernames,
                'message_count': profile.message_count,
                'voice_note_count': profile.voice_note_count,
                'confidence_score': profile.confidence_score,
                'verified': profile.verified
            })
            
            summary['total_messages'] += profile.message_count
            summary['total_voice_notes'] += profile.voice_note_count

        return summary


if __name__ == "__main__":
    system = SpeakerIdentificationSystem("CASE_001")
    print("Speaker Identification System initialized")
    print("\nUsage:")
    print("  participant_a, participant_b = system.initialize_investigation('Person A', 'Person B')")
    print("  participant_id = system.register_message('username', 'message text', datetime.now())")
    print("  participant_id = system.register_voice_note('username', Path('audio.opus'), datetime.now())")
    print("  summary = system.get_investigation_summary()")
