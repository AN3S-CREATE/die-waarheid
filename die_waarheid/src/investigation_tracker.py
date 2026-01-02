"""
Continuous Investigation Tracker for Die Waarheid
Persistent storage and incremental analysis as evidence is added over time

FEATURES:
- Persistent SQLite storage of all evidence
- Incremental analysis (only new evidence analyzed)
- Investigation timeline tracking
- Evidence versioning and change history
- Automatic report updates
- Cross-reference analysis across all evidence
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import json
import hashlib

from sqlalchemy import create_engine, Column, String, DateTime, Integer, Float, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

logger = logging.getLogger(__name__)

Base = declarative_base()


class EvidenceType(Enum):
    """Type of evidence"""
    CHAT_EXPORT = "chat_export"
    VOICE_NOTE = "voice_note"
    IMAGE = "image"
    VIDEO = "video"
    DOCUMENT = "document"
    TRANSCRIPTION = "transcription"
    ANALYSIS = "analysis"


class AnalysisStatus(Enum):
    """Status of analysis"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_REVIEW = "requires_review"


class EvidenceRecord(Base):
    """Database model for evidence"""
    __tablename__ = 'evidence'
    
    id = Column(String, primary_key=True)
    case_id = Column(String, nullable=False, index=True)
    evidence_type = Column(String, nullable=False)
    file_path = Column(String)
    file_hash = Column(String, unique=True)
    file_size = Column(Integer)
    
    # Content
    original_content = Column(Text)
    processed_content = Column(Text)
    
    # Metadata
    source_timestamp = Column(DateTime)
    added_timestamp = Column(DateTime, default=datetime.now)
    sender = Column(String)
    duration = Column(Float)
    
    # Analysis
    analysis_status = Column(String, default=AnalysisStatus.PENDING.value)
    analysis_results = Column(JSON)
    forensic_flags = Column(JSON, default=[])
    risk_score = Column(Float, default=0.0)
    
    # Verification
    afrikaans_verified = Column(Boolean, default=False)
    verification_notes = Column(Text)
    confidence = Column(Float, default=0.0)
    
    # Tracking
    version = Column(Integer, default=1)
    is_deleted = Column(Boolean, default=False)
    notes = Column(Text)


class AnalysisUpdate(Base):
    """Track analysis updates over time"""
    __tablename__ = 'analysis_updates'
    
    id = Column(String, primary_key=True)
    case_id = Column(String, nullable=False, index=True)
    evidence_id = Column(String, nullable=False)
    
    # Update info
    update_timestamp = Column(DateTime, default=datetime.now)
    analysis_type = Column(String)
    update_reason = Column(String)
    
    # Results
    previous_results = Column(JSON)
    new_results = Column(JSON)
    changes = Column(JSON)
    
    # Impact
    risk_change = Column(Float)
    new_flags = Column(JSON)
    removed_flags = Column(JSON)


class InvestigationSession(Base):
    """Track investigation sessions"""
    __tablename__ = 'investigation_sessions'
    
    id = Column(String, primary_key=True)
    case_id = Column(String, nullable=False, index=True)
    
    # Session info
    session_start = Column(DateTime, default=datetime.now)
    session_end = Column(DateTime)
    
    # Activity
    evidence_added = Column(Integer, default=0)
    evidence_analyzed = Column(Integer, default=0)
    new_findings = Column(Integer, default=0)
    
    # Notes
    session_notes = Column(Text)
    investigator = Column(String)


class CaseRecord(Base):
    """Track investigation cases"""
    __tablename__ = 'cases'
    
    id = Column(String, primary_key=True)
    case_name = Column(String, nullable=False)
    case_description = Column(Text)
    
    # Timeline
    created_timestamp = Column(DateTime, default=datetime.now)
    last_updated = Column(DateTime, default=datetime.now)
    
    # Statistics
    total_evidence = Column(Integer, default=0)
    total_findings = Column(Integer, default=0)
    current_risk_level = Column(String)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_closed = Column(Boolean, default=False)


class ContinuousInvestigationTracker:
    """
    Main investigation tracker with persistent storage
    """

    def __init__(self, case_id: str, db_path: Optional[Path] = None):
        """
        Initialize investigation tracker

        Args:
            case_id: Unique case identifier
            db_path: Path to SQLite database (default: data/investigations.db)
        """
        self.case_id = case_id
        
        if db_path is None:
            db_path = Path("data/investigations.db")
        
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Initialize case if not exists
        self._init_case()
        
        # Load unified analyzer
        try:
            from unified_analyzer import UnifiedAnalyzer
            self.analyzer = UnifiedAnalyzer()
            logger.info("Unified analyzer loaded")
        except ImportError as e:
            logger.warning(f"Unified analyzer not available: {e}")
            self.analyzer = None
        
        self.entry_counter = 0
        self.session_counter = 0

    def _init_case(self):
        """Initialize case record if not exists"""
        session = self.SessionLocal()
        try:
            case = session.query(CaseRecord).filter_by(id=self.case_id).first()
            if not case:
                case = CaseRecord(
                    id=self.case_id,
                    case_name=self.case_id,
                    created_timestamp=datetime.now()
                )
                session.add(case)
                session.commit()
                logger.info(f"Created new case: {self.case_id}")
        finally:
            session.close()

    def add_evidence(
        self,
        file_path: Path,
        evidence_type: EvidenceType,
        sender: Optional[str] = None,
        source_timestamp: Optional[datetime] = None,
        notes: str = ""
    ) -> str:
        """
        Add evidence to investigation

        Args:
            file_path: Path to evidence file
            evidence_type: Type of evidence
            sender: Sender/source of evidence
            source_timestamp: Original timestamp of evidence
            notes: Notes about the evidence

        Returns:
            Evidence ID
        """
        self.entry_counter += 1
        evidence_id = f"EV_{self.case_id}_{self.entry_counter:06d}"
        
        session = self.SessionLocal()
        try:
            # Calculate file hash
            file_hash = self._calculate_file_hash(file_path)
            
            # Check if already exists
            existing = session.query(EvidenceRecord).filter_by(file_hash=file_hash).first()
            if existing:
                logger.warning(f"Evidence already exists: {existing.id}")
                return existing.id
            
            # Read file content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except (UnicodeDecodeError, IsADirectoryError):
                content = None
            
            # Get file stats
            stat = file_path.stat()
            
            # Create evidence record
            evidence = EvidenceRecord(
                id=evidence_id,
                case_id=self.case_id,
                evidence_type=evidence_type.value,
                file_path=str(file_path),
                file_hash=file_hash,
                file_size=stat.st_size,
                original_content=content,
                source_timestamp=source_timestamp or datetime.fromtimestamp(stat.st_mtime),
                added_timestamp=datetime.now(),
                sender=sender,
                analysis_status=AnalysisStatus.PENDING.value,
                notes=notes
            )
            
            session.add(evidence)
            session.commit()
            
            logger.info(f"Added evidence: {evidence_id} ({evidence_type.value})")
            
            # Update case
            self._update_case_stats()
            
            return evidence_id

        except Exception as e:
            logger.error(f"Error adding evidence: {e}")
            session.rollback()
            return ""
        finally:
            session.close()

    def analyze_pending_evidence(self) -> int:
        """
        Analyze all pending evidence

        Returns:
            Number of evidence items analyzed
        """
        session = self.SessionLocal()
        try:
            pending = session.query(EvidenceRecord).filter(
                EvidenceRecord.case_id == self.case_id,
                EvidenceRecord.analysis_status == AnalysisStatus.PENDING.value
            ).all()
            
            analyzed = 0
            
            for evidence in pending:
                try:
                    # Update status
                    evidence.analysis_status = AnalysisStatus.IN_PROGRESS.value
                    session.commit()
                    
                    # Run analysis based on type
                    results = self._analyze_evidence(evidence)
                    
                    # Store results
                    evidence.analysis_results = results
                    evidence.analysis_status = AnalysisStatus.COMPLETED.value
                    
                    # Extract flags and risk
                    if results:
                        evidence.forensic_flags = results.get('flags', [])
                        evidence.risk_score = results.get('risk_score', 0.0)
                        
                        if results.get('requires_review'):
                            evidence.analysis_status = AnalysisStatus.REQUIRES_REVIEW.value
                    
                    session.commit()
                    analyzed += 1
                    
                    logger.info(f"Analyzed evidence: {evidence.id}")

                except Exception as e:
                    logger.error(f"Error analyzing {evidence.id}: {e}")
                    evidence.analysis_status = AnalysisStatus.FAILED.value
                    session.commit()

            return analyzed

        finally:
            session.close()

    def _analyze_evidence(self, evidence: EvidenceRecord) -> Dict[str, Any]:
        """
        Analyze single evidence item

        Args:
            evidence: Evidence record to analyze

        Returns:
            Analysis results
        """
        results = {
            'flags': [],
            'risk_score': 0.0,
            'requires_review': False,
            'analysis_type': evidence.evidence_type
        }

        try:
            if evidence.evidence_type == EvidenceType.CHAT_EXPORT.value:
                results = self._analyze_chat_export(evidence)
            elif evidence.evidence_type == EvidenceType.VOICE_NOTE.value:
                results = self._analyze_voice_note(evidence)
            elif evidence.evidence_type == EvidenceType.TRANSCRIPTION.value:
                results = self._analyze_transcription(evidence)

        except Exception as e:
            logger.error(f"Analysis error: {e}")
            results['flags'].append(f"ANALYSIS_ERROR: {str(e)}")

        return results

    def _analyze_chat_export(self, evidence: EvidenceRecord) -> Dict[str, Any]:
        """Analyze chat export file"""
        results = {
            'flags': [],
            'risk_score': 0.0,
            'requires_review': False,
            'message_count': 0
        }

        try:
            if not self.analyzer:
                return results

            # Load chat
            messages = self.analyzer.chat_parser.parse_file(evidence.file_path) if self.analyzer.chat_parser else []
            results['message_count'] = len(messages)

            # Run text forensics
            text_results = self.analyzer.run_text_forensics()
            
            if text_results.get('summary'):
                results['text_analysis'] = text_results['summary']
                results['risk_score'] = max(results['risk_score'], 
                    text_results.get('summary', {}).get('critical_count', 0) * 10)

            # Check for critical findings
            if text_results.get('summary', {}).get('critical_count', 0) > 0:
                results['flags'].append('CRITICAL_TEXT_FINDINGS')
                results['requires_review'] = True

        except Exception as e:
            logger.error(f"Chat analysis error: {e}")

        return results

    def _analyze_voice_note(self, evidence: EvidenceRecord) -> Dict[str, Any]:
        """Analyze voice note"""
        results = {
            'flags': [],
            'risk_score': 0.0,
            'requires_review': False,
            'duration': 0.0
        }

        try:
            if not self.analyzer:
                return results

            # Transcribe
            path = Path(evidence.file_path)
            if path.exists():
                transcription = self.analyzer.audio_forensics.transcribe_audio(path) if self.analyzer.audio_forensics else None
                
                if transcription:
                    evidence.processed_content = transcription.get('text', '')
                    results['duration'] = transcription.get('duration', 0)
                    
                    # Run audio forensics
                    audio_results = self.analyzer.run_audio_forensics()
                    results['audio_analysis'] = audio_results
                    
                    # Check stress levels
                    if audio_results.get('high_stress', 0) > 0:
                        results['flags'].append('HIGH_STRESS_DETECTED')
                        results['risk_score'] = 40.0
                        results['requires_review'] = True

        except Exception as e:
            logger.error(f"Voice note analysis error: {e}")

        return results

    def _analyze_transcription(self, evidence: EvidenceRecord) -> Dict[str, Any]:
        """Analyze transcription"""
        results = {
            'flags': [],
            'risk_score': 0.0,
            'requires_review': False
        }

        try:
            # Afrikaans verification
            if evidence.original_content:
                try:
                    from afrikaans_processor import AfrikaansProcessor
                    processor = AfrikaansProcessor()
                    verification = processor.process_text(
                        evidence.original_content,
                        speaker_id=evidence.sender
                    )
                    
                    results['afrikaans_verification'] = {
                        'confidence': verification.overall_confidence,
                        'confidence_level': verification.overall_confidence_level.value,
                        'requires_review': verification.requires_human_review
                    }
                    
                    if verification.requires_human_review:
                        results['flags'].append('AFRIKAANS_REVIEW_REQUIRED')
                        results['requires_review'] = True

                except ImportError:
                    pass

        except Exception as e:
            logger.error(f"Transcription analysis error: {e}")

        return results

    def get_investigation_summary(self) -> Dict[str, Any]:
        """
        Get current investigation summary

        Returns:
            Investigation summary
        """
        session = self.SessionLocal()
        try:
            # Get case
            case = session.query(CaseRecord).filter_by(id=self.case_id).first()
            if not case:
                return {'error': 'Case not found'}

            # Get evidence statistics
            all_evidence = session.query(EvidenceRecord).filter_by(case_id=self.case_id).all()
            
            by_type = {}
            by_status = {}
            total_risk = 0.0
            flagged_count = 0
            
            for evidence in all_evidence:
                # By type
                etype = evidence.evidence_type
                by_type[etype] = by_type.get(etype, 0) + 1
                
                # By status
                status = evidence.analysis_status
                by_status[status] = by_status.get(status, 0) + 1
                
                # Risk
                total_risk += evidence.risk_score
                
                # Flagged
                if evidence.forensic_flags:
                    flagged_count += 1

            avg_risk = total_risk / max(len(all_evidence), 1)

            return {
                'case_id': self.case_id,
                'case_name': case.case_name,
                'created': case.created_timestamp.isoformat(),
                'last_updated': case.last_updated.isoformat(),
                'total_evidence': len(all_evidence),
                'evidence_by_type': by_type,
                'evidence_by_status': by_status,
                'flagged_evidence': flagged_count,
                'average_risk_score': avg_risk,
                'current_risk_level': case.current_risk_level or 'UNKNOWN'
            }

        finally:
            session.close()

    def get_flagged_evidence(self) -> List[Dict[str, Any]]:
        """
        Get all flagged evidence

        Returns:
            List of flagged evidence
        """
        session = self.SessionLocal()
        try:
            flagged = session.query(EvidenceRecord).filter(
                EvidenceRecord.case_id == self.case_id,
                EvidenceRecord.forensic_flags != None
            ).all()
            
            results = []
            for evidence in sorted(flagged, key=lambda x: x.risk_score, reverse=True):
                results.append({
                    'id': evidence.id,
                    'type': evidence.evidence_type,
                    'sender': evidence.sender,
                    'timestamp': evidence.source_timestamp.isoformat() if evidence.source_timestamp else None,
                    'risk_score': evidence.risk_score,
                    'flags': evidence.forensic_flags,
                    'status': evidence.analysis_status,
                    'notes': evidence.notes
                })
            
            return results

        finally:
            session.close()

    def get_timeline(self) -> List[Dict[str, Any]]:
        """
        Get chronological timeline of all evidence

        Returns:
            Sorted timeline
        """
        session = self.SessionLocal()
        try:
            evidence = session.query(EvidenceRecord).filter_by(
                case_id=self.case_id
            ).all()
            
            timeline = []
            for ev in sorted(evidence, key=lambda x: x.source_timestamp or x.added_timestamp):
                timeline.append({
                    'id': ev.id,
                    'type': ev.evidence_type,
                    'timestamp': (ev.source_timestamp or ev.added_timestamp).isoformat(),
                    'sender': ev.sender,
                    'risk_score': ev.risk_score,
                    'status': ev.analysis_status,
                    'flags': ev.forensic_flags or []
                })
            
            return timeline

        finally:
            session.close()

    def start_session(self, investigator: str = "", notes: str = "") -> str:
        """
        Start investigation session

        Args:
            investigator: Investigator name
            notes: Session notes

        Returns:
            Session ID
        """
        self.session_counter += 1
        session_id = f"SES_{self.case_id}_{self.session_counter:04d}"
        
        session = self.SessionLocal()
        try:
            inv_session = InvestigationSession(
                id=session_id,
                case_id=self.case_id,
                investigator=investigator,
                session_notes=notes
            )
            session.add(inv_session)
            session.commit()
            
            logger.info(f"Started investigation session: {session_id}")
            return session_id

        finally:
            session.close()

    def end_session(self, session_id: str):
        """
        End investigation session

        Args:
            session_id: Session ID to end
        """
        session = self.SessionLocal()
        try:
            inv_session = session.query(InvestigationSession).filter_by(id=session_id).first()
            if inv_session:
                inv_session.session_end = datetime.now()
                session.commit()
                logger.info(f"Ended investigation session: {session_id}")

        finally:
            session.close()

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _update_case_stats(self):
        """Update case statistics"""
        session = self.SessionLocal()
        try:
            case = session.query(CaseRecord).filter_by(id=self.case_id).first()
            if case:
                evidence = session.query(EvidenceRecord).filter_by(case_id=self.case_id).all()
                case.total_evidence = len(evidence)
                case.last_updated = datetime.now()
                
                # Calculate risk level
                avg_risk = sum(e.risk_score for e in evidence) / max(len(evidence), 1)
                if avg_risk > 70:
                    case.current_risk_level = 'CRITICAL'
                elif avg_risk > 50:
                    case.current_risk_level = 'HIGH'
                elif avg_risk > 30:
                    case.current_risk_level = 'MEDIUM'
                else:
                    case.current_risk_level = 'LOW'
                
                session.commit()

        finally:
            session.close()

    def export_investigation(self, output_path: Path, format: str = 'json') -> bool:
        """
        Export complete investigation

        Args:
            output_path: Output file path
            format: Export format (json, html, markdown)

        Returns:
            Success status
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            summary = self.get_investigation_summary()
            timeline = self.get_timeline()
            flagged = self.get_flagged_evidence()
            
            data = {
                'summary': summary,
                'timeline': timeline,
                'flagged_evidence': flagged,
                'export_timestamp': datetime.now().isoformat()
            }
            
            if format == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Exported investigation to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Export error: {e}")
            return False


if __name__ == "__main__":
    tracker = ContinuousInvestigationTracker("CASE_001")
    print("Investigation Tracker initialized")
    print("\nUsage:")
    print("  tracker.add_evidence(Path('chat.txt'), EvidenceType.CHAT_EXPORT)")
    print("  tracker.analyze_pending_evidence()")
    print("  summary = tracker.get_investigation_summary()")
    print("  flagged = tracker.get_flagged_evidence()")
    print("  timeline = tracker.get_timeline()")
