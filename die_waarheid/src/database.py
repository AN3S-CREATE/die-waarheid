"""
Database backend for Die Waarheid
SQLite database with SQLAlchemy ORM for persistent storage
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from contextlib import contextmanager

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool, QueuePool

from config import TEMP_DIR

logger = logging.getLogger(__name__)

Base = declarative_base()

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{TEMP_DIR}/die_waarheid.db")

# Database connection pool settings
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))
DB_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "3600"))  # 1 hour
DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))  # 30 seconds


class AnalysisResult(Base):
    """Forensic analysis result storage"""
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String, index=True)
    filename = Column(String, unique=True, index=True)
    duration = Column(Float)
    pitch_volatility = Column(Float)
    silence_ratio = Column(Float)
    intensity_mean = Column(Float)
    intensity_max = Column(Float)
    intensity_std = Column(Float)
    mfcc_variance = Column(Float)
    zero_crossing_rate = Column(Float)
    spectral_centroid = Column(Float)
    stress_level = Column(Float)
    stress_threshold_exceeded = Column(Boolean)
    high_cognitive_load = Column(Boolean)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Message(Base):
    """WhatsApp message storage"""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String, index=True)
    timestamp = Column(DateTime, index=True)
    sender = Column(String, index=True)
    text = Column(Text)
    message_type = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class ConversationAnalysis(Base):
    """Conversation analysis storage"""
    __tablename__ = "conversation_analyses"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String, index=True)
    total_messages = Column(Integer)
    overall_tone = Column(String)
    power_dynamics = Column(String)
    communication_style = Column(String)
    conflict_level = Column(Float)
    manipulation_indicators = Column(JSON)
    summary = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class PsychologicalProfile(Base):
    """Psychological profile storage"""
    __tablename__ = "psychological_profiles"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String, index=True)
    subject_name = Column(String)
    personality_traits = Column(JSON)
    communication_patterns = Column(JSON)
    emotional_regulation = Column(String)
    stress_indicators = Column(JSON)
    relationship_dynamics = Column(Text)
    risk_assessment = Column(String)
    recommendations = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class AnalysisSession(Base):
    """Analysis session tracking"""
    __tablename__ = "analysis_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    case_id = Column(String, index=True)
    total_messages_analyzed = Column(Integer, default=0)
    total_audio_analyzed = Column(Integer, default=0)
    average_stress_level = Column(Float, nullable=True)
    trust_score = Column(Float, nullable=True)
    status = Column(String, default="active")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DatabaseManager:
    """
    Database manager for Die Waarheid
    Handles all database operations
    """

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database manager

        Args:
            database_url: Database URL (uses environment variable if None)
        """
        if database_url is None:
            database_url = DATABASE_URL

        self.database_url = database_url
        self.engine = None
        self.SessionLocal = None
        self.initialize()

    def initialize(self) -> bool:
        """
        Initialize database connection with connection pooling and create tables

        Returns:
            True if successful
        """
        try:
            # Configure connection pooling based on database type
            if "sqlite" in self.database_url:
                # SQLite configuration with StaticPool for thread safety
                self.engine = create_engine(
                    self.database_url,
                    connect_args={
                        "check_same_thread": False,
                        "timeout": DB_POOL_TIMEOUT
                    },
                    poolclass=StaticPool,
                    pool_pre_ping=True,  # Validate connections before use
                    echo=os.getenv("DB_ECHO", "false").lower() == "true"
                )
                logger.info("Initialized SQLite database with StaticPool")
            else:
                # PostgreSQL/MySQL configuration with QueuePool
                self.engine = create_engine(
                    self.database_url,
                    poolclass=QueuePool,
                    pool_size=DB_POOL_SIZE,
                    max_overflow=DB_MAX_OVERFLOW,
                    pool_recycle=DB_POOL_RECYCLE,
                    pool_timeout=DB_POOL_TIMEOUT,
                    pool_pre_ping=True,  # Validate connections before use
                    echo=os.getenv("DB_ECHO", "false").lower() == "true"
                )
                logger.info(f"Initialized database with QueuePool: size={DB_POOL_SIZE}, max_overflow={DB_MAX_OVERFLOW}")

            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False, 
                autoflush=False, 
                bind=self.engine,
                expire_on_commit=False  # Keep objects accessible after commit
            )
            
            # Create all tables
            Base.metadata.create_all(bind=self.engine)
            
            # Test connection
            with self.get_session() as session:
                session.execute("SELECT 1")
            
            logger.info(f"Database initialized successfully: {self.database_url}")
            return True

        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            return False

    @contextmanager
    def get_session(self):
        """
        Context manager for database sessions with automatic cleanup
        
        Yields:
            SQLAlchemy session
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    def get_pool_status(self) -> dict:
        """
        Get connection pool status information
        
        Returns:
            Dictionary with pool statistics
        """
        if not self.engine:
            return {"status": "not_initialized"}
        
        pool = self.engine.pool
        return {
            "pool_size": getattr(pool, 'size', lambda: 'N/A')(),
            "checked_in": getattr(pool, 'checkedin', lambda: 'N/A')(),
            "checked_out": getattr(pool, 'checkedout', lambda: 'N/A')(),
            "overflow": getattr(pool, 'overflow', lambda: 'N/A')(),
            "invalid": getattr(pool, 'invalid', lambda: 'N/A')(),
        }

    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()

    def store_analysis_result(self, case_id: str, result: dict) -> bool:
        """
        Store forensic analysis result

        Args:
            case_id: Case identifier
            result: Analysis result dictionary

        Returns:
            True if successful
        """
        try:
            session = self.get_session()
            analysis = AnalysisResult(
                case_id=case_id,
                filename=result.get('filename'),
                duration=result.get('duration'),
                pitch_volatility=result.get('pitch_volatility'),
                silence_ratio=result.get('silence_ratio'),
                intensity_mean=result.get('intensity', {}).get('mean'),
                intensity_max=result.get('intensity', {}).get('max'),
                intensity_std=result.get('intensity', {}).get('std'),
                mfcc_variance=result.get('mfcc_variance'),
                zero_crossing_rate=result.get('zero_crossing_rate'),
                spectral_centroid=result.get('spectral_centroid'),
                stress_level=result.get('stress_level'),
                stress_threshold_exceeded=result.get('stress_threshold_exceeded'),
                high_cognitive_load=result.get('high_cognitive_load')
            )
            session.add(analysis)
            session.commit()
            session.close()
            logger.info(f"Stored analysis result for {result.get('filename')}")
            return True

        except Exception as e:
            logger.error(f"Error storing analysis result: {str(e)}")
            return False

    def store_message(self, case_id: str, message: dict) -> bool:
        """
        Store message

        Args:
            case_id: Case identifier
            message: Message dictionary

        Returns:
            True if successful
        """
        try:
            session = self.get_session()
            msg = Message(
                case_id=case_id,
                timestamp=message.get('timestamp'),
                sender=message.get('sender'),
                text=message.get('text'),
                message_type=message.get('message_type', 'text')
            )
            session.add(msg)
            session.commit()
            session.close()
            return True

        except Exception as e:
            logger.error(f"Error storing message: {str(e)}")
            return False

    def store_conversation_analysis(self, case_id: str, analysis: dict) -> bool:
        """
        Store conversation analysis

        Args:
            case_id: Case identifier
            analysis: Analysis dictionary

        Returns:
            True if successful
        """
        try:
            session = self.get_session()
            conv_analysis = ConversationAnalysis(
                case_id=case_id,
                total_messages=analysis.get('total_messages'),
                overall_tone=analysis.get('overall_tone'),
                power_dynamics=analysis.get('power_dynamics'),
                communication_style=analysis.get('communication_style'),
                conflict_level=analysis.get('conflict_level'),
                manipulation_indicators=analysis.get('manipulation_indicators'),
                summary=analysis.get('summary')
            )
            session.add(conv_analysis)
            session.commit()
            session.close()
            logger.info(f"Stored conversation analysis for case {case_id}")
            return True

        except Exception as e:
            logger.error(f"Error storing conversation analysis: {str(e)}")
            return False

    def store_psychological_profile(self, case_id: str, profile: dict) -> bool:
        """
        Store psychological profile

        Args:
            case_id: Case identifier
            profile: Profile dictionary

        Returns:
            True if successful
        """
        try:
            session = self.get_session()
            psych_profile = PsychologicalProfile(
                case_id=case_id,
                personality_traits=profile.get('personality_traits'),
                communication_patterns=profile.get('communication_patterns'),
                emotional_regulation=profile.get('emotional_regulation'),
                stress_indicators=profile.get('stress_indicators'),
                relationship_dynamics=profile.get('relationship_dynamics'),
                risk_assessment=profile.get('risk_assessment'),
                recommendations=profile.get('recommendations')
            )
            session.add(psych_profile)
            session.commit()
            session.close()
            logger.info(f"Stored psychological profile for case {case_id}")
            return True

        except Exception as e:
            logger.error(f"Error storing psychological profile: {str(e)}")
            return False

    def get_analysis_results(self, case_id: str) -> List[dict]:
        """
        Get all analysis results for a case

        Args:
            case_id: Case identifier

        Returns:
            List of analysis results
        """
        try:
            session = self.get_session()
            results = session.query(AnalysisResult).filter(
                AnalysisResult.case_id == case_id
            ).all()
            session.close()

            return [
                {
                    'filename': r.filename,
                    'stress_level': r.stress_level,
                    'duration': r.duration,
                    'created_at': r.created_at.isoformat()
                }
                for r in results
            ]

        except Exception as e:
            logger.error(f"Error retrieving analysis results: {str(e)}")
            return []

    def get_case_statistics(self, case_id: str) -> dict:
        """
        Get statistics for a case

        Args:
            case_id: Case identifier

        Returns:
            Dictionary with case statistics
        """
        try:
            session = self.get_session()

            message_count = session.query(Message).filter(
                Message.case_id == case_id
            ).count()

            analysis_count = session.query(AnalysisResult).filter(
                AnalysisResult.case_id == case_id
            ).count()

            avg_stress = session.query(AnalysisResult).filter(
                AnalysisResult.case_id == case_id
            ).with_entities(
                AnalysisResult.stress_level.avg()
            ).scalar()

            session.close()

            return {
                'case_id': case_id,
                'total_messages': message_count,
                'total_analyses': analysis_count,
                'average_stress_level': float(avg_stress) if avg_stress else 0.0
            }

        except Exception as e:
            logger.error(f"Error retrieving case statistics: {str(e)}")
            return {}

    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")


if __name__ == "__main__":
    db = DatabaseManager()
    print("Database initialized successfully")
    stats = db.get_case_statistics("TEST_001")
    print(f"Case statistics: {stats}")
