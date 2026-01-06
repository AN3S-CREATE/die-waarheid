"""
Database backend for Die Waarheid
SQLite database with SQLAlchemy ORM for persistent storage
"""

import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from functools import wraps

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON, Index, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, joinedload, selectinload
from sqlalchemy.pool import StaticPool, QueuePool
from sqlalchemy.sql import text

from config import TEMP_DIR

logger = logging.getLogger(__name__)

Base = declarative_base()

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{TEMP_DIR}/die_waarheid.db")

# Database connection pool settings
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))
DB_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "3600"))  # 1 hour
DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))  # 30 seconds

# Query caching settings
QUERY_CACHE_TTL = int(os.getenv("QUERY_CACHE_TTL", "300"))  # 5 minutes
ENABLE_QUERY_CACHE = os.getenv("ENABLE_QUERY_CACHE", "true").lower() == "true"

# Simple in-memory cache for query results
_query_cache: Dict[str, Dict[str, Any]] = {}

def query_cache(ttl: int = QUERY_CACHE_TTL):
    """Decorator for caching query results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not ENABLE_QUERY_CACHE:
                return func(*args, **kwargs)
            
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Check if cached result exists and is still valid
            if cache_key in _query_cache:
                cached_data = _query_cache[cache_key]
                if time.time() - cached_data['timestamp'] < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_data['result']
                else:
                    # Remove expired cache entry
                    del _query_cache[cache_key]
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            _query_cache[cache_key] = {
                'result': result,
                'timestamp': time.time()
            }
            logger.debug(f"Cached result for {func.__name__}")
            return result
        return wrapper
    return decorator

def clear_query_cache():
    """Clear all cached query results"""
    global _query_cache
    _query_cache.clear()
    logger.info("Query cache cleared")

def get_cache_stats() -> Dict[str, Any]:
    """Get query cache statistics"""
    current_time = time.time()
    valid_entries = sum(1 for entry in _query_cache.values() 
                       if current_time - entry['timestamp'] < QUERY_CACHE_TTL)
    
    return {
        'total_entries': len(_query_cache),
        'valid_entries': valid_entries,
        'expired_entries': len(_query_cache) - valid_entries,
        'cache_enabled': ENABLE_QUERY_CACHE
    }


class AnalysisResult(Base):
    """Forensic analysis result storage with optimized indexes"""
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
    stress_level = Column(Float, index=True)  # Frequently queried
    stress_threshold_exceeded = Column(Boolean, index=True)  # Frequently filtered
    high_cognitive_load = Column(Boolean, index=True)  # Frequently filtered
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Composite indexes for common query patterns
    __table_args__ = (
        Index('idx_case_created', 'case_id', 'created_at'),  # Case timeline queries
        Index('idx_case_stress', 'case_id', 'stress_level'),  # Stress analysis queries
        Index('idx_stress_threshold', 'stress_threshold_exceeded', 'stress_level'),  # Threshold queries
    )


class Message(Base):
    """WhatsApp message storage with optimized indexes"""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String, index=True)
    timestamp = Column(DateTime, index=True)
    sender = Column(String, index=True)
    text = Column(Text)
    message_type = Column(String, index=True)  # Frequently filtered
    created_at = Column(DateTime, default=datetime.utcnow)

    # Composite indexes for common query patterns
    __table_args__ = (
        Index('idx_case_timestamp', 'case_id', 'timestamp'),  # Timeline queries
        Index('idx_case_sender', 'case_id', 'sender'),  # Sender analysis queries
        Index('idx_sender_timestamp', 'sender', 'timestamp'),  # Sender timeline queries
    )


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
        Initialize database connection with pooling and create tables

        Returns:
            True if successful
        """
        try:
            # Configure connection pooling based on database type
            if "sqlite" in self.database_url:
                # SQLite with StaticPool for thread safety
                self.engine = create_engine(
                    self.database_url,
                    poolclass=StaticPool,
                    connect_args={
                        "check_same_thread": False,
                        "timeout": DB_POOL_TIMEOUT
                    },
                    pool_pre_ping=True,
                    echo=False
                )
            else:
                # PostgreSQL/MySQL with QueuePool for production
                self.engine = create_engine(
                    self.database_url,
                    poolclass=QueuePool,
                    pool_size=DB_POOL_SIZE,
                    max_overflow=DB_MAX_OVERFLOW,
                    pool_recycle=DB_POOL_RECYCLE,
                    pool_timeout=DB_POOL_TIMEOUT,
                    pool_pre_ping=True,
                    echo=False
                )
            
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            Base.metadata.create_all(bind=self.engine)
            logger.info(f"Database initialized with connection pooling: {self.database_url}")
            return True

        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            return False

    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

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

    @query_cache(ttl=QUERY_CACHE_TTL)
    def get_analysis_results(self, case_id: str) -> List[dict]:
        """
        Get all analysis results for a case with caching and optimized query

        Args:
            case_id: Case identifier

        Returns:
            List of analysis results
        """
        try:
            with self.session_scope() as session:
                # Optimized query with ordering for consistent results
                results = session.query(AnalysisResult).filter(
                    AnalysisResult.case_id == case_id
                ).order_by(AnalysisResult.created_at.desc()).all()

                return [
                    {
                        'id': r.id,
                        'filename': r.filename,
                        'stress_level': r.stress_level,
                        'stress_threshold_exceeded': r.stress_threshold_exceeded,
                        'high_cognitive_load': r.high_cognitive_load,
                        'duration': r.duration,
                        'pitch_volatility': r.pitch_volatility,
                        'silence_ratio': r.silence_ratio,
                        'intensity_mean': r.intensity_mean,
                        'created_at': r.created_at.isoformat() if r.created_at else None
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

    @query_cache(ttl=600)  # Cache for 10 minutes
    def get_high_stress_results(self, case_id: str, threshold: float = 50.0) -> List[dict]:
        """
        Get analysis results with high stress levels (optimized query)
        
        Args:
            case_id: Case identifier
            threshold: Stress level threshold
            
        Returns:
            List of high-stress analysis results
        """
        try:
            with self.session_scope() as session:
                # Use index on stress_level for fast filtering
                results = session.query(AnalysisResult).filter(
                    AnalysisResult.case_id == case_id,
                    AnalysisResult.stress_level >= threshold
                ).order_by(AnalysisResult.stress_level.desc()).all()
                
                return [
                    {
                        'filename': r.filename,
                        'stress_level': r.stress_level,
                        'duration': r.duration,
                        'created_at': r.created_at.isoformat() if r.created_at else None
                    }
                    for r in results
                ]
        except Exception as e:
            logger.error(f"Error retrieving high stress results: {str(e)}")
            return []

    @query_cache(ttl=300)  # Cache for 5 minutes
    def get_conversation_timeline(self, case_id: str, limit: int = 100) -> List[dict]:
        """
        Get conversation timeline with optimized query using composite index
        
        Args:
            case_id: Case identifier
            limit: Maximum number of messages to return
            
        Returns:
            List of messages in chronological order
        """
        try:
            with self.session_scope() as session:
                # Use composite index idx_case_timestamp for fast timeline queries
                messages = session.query(Message).filter(
                    Message.case_id == case_id
                ).order_by(Message.timestamp.asc()).limit(limit).all()
                
                return [
                    {
                        'id': m.id,
                        'timestamp': m.timestamp.isoformat() if m.timestamp else None,
                        'sender': m.sender,
                        'text': m.text,
                        'message_type': m.message_type
                    }
                    for m in messages
                ]
        except Exception as e:
            logger.error(f"Error retrieving conversation timeline: {str(e)}")
            return []

    def get_query_performance_stats(self) -> Dict[str, Any]:
        """
        Get database query performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            with self.session_scope() as session:
                # Get table row counts for performance monitoring
                analysis_count = session.query(func.count(AnalysisResult.id)).scalar()
                message_count = session.query(func.count(Message.id)).scalar()
                
                # Get cache statistics
                cache_stats = get_cache_stats()
                
                return {
                    'database_url': self.database_url,
                    'analysis_results_count': analysis_count,
                    'messages_count': message_count,
                    'cache_stats': cache_stats,
                    'pool_size': DB_POOL_SIZE,
                    'max_overflow': DB_MAX_OVERFLOW,
                    'pool_timeout': DB_POOL_TIMEOUT
                }
        except Exception as e:
            logger.error(f"Error retrieving performance stats: {str(e)}")
            return {'error': str(e)}

    def clear_cache(self):
        """Clear query cache manually"""
        clear_query_cache()
        logger.info("Database query cache cleared")

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
