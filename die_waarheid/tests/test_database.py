"""
Unit tests for database backend
Tests database operations, queries, and persistence
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from src.database import DatabaseManager, AnalysisResult, Message, ConversationAnalysis


class TestDatabaseManager:
    """Test DatabaseManager functionality"""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"sqlite:///{tmpdir}/test.db"
            yield db_path

    @pytest.fixture
    def db(self, temp_db_path):
        """Create database manager instance"""
        manager = DatabaseManager(database_url=temp_db_path)
        yield manager
        manager.close()

    def test_database_initialization(self, db):
        """Test database initialization"""
        assert db.engine is not None
        assert db.SessionLocal is not None

    def test_store_analysis_result(self, db):
        """Test storing analysis result"""
        result = {
            'filename': 'test.wav',
            'duration': 10.5,
            'pitch_volatility': 45.2,
            'silence_ratio': 0.3,
            'intensity': {'mean': -20.0, 'max': -5.0, 'std': 8.5},
            'mfcc_variance': 2.3,
            'zero_crossing_rate': 0.15,
            'spectral_centroid': 2500.0,
            'stress_level': 52.1,
            'stress_threshold_exceeded': True,
            'high_cognitive_load': False
        }

        success = db.store_analysis_result('CASE_001', result)
        assert success is True

    def test_store_multiple_results(self, db):
        """Test storing multiple analysis results"""
        for i in range(5):
            result = {
                'filename': f'test_{i}.wav',
                'duration': 10.5 + i,
                'pitch_volatility': 45.2 + i,
                'silence_ratio': 0.3,
                'intensity': {'mean': -20.0, 'max': -5.0, 'std': 8.5},
                'mfcc_variance': 2.3,
                'zero_crossing_rate': 0.15,
                'spectral_centroid': 2500.0,
                'stress_level': 52.1 + i,
                'stress_threshold_exceeded': True,
                'high_cognitive_load': False
            }
            success = db.store_analysis_result('CASE_001', result)
            assert success is True

    def test_store_message(self, db):
        """Test storing message"""
        message = {
            'timestamp': datetime.now(),
            'sender': 'Alice',
            'text': 'Hello, how are you?',
            'message_type': 'text'
        }

        success = db.store_message('CASE_001', message)
        assert success is True

    def test_store_conversation_analysis(self, db):
        """Test storing conversation analysis"""
        analysis = {
            'total_messages': 100,
            'overall_tone': 'mixed',
            'power_dynamics': 'balanced',
            'communication_style': 'direct',
            'conflict_level': 0.3,
            'manipulation_indicators': ['gaslighting'],
            'summary': 'Balanced conversation'
        }

        success = db.store_conversation_analysis('CASE_001', analysis)
        assert success is True

    def test_get_analysis_results(self, db):
        """Test retrieving analysis results"""
        result = {
            'filename': 'test.wav',
            'duration': 10.5,
            'pitch_volatility': 45.2,
            'silence_ratio': 0.3,
            'intensity': {'mean': -20.0, 'max': -5.0, 'std': 8.5},
            'mfcc_variance': 2.3,
            'zero_crossing_rate': 0.15,
            'spectral_centroid': 2500.0,
            'stress_level': 52.1,
            'stress_threshold_exceeded': True,
            'high_cognitive_load': False
        }

        db.store_analysis_result('CASE_001', result)
        results = db.get_analysis_results('CASE_001')

        assert len(results) > 0
        assert results[0]['filename'] == 'test.wav'
        assert results[0]['stress_level'] == 52.1

    def test_get_case_statistics(self, db):
        """Test retrieving case statistics"""
        result = {
            'filename': 'test.wav',
            'duration': 10.5,
            'pitch_volatility': 45.2,
            'silence_ratio': 0.3,
            'intensity': {'mean': -20.0, 'max': -5.0, 'std': 8.5},
            'mfcc_variance': 2.3,
            'zero_crossing_rate': 0.15,
            'spectral_centroid': 2500.0,
            'stress_level': 52.1,
            'stress_threshold_exceeded': True,
            'high_cognitive_load': False
        }

        db.store_analysis_result('CASE_001', result)

        message = {
            'timestamp': datetime.now(),
            'sender': 'Alice',
            'text': 'Hello',
            'message_type': 'text'
        }
        db.store_message('CASE_001', message)

        stats = db.get_case_statistics('CASE_001')

        assert stats['case_id'] == 'CASE_001'
        assert stats['total_messages'] == 1
        assert stats['total_analyses'] == 1
        assert stats['average_stress_level'] == 52.1

    def test_get_case_statistics_empty_case(self, db):
        """Test getting statistics for empty case"""
        stats = db.get_case_statistics('EMPTY_CASE')

        assert stats['case_id'] == 'EMPTY_CASE'
        assert stats['total_messages'] == 0
        assert stats['total_analyses'] == 0
        assert stats['average_stress_level'] == 0.0

    def test_multiple_cases(self, db):
        """Test handling multiple cases"""
        result1 = {
            'filename': 'case1.wav',
            'duration': 10.5,
            'pitch_volatility': 45.2,
            'silence_ratio': 0.3,
            'intensity': {'mean': -20.0, 'max': -5.0, 'std': 8.5},
            'mfcc_variance': 2.3,
            'zero_crossing_rate': 0.15,
            'spectral_centroid': 2500.0,
            'stress_level': 52.1,
            'stress_threshold_exceeded': True,
            'high_cognitive_load': False
        }

        result2 = {
            'filename': 'case2.wav',
            'duration': 15.0,
            'pitch_volatility': 55.0,
            'silence_ratio': 0.4,
            'intensity': {'mean': -25.0, 'max': -10.0, 'std': 9.0},
            'mfcc_variance': 3.0,
            'zero_crossing_rate': 0.20,
            'spectral_centroid': 3000.0,
            'stress_level': 65.0,
            'stress_threshold_exceeded': True,
            'high_cognitive_load': True
        }

        db.store_analysis_result('CASE_001', result1)
        db.store_analysis_result('CASE_002', result2)

        stats1 = db.get_case_statistics('CASE_001')
        stats2 = db.get_case_statistics('CASE_002')

        assert stats1['average_stress_level'] == 52.1
        assert stats2['average_stress_level'] == 65.0

    def test_database_error_handling(self, db):
        """Test error handling in database operations"""
        invalid_result = {
            'filename': None,  # Invalid
            'duration': 10.5
        }

        success = db.store_analysis_result('CASE_001', invalid_result)
        assert success is False

    def test_database_close(self, db):
        """Test closing database connection"""
        db.close()
        assert db.engine is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
