"""
Unit tests for Pydantic data models
Tests validation, serialization, and error handling
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from src.models import (
    ForensicsResult,
    IntensityMetrics,
    Message,
    ConversationAnalysis,
    ContradictionAnalysis,
    PsychologicalProfile,
    ToxicityDetection,
    GaslightingDetection,
    NarcissisticDetection,
    MessageAnalysis,
    AnalysisSession
)


class TestIntensityMetrics:
    """Test IntensityMetrics model"""

    def test_valid_intensity_metrics(self):
        """Test creating valid intensity metrics"""
        metrics = IntensityMetrics(mean=-20.0, max=-5.0, std=8.5)
        assert metrics.mean == -20.0
        assert metrics.max == -5.0
        assert metrics.std == 8.5

    def test_invalid_mean_too_high(self):
        """Test that mean > 0 raises validation error"""
        with pytest.raises(ValidationError):
            IntensityMetrics(mean=5.0, max=-5.0, std=8.5)

    def test_invalid_std_negative(self):
        """Test that negative std raises validation error"""
        with pytest.raises(ValidationError):
            IntensityMetrics(mean=-20.0, max=-5.0, std=-1.0)


class TestForensicsResult:
    """Test ForensicsResult model"""

    def test_valid_forensics_result(self):
        """Test creating valid forensics result"""
        result = ForensicsResult(
            success=True,
            filename="test.wav",
            duration=10.5,
            pitch_volatility=45.2,
            silence_ratio=0.3,
            intensity=IntensityMetrics(mean=-20.0, max=-5.0, std=8.5),
            mfcc_variance=2.3,
            zero_crossing_rate=0.15,
            spectral_centroid=2500.0,
            stress_level=52.1,
            stress_threshold_exceeded=True,
            high_cognitive_load=False
        )
        assert result.success is True
        assert result.filename == "test.wav"
        assert result.stress_level == 52.1

    def test_invalid_duration_negative(self):
        """Test that negative duration raises validation error"""
        with pytest.raises(ValidationError):
            ForensicsResult(
                success=True,
                filename="test.wav",
                duration=-1.0,
                pitch_volatility=45.2,
                silence_ratio=0.3,
                intensity=IntensityMetrics(mean=-20.0, max=-5.0, std=8.5),
                mfcc_variance=2.3,
                zero_crossing_rate=0.15,
                spectral_centroid=2500.0,
                stress_level=52.1,
                stress_threshold_exceeded=True,
                high_cognitive_load=False
            )

    def test_invalid_stress_level_out_of_bounds(self):
        """Test that stress level > 100 raises validation error"""
        with pytest.raises(ValidationError):
            ForensicsResult(
                success=True,
                filename="test.wav",
                duration=10.5,
                pitch_volatility=45.2,
                silence_ratio=0.3,
                intensity=IntensityMetrics(mean=-20.0, max=-5.0, std=8.5),
                mfcc_variance=2.3,
                zero_crossing_rate=0.15,
                spectral_centroid=2500.0,
                stress_level=150.0,
                stress_threshold_exceeded=True,
                high_cognitive_load=False
            )

    def test_empty_filename_raises_error(self):
        """Test that empty filename raises validation error"""
        with pytest.raises(ValidationError):
            ForensicsResult(
                success=True,
                filename="",
                duration=10.5,
                pitch_volatility=45.2,
                silence_ratio=0.3,
                intensity=IntensityMetrics(mean=-20.0, max=-5.0, std=8.5),
                mfcc_variance=2.3,
                zero_crossing_rate=0.15,
                spectral_centroid=2500.0,
                stress_level=52.1,
                stress_threshold_exceeded=True,
                high_cognitive_load=False
            )

    def test_json_serialization(self):
        """Test JSON serialization of forensics result"""
        result = ForensicsResult(
            success=True,
            filename="test.wav",
            duration=10.5,
            pitch_volatility=45.2,
            silence_ratio=0.3,
            intensity=IntensityMetrics(mean=-20.0, max=-5.0, std=8.5),
            mfcc_variance=2.3,
            zero_crossing_rate=0.15,
            spectral_centroid=2500.0,
            stress_level=52.1,
            stress_threshold_exceeded=True,
            high_cognitive_load=False
        )
        json_str = result.json()
        assert "test.wav" in json_str
        assert "52.1" in json_str


class TestMessage:
    """Test Message model"""

    def test_valid_message(self):
        """Test creating valid message"""
        msg = Message(
            timestamp=datetime.now(),
            sender="Alice",
            text="Hello, how are you?",
            message_type="text"
        )
        assert msg.sender == "Alice"
        assert msg.message_type == "text"

    def test_empty_sender_raises_error(self):
        """Test that empty sender raises validation error"""
        with pytest.raises(ValidationError):
            Message(
                timestamp=datetime.now(),
                sender="",
                text="Hello",
                message_type="text"
            )

    def test_empty_text_raises_error(self):
        """Test that empty text raises validation error"""
        with pytest.raises(ValidationError):
            Message(
                timestamp=datetime.now(),
                sender="Alice",
                text="",
                message_type="text"
            )

    def test_invalid_message_type(self):
        """Test that invalid message type raises validation error"""
        with pytest.raises(ValidationError):
            Message(
                timestamp=datetime.now(),
                sender="Alice",
                text="Hello",
                message_type="invalid_type"
            )

    def test_valid_message_types(self):
        """Test all valid message types"""
        valid_types = ["text", "image", "audio", "video", "media", "link"]
        for msg_type in valid_types:
            msg = Message(
                timestamp=datetime.now(),
                sender="Alice",
                text="Hello",
                message_type=msg_type
            )
            assert msg.message_type == msg_type


class TestConversationAnalysis:
    """Test ConversationAnalysis model"""

    def test_valid_conversation_analysis(self):
        """Test creating valid conversation analysis"""
        analysis = ConversationAnalysis(
            success=True,
            total_messages=100,
            overall_tone="mixed",
            power_dynamics="balanced",
            communication_style="direct",
            conflict_level=0.3,
            manipulation_indicators=["gaslighting", "blame_shifting"],
            summary="Conversation shows balanced dynamics"
        )
        assert analysis.success is True
        assert analysis.total_messages == 100
        assert len(analysis.manipulation_indicators) == 2

    def test_invalid_tone(self):
        """Test that invalid tone raises validation error"""
        with pytest.raises(ValidationError):
            ConversationAnalysis(
                success=True,
                total_messages=100,
                overall_tone="invalid_tone",
                power_dynamics="balanced",
                communication_style="direct",
                conflict_level=0.3
            )

    def test_invalid_conflict_level(self):
        """Test that conflict level out of bounds raises error"""
        with pytest.raises(ValidationError):
            ConversationAnalysis(
                success=True,
                total_messages=100,
                overall_tone="mixed",
                power_dynamics="balanced",
                communication_style="direct",
                conflict_level=1.5
            )


class TestPsychologicalProfile:
    """Test PsychologicalProfile model"""

    def test_valid_profile(self):
        """Test creating valid psychological profile"""
        profile = PsychologicalProfile(
            success=True,
            personality_traits=["assertive", "analytical"],
            communication_patterns=["direct", "clear"],
            emotional_regulation="high",
            stress_indicators=["none"],
            relationship_dynamics="healthy",
            risk_assessment="low",
            recommendations=["continue current approach"]
        )
        assert profile.success is True
        assert len(profile.personality_traits) == 2

    def test_invalid_emotional_regulation(self):
        """Test that invalid emotional regulation raises error"""
        with pytest.raises(ValidationError):
            PsychologicalProfile(
                success=True,
                emotional_regulation="invalid"
            )

    def test_invalid_risk_assessment(self):
        """Test that invalid risk assessment raises error"""
        with pytest.raises(ValidationError):
            PsychologicalProfile(
                success=True,
                risk_assessment="invalid"
            )


class TestToxicityDetection:
    """Test ToxicityDetection model"""

    def test_valid_toxicity_detection(self):
        """Test creating valid toxicity detection"""
        detection = ToxicityDetection(
            toxicity_detected=True,
            toxicity_score=0.85,
            detected_phrases=["stupid", "idiot"],
            confidence=0.92
        )
        assert detection.toxicity_detected is True
        assert detection.toxicity_score == 0.85

    def test_invalid_toxicity_score(self):
        """Test that toxicity score out of bounds raises error"""
        with pytest.raises(ValidationError):
            ToxicityDetection(
                toxicity_detected=True,
                toxicity_score=1.5,
                detected_phrases=["stupid"],
                confidence=0.92
            )


class TestMessageAnalysis:
    """Test MessageAnalysis model"""

    def test_valid_message_analysis(self):
        """Test creating valid message analysis"""
        analysis = MessageAnalysis(
            success=True,
            text="This is a test message",
            emotion="positive",
            toxicity_score=0.1,
            aggression_level="low",
            confidence=0.95
        )
        assert analysis.success is True
        assert analysis.emotion == "positive"

    def test_invalid_emotion(self):
        """Test that invalid emotion raises error"""
        with pytest.raises(ValidationError):
            MessageAnalysis(
                success=True,
                text="Test",
                emotion="invalid_emotion",
                toxicity_score=0.1,
                aggression_level="low",
                confidence=0.95
            )

    def test_invalid_aggression_level(self):
        """Test that invalid aggression level raises error"""
        with pytest.raises(ValidationError):
            MessageAnalysis(
                success=True,
                text="Test",
                emotion="positive",
                toxicity_score=0.1,
                aggression_level="invalid",
                confidence=0.95
            )


class TestAnalysisSession:
    """Test AnalysisSession model"""

    def test_valid_session(self):
        """Test creating valid analysis session"""
        session = AnalysisSession(
            session_id="SESSION_001",
            case_id="CASE_001",
            total_messages_analyzed=50,
            total_audio_analyzed=10,
            average_stress_level=45.5,
            trust_score=75.0,
            status="active"
        )
        assert session.session_id == "SESSION_001"
        assert session.status == "active"

    def test_invalid_status(self):
        """Test that invalid status raises error"""
        with pytest.raises(ValidationError):
            AnalysisSession(
                session_id="SESSION_001",
                case_id="CASE_001",
                status="invalid_status"
            )

    def test_default_values(self):
        """Test default values for optional fields"""
        session = AnalysisSession(
            session_id="SESSION_001",
            case_id="CASE_001"
        )
        assert session.total_messages_analyzed == 0
        assert session.total_audio_analyzed == 0
        assert session.status == "active"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
