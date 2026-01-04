"""
Pydantic data models for Die Waarheid
Structured data validation for all major components
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime


class IntensityMetrics(BaseModel):
    """Audio intensity metrics"""
    mean: float = Field(ge=-100, le=0, description="Mean intensity in dB")
    max: float = Field(ge=-100, le=0, description="Maximum intensity in dB")
    std: float = Field(ge=0, description="Standard deviation of intensity")


class ForensicsResult(BaseModel):
    """Complete forensic analysis result"""
    success: bool
    filename: str
    duration: float = Field(ge=0, description="Duration in seconds")
    pitch_volatility: float = Field(ge=0, le=100, description="Pitch volatility score")
    silence_ratio: float = Field(ge=0, le=1, description="Silence ratio")
    intensity: IntensityMetrics
    mfcc_variance: float = Field(ge=0, description="MFCC variance")
    zero_crossing_rate: float = Field(ge=0, description="Zero crossing rate")
    spectral_centroid: float = Field(ge=0, description="Spectral centroid in Hz")
    stress_level: float = Field(ge=0, le=100, description="Composite stress level")
    stress_threshold_exceeded: bool
    high_cognitive_load: bool
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    @field_validator('filename')
    @classmethod
    def filename_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Filename cannot be empty')
        return v


class Message(BaseModel):
    """WhatsApp message"""
    timestamp: datetime
    sender: str = Field(min_length=1)
    text: str
    message_type: str = Field(default="text", pattern="^(text|image|audio|video|media|link)$")

    @field_validator('sender')
    @classmethod
    def sender_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Sender cannot be empty')
        return v.strip()

    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()


class ConversationAnalysis(BaseModel):
    """Conversation-level analysis result"""
    success: bool
    total_messages: int = Field(ge=0)
    overall_tone: str = Field(pattern="^(positive|negative|neutral|mixed|unknown)$")
    power_dynamics: str = Field(pattern="^(balanced|one_sided|abusive|unknown)$")
    communication_style: str = Field(pattern="^(direct|indirect|passive_aggressive|unknown)$")
    conflict_level: float = Field(ge=0, le=1)
    manipulation_indicators: List[str] = Field(default_factory=list)
    summary: str = ""
    message: Optional[str] = None


class ContradictionAnalysis(BaseModel):
    """Contradiction detection result"""
    success: bool
    contradictions: List[Dict[str, str]] = Field(default_factory=list)
    inconsistency_score: float = Field(ge=0, le=1)
    reliability_assessment: str = Field(pattern="^(high|medium|low|unknown)$")
    message: Optional[str] = None


class PsychologicalProfile(BaseModel):
    """Psychological profile result"""
    success: bool
    personality_traits: List[str] = Field(default_factory=list)
    communication_patterns: List[str] = Field(default_factory=list)
    emotional_regulation: str = Field(pattern="^(low|moderate|high|unknown)$")
    stress_indicators: List[str] = Field(default_factory=list)
    relationship_dynamics: str = ""
    risk_assessment: str = Field(pattern="^(low|medium|high|unknown)$")
    recommendations: List[str] = Field(default_factory=list)
    message: Optional[str] = None


class ToxicityDetection(BaseModel):
    """Toxicity detection result"""
    toxicity_detected: bool
    toxicity_score: float = Field(ge=0, le=1)
    detected_phrases: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class GaslightingDetection(BaseModel):
    """Gaslighting pattern detection result"""
    gaslighting_detected: bool
    gaslighting_score: float = Field(ge=0, le=1)
    detected_phrases: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class NarcissisticDetection(BaseModel):
    """Narcissistic pattern detection result"""
    narcissistic_patterns_detected: bool
    narcissism_score: float = Field(ge=0, le=1)
    detected_patterns: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class MessageAnalysis(BaseModel):
    """Single message analysis result"""
    success: bool
    text: str
    emotion: str = Field(pattern="^(positive|negative|neutral|mixed|unknown)$")
    toxicity_score: float = Field(ge=0, le=1)
    aggression_level: str = Field(pattern="^(low|medium|high)$")
    confidence: float = Field(ge=0, le=1)
    message: Optional[str] = None


class ChatExportMetadata(BaseModel):
    """WhatsApp chat export metadata"""
    total_messages: int = Field(ge=0)
    unique_senders: int = Field(ge=0)
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    message_types: Dict[str, int] = Field(default_factory=dict)
    average_message_length: float = Field(ge=0)


class TimelineEntry(BaseModel):
    """Forensic timeline entry"""
    index: int = Field(ge=0)
    msg_id: str
    sender: str
    recorded_at: datetime
    speaker_count: int = Field(ge=1)
    transcript: str
    tone_emotion: str
    pitch_volatility: Optional[float] = Field(None, ge=0, le=100)
    silence_ratio: Optional[float] = Field(None, ge=0, le=1)
    intensity_max: Optional[float] = Field(None, ge=-100, le=0)
    forensic_flag: Optional[str] = None


class AnalysisSession(BaseModel):
    """Analysis session metadata"""
    session_id: str
    case_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    total_messages_analyzed: int = Field(ge=0, default=0)
    total_audio_analyzed: int = Field(ge=0, default=0)
    average_stress_level: Optional[float] = Field(None, ge=0, le=100)
    trust_score: Optional[float] = Field(None, ge=0, le=100)
    status: str = Field(pattern="^(active|completed|paused|error)$", default="active")


if __name__ == "__main__":
    forensics_result = ForensicsResult(
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
    print("ForensicsResult validation passed")
    print(forensics_result.json(indent=2))
