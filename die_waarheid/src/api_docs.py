"""
API Documentation for Die Waarheid
Comprehensive API reference with usage examples and parameter documentation
"""

from typing import Dict, List, Any


class APIDocumentation:
    """
    Complete API documentation for Die Waarheid
    Provides reference information for all major modules and functions
    """

    FORENSICS_API = {
        "module": "src.forensics",
        "class": "ForensicsEngine",
        "description": "Audio forensics analysis engine for bio-signal detection",
        "methods": {
            "analyze": {
                "description": "Complete forensic analysis of audio file with caching support",
                "parameters": {
                    "file_path": {
                        "type": "Path",
                        "description": "Path to audio file",
                        "required": True
                    }
                },
                "returns": {
                    "type": "Dict",
                    "description": "Dictionary with forensic metrics including stress level, pitch volatility, silence ratio, etc."
                },
                "example": """
from pathlib import Path
from src.forensics import ForensicsEngine

engine = ForensicsEngine(use_cache=True)
result = engine.analyze(Path("audio/test.wav"))
print(f"Stress Level: {result['stress_level']}")
                """
            },
            "batch_analyze": {
                "description": "Analyze multiple audio files sequentially",
                "parameters": {
                    "file_paths": {
                        "type": "List[Path]",
                        "description": "List of paths to audio files",
                        "required": True
                    },
                    "progress_callback": {
                        "type": "Optional[Callable]",
                        "description": "Optional callback for progress updates",
                        "required": False
                    }
                },
                "returns": {
                    "type": "List[Dict]",
                    "description": "List of analysis results"
                },
                "example": """
def progress(current, total, filename):
    print(f"Processing {current}/{total}: {filename}")

results = engine.batch_analyze(file_paths, progress_callback=progress)
                """
            },
            "batch_analyze_parallel": {
                "description": "Analyze multiple audio files in parallel",
                "parameters": {
                    "file_paths": {
                        "type": "List[Path]",
                        "description": "List of paths to audio files",
                        "required": True
                    },
                    "max_workers": {
                        "type": "int",
                        "description": "Maximum number of worker threads",
                        "required": False,
                        "default": 4
                    },
                    "progress_callback": {
                        "type": "Optional[Callable]",
                        "description": "Optional callback for progress updates",
                        "required": False
                    }
                },
                "returns": {
                    "type": "List[Dict]",
                    "description": "List of analysis results"
                },
                "example": """
results = engine.batch_analyze_parallel(
    file_paths,
    max_workers=4,
    progress_callback=progress
)
                """
            }
        }
    }

    AI_ANALYZER_API = {
        "module": "src.ai_analyzer",
        "class": "AIAnalyzer",
        "description": "Gemini-powered psychological profiling and pattern detection",
        "methods": {
            "analyze_message": {
                "description": "Analyze single message for toxicity, emotion, and patterns",
                "parameters": {
                    "text": {
                        "type": "str",
                        "description": "Message text to analyze",
                        "required": True
                    }
                },
                "returns": {
                    "type": "Dict",
                    "description": "Analysis result with emotion, toxicity_score, aggression_level, confidence"
                },
                "example": """
from src.ai_analyzer import AIAnalyzer

analyzer = AIAnalyzer()
result = analyzer.analyze_message("This is a test message")
print(f"Emotion: {result['emotion']}")
print(f"Toxicity: {result['toxicity_score']}")
                """
            },
            "analyze_conversation": {
                "description": "Analyze entire conversation for patterns and dynamics",
                "parameters": {
                    "messages": {
                        "type": "List[Dict]",
                        "description": "List of message dicts with 'sender' and 'text' keys",
                        "required": True
                    }
                },
                "returns": {
                    "type": "Dict",
                    "description": "Conversation analysis with tone, power dynamics, communication style"
                },
                "example": """
messages = [
    {"sender": "Alice", "text": "Hello"},
    {"sender": "Bob", "text": "Hi there"}
]
result = analyzer.analyze_conversation(messages)
print(f"Overall Tone: {result['overall_tone']}")
                """
            },
            "detect_contradictions": {
                "description": "Detect contradictions and inconsistencies in conversation",
                "parameters": {
                    "messages": {
                        "type": "List[Dict]",
                        "description": "List of message dictionaries",
                        "required": True
                    }
                },
                "returns": {
                    "type": "Dict",
                    "description": "Contradictions with statements and explanations"
                }
            },
            "generate_psychological_profile": {
                "description": "Generate comprehensive psychological profile",
                "parameters": {
                    "messages": {
                        "type": "List[Dict]",
                        "description": "List of message dictionaries",
                        "required": True
                    },
                    "forensics_data": {
                        "type": "Optional[List[Dict]]",
                        "description": "Optional forensic analysis results",
                        "required": False
                    }
                },
                "returns": {
                    "type": "Dict",
                    "description": "Psychological profile with traits, patterns, risk assessment"
                }
            }
        }
    }

    DATABASE_API = {
        "module": "src.database",
        "class": "DatabaseManager",
        "description": "SQLite database backend for persistent storage",
        "methods": {
            "store_analysis_result": {
                "description": "Store forensic analysis result",
                "parameters": {
                    "case_id": {
                        "type": "str",
                        "description": "Case identifier",
                        "required": True
                    },
                    "result": {
                        "type": "Dict",
                        "description": "Analysis result dictionary",
                        "required": True
                    }
                },
                "returns": {
                    "type": "bool",
                    "description": "True if successful"
                },
                "example": """
from src.database import DatabaseManager

db = DatabaseManager()
success = db.store_analysis_result("CASE_001", result)
                """
            },
            "get_case_statistics": {
                "description": "Get statistics for a case",
                "parameters": {
                    "case_id": {
                        "type": "str",
                        "description": "Case identifier",
                        "required": True
                    }
                },
                "returns": {
                    "type": "Dict",
                    "description": "Case statistics with message count, analysis count, average stress"
                },
                "example": """
stats = db.get_case_statistics("CASE_001")
print(f"Total Messages: {stats['total_messages']}")
print(f"Average Stress: {stats['average_stress_level']}")
                """
            }
        }
    }

    CACHE_API = {
        "module": "src.cache",
        "class": "AnalysisCache",
        "description": "Persistent caching layer for analysis results",
        "methods": {
            "get": {
                "description": "Get cached analysis result",
                "parameters": {
                    "file_path": {
                        "type": "Path",
                        "description": "Path to audio file",
                        "required": True
                    }
                },
                "returns": {
                    "type": "Optional[Dict]",
                    "description": "Cached result or None if not found"
                },
                "example": """
from src.cache import AnalysisCache

cache = AnalysisCache()
result = cache.get(Path("audio/test.wav"))
                """
            },
            "set": {
                "description": "Store analysis result in cache",
                "parameters": {
                    "file_path": {
                        "type": "Path",
                        "description": "Path to audio file",
                        "required": True
                    },
                    "result": {
                        "type": "Dict",
                        "description": "Analysis result to cache",
                        "required": True
                    }
                },
                "returns": {
                    "type": "bool",
                    "description": "True if successful"
                }
            }
        }
    }

    HEALTH_API = {
        "module": "src.health",
        "class": "HealthChecker",
        "description": "System health monitoring and diagnostics",
        "methods": {
            "get_status_summary": {
                "description": "Get brief health status summary",
                "returns": {
                    "type": "Dict",
                    "description": "Status summary with CPU, memory, disk, and service status"
                },
                "example": """
from src.health import HealthChecker

checker = HealthChecker()
status = checker.get_status_summary()
print(f"Overall Status: {status['overall_status']}")
print(f"CPU: {status['cpu_percent']:.1f}%")
                """
            },
            "get_full_health_status": {
                "description": "Get complete system health status",
                "returns": {
                    "type": "Dict",
                    "description": "Detailed health status with all checks"
                }
            },
            "get_diagnostics": {
                "description": "Get complete diagnostics information",
                "returns": {
                    "type": "Dict",
                    "description": "Diagnostics including health, performance, and system info"
                }
            }
        }
    }

    DIARIZATION_API = {
        "module": "src.diarization",
        "class": "DiarizationPipeline",
        "description": "Speaker diarization and identification",
        "methods": {
            "diarize": {
                "description": "Perform speaker diarization",
                "parameters": {
                    "audio": {
                        "type": "np.ndarray",
                        "description": "Audio signal",
                        "required": True
                    },
                    "num_speakers": {
                        "type": "int",
                        "description": "Expected number of speakers",
                        "required": False,
                        "default": 2
                    }
                },
                "returns": {
                    "type": "List[Dict]",
                    "description": "List of speaker segments with timing and speaker ID"
                },
                "example": """
from src.diarization import DiarizationPipeline
import numpy as np

diarizer = DiarizationPipeline()
segments = diarizer.diarize(audio, num_speakers=2)
for seg in segments:
    print(f"{seg['speaker']}: {seg['start']:.2f}s - {seg['end']:.2f}s")
                """
            },
            "get_speaker_statistics": {
                "description": "Get speaker statistics",
                "parameters": {
                    "segments": {
                        "type": "List[Dict]",
                        "description": "List of speaker segments",
                        "required": True
                    }
                },
                "returns": {
                    "type": "Dict",
                    "description": "Speaker statistics with duration and percentage"
                }
            }
        }
    }

    MODELS_API = {
        "module": "src.models",
        "description": "Pydantic data models for validation",
        "models": {
            "ForensicsResult": {
                "description": "Complete forensic analysis result",
                "fields": {
                    "success": "bool - Analysis success status",
                    "filename": "str - Audio filename",
                    "duration": "float - Duration in seconds",
                    "stress_level": "float - Composite stress level (0-100)",
                    "pitch_volatility": "float - Pitch volatility score (0-100)",
                    "silence_ratio": "float - Silence ratio (0-1)",
                    "intensity": "IntensityMetrics - Intensity metrics",
                    "mfcc_variance": "float - MFCC variance",
                    "zero_crossing_rate": "float - Zero crossing rate",
                    "spectral_centroid": "float - Spectral centroid in Hz"
                }
            },
            "Message": {
                "description": "WhatsApp message",
                "fields": {
                    "timestamp": "datetime - Message timestamp",
                    "sender": "str - Message sender",
                    "text": "str - Message text",
                    "message_type": "str - Type (text, image, audio, video, media, link)"
                }
            },
            "ConversationAnalysis": {
                "description": "Conversation-level analysis",
                "fields": {
                    "success": "bool - Analysis success",
                    "overall_tone": "str - Overall tone (positive, negative, neutral, mixed)",
                    "power_dynamics": "str - Power dynamics (balanced, one_sided, abusive)",
                    "conflict_level": "float - Conflict level (0-1)",
                    "manipulation_indicators": "List[str] - Detected manipulation patterns"
                }
            },
            "PsychologicalProfile": {
                "description": "Psychological profile",
                "fields": {
                    "personality_traits": "List[str] - Detected personality traits",
                    "communication_patterns": "List[str] - Communication patterns",
                    "emotional_regulation": "str - Emotional regulation level",
                    "stress_indicators": "List[str] - Stress indicators",
                    "risk_assessment": "str - Risk level (low, medium, high)"
                }
            }
        }
    }

    @classmethod
    def get_api_reference(cls) -> Dict[str, Any]:
        """Get complete API reference"""
        return {
            "forensics": cls.FORENSICS_API,
            "ai_analyzer": cls.AI_ANALYZER_API,
            "database": cls.DATABASE_API,
            "cache": cls.CACHE_API,
            "health": cls.HEALTH_API,
            "diarization": cls.DIARIZATION_API,
            "models": cls.MODELS_API
        }

    @classmethod
    def get_quick_start(cls) -> str:
        """Get quick start guide"""
        return """
# Die Waarheid Quick API Reference

## Basic Forensic Analysis
```python
from pathlib import Path
from src.forensics import ForensicsEngine

engine = ForensicsEngine(use_cache=True)
result = engine.analyze(Path("audio/test.wav"))
print(f"Stress Level: {result['stress_level']}")
```

## AI Analysis
```python
from src.ai_analyzer import AIAnalyzer

analyzer = AIAnalyzer()
result = analyzer.analyze_message("Test message")
print(f"Emotion: {result['emotion']}")
```

## Database Operations
```python
from src.database import DatabaseManager

db = DatabaseManager()
db.store_analysis_result("CASE_001", result)
stats = db.get_case_statistics("CASE_001")
```

## Health Monitoring
```python
from src.health import HealthChecker

checker = HealthChecker()
status = checker.get_status_summary()
print(f"System Status: {status['overall_status']}")
```

## Data Validation
```python
from src.models import ForensicsResult

result = ForensicsResult(
    success=True,
    filename="test.wav",
    duration=10.5,
    # ... other fields
)
```
        """


if __name__ == "__main__":
    docs = APIDocumentation()
    api_ref = docs.get_api_reference()
    print("API Reference loaded successfully")
    print(f"Available modules: {list(api_ref.keys())}")
