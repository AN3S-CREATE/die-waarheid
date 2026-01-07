"""
FastAPI Backend for Die Waarheid React Frontend
Provides REST API endpoints for transcription, analysis, and speaker training
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pathlib import Path
from datetime import datetime
import tempfile
import logging
import os
import secrets
from typing import Optional
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Import advanced security features
try:
    from src.security import (
        add_security_headers,
        validate_request_security,
        sanitize_user_input,
        validate_file_security,
        security_validator
    )
    ADVANCED_SECURITY_AVAILABLE = True
except ImportError:
    logger.warning("Advanced security module not available, using basic security")
    ADVANCED_SECURITY_AVAILABLE = False

from src.whisper_transcriber import WhisperTranscriber
from src.forensics import ForensicsEngine
from src.speaker_identification import SpeakerIdentificationSystem
from config import AUDIO_DIR, DATA_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app
app = FastAPI(
    title="Die Waarheid API",
    description="Forensic Analysis API for WhatsApp Communications",
    version="1.0.0"
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security configuration
security = HTTPBearer()

# Get API key from environment or generate one
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    API_KEY = secrets.token_urlsafe(32)
    logger.warning(f"No API_KEY found in environment. Generated temporary key: {API_KEY}")
    logger.warning("Set API_KEY environment variable for production!")

# File size limits
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 500 * 1024 * 1024))  # 500MB default

# Pydantic models for request validation
class TranscriptionRequest(BaseModel):
    language: str = Field(default="af", pattern="^(af|en|nl)$")
    model_size: str = Field(default="small", pattern="^(tiny|small|medium|large)$")
    
    @validator('language')
    def validate_language(cls, v):
        if v not in ['af', 'en', 'nl']:
            raise ValueError('Language must be af, en, or nl')
        return v

class SpeakerInitRequest(BaseModel):
    participant_a: str = Field(..., min_length=1, max_length=100)
    participant_b: str = Field(..., min_length=1, max_length=100)
    
    @validator('participant_a', 'participant_b')
    def validate_name(cls, v):
        if not v or v.isspace():
            raise ValueError('Participant name cannot be empty')
        return v.strip()

# Authentication dependency
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API key from Authorization header"""
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return credentials.credentials

# Enhanced file validation with security
async def validate_file_security_and_size(file: UploadFile) -> None:
    """
    Comprehensive file validation including size and security checks
    
    Args:
        file: Uploaded file to validate
        
    Raises:
        HTTPException: If file validation fails
    """
    content = await file.read()
    
    # Size validation
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE/(1024*1024):.0f}MB"
        )
    
    # Advanced security validation if available
    if ADVANCED_SECURITY_AVAILABLE:
        allowed_audio_types = ['.wav', '.mp3', '.mp4', '.m4a', '.flac', '.ogg']
        validate_file_security(content, file.filename or "unknown", allowed_audio_types)
    
    await file.seek(0)  # Reset file pointer


# Legacy function for backward compatibility
async def validate_file_size(file: UploadFile) -> None:
    """Legacy file size validation (deprecated - use validate_file_security_and_size)"""
    await validate_file_security_and_size(file)

# Configure CORS with security
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://localhost:8501,http://127.0.0.1:3000,http://127.0.0.1:5173,http://127.0.0.1:8501"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Specific methods only
    allow_headers=[
        "Content-Type",
        "Authorization",
        "Accept",
        "Origin",
        "X-Requested-With"
    ],  # Specific headers only
    max_age=3600,  # Cache preflight for 1 hour
)

# Global instances
transcriber: Optional[WhisperTranscriber] = None
forensics_engine: Optional[ForensicsEngine] = None
speaker_system: Optional[SpeakerIdentificationSystem] = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global forensics_engine, speaker_system
    
    logger.info("Initializing services...")
    
    try:
        # Initialize speaker system
        speaker_system = SpeakerIdentificationSystem("MAIN_CASE")
        logger.info("Speaker identification system initialized")
    except Exception as e:
        logger.error(f"Failed to initialize speaker system: {e}")
    
    try:
        # Initialize forensics engine
        forensics_engine = ForensicsEngine(use_cache=True)
        logger.info("Forensics engine initialized")
    except Exception as e:
        logger.error(f"Failed to initialize forensics engine: {e}")
    
    # Initialize health monitoring
    try:
        from src.health_monitor import start_health_monitoring
        start_health_monitoring()
        logger.info("Health monitoring started")
    except Exception as e:
        logger.warning(f"Failed to start health monitoring: {e}")
    
    logger.info("Services initialized successfully")


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Security middleware for request validation and response headers"""
    try:
        # Validate request security if advanced security is available
        if ADVANCED_SECURITY_AVAILABLE:
            validate_request_security(request)
        
        # Process request
        response = await call_next(request)
        
        # Add security headers if advanced security is available
        if ADVANCED_SECURITY_AVAILABLE:
            response = add_security_headers(response)
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions (rate limiting, blocked IPs, etc.)
        raise
    except Exception as e:
        logger.error(f"Security middleware error: {str(e)}")
        # Continue processing on middleware errors
        response = await call_next(request)
        return response


@app.get("/")
async def root():
    """Root endpoint with security headers"""
    return {
        "name": "Die Waarheid API",
        "version": "1.0.0",
        "status": "running",
        "security": "enhanced" if ADVANCED_SECURITY_AVAILABLE else "basic"
    }


@app.get("/api/health")
@limiter.limit("30/minute")
async def health_check(request: Request):
    """Comprehensive health check endpoint with system monitoring - no authentication required"""
    try:
        # Import health monitoring features
        from src.health_monitor import get_health_summary, check_component_health
        
        # Get comprehensive health summary
        health_summary = get_health_summary()
        
        # Check component health
        component_checks = {}
        
        # Check transcriber health
        if transcriber is not None:
            def check_transcriber():
                return {
                    "loaded": transcriber.model is not None,
                    "model_size": transcriber.model_size,
                    "available_models": transcriber.AVAILABLE_MODELS
                }
            component_checks["transcriber"] = check_component_health("transcriber", check_transcriber)
        
        # Check forensics engine health
        if forensics_engine is not None:
            def check_forensics():
                return {"available": True, "engine_type": "forensics"}
            component_checks["forensics"] = check_component_health("forensics", check_forensics)
        
        # Check speaker system health
        if speaker_system is not None:
            def check_speaker_system():
                return {"available": True, "system_type": "speaker_identification"}
            component_checks["speaker_system"] = check_component_health("speaker_system", check_speaker_system)
        
        # Combine health information
        health_info = {
            "status": health_summary.get("overall_status", "unknown"),
            "timestamp": health_summary.get("timestamp", datetime.now().isoformat()),
            "system_metrics": health_summary.get("system_metrics", {}),
            "component_health": {
                name: {
                    "status": health.status,
                    "response_time_ms": health.response_time_ms,
                    "error_message": health.error_message
                }
                for name, health in component_checks.items()
            },
            "component_summary": health_summary.get("component_summary", {}),
            "system_info": {
                "platform": health_summary.get("system_info", {}).get("platform", "unknown"),
                "architecture": health_summary.get("system_info", {}).get("architecture", "unknown"),
                "python_version": health_summary.get("system_info", {}).get("python_version", "unknown"),
                "hostname": health_summary.get("system_info", {}).get("hostname", "unknown"),
                "uptime": health_summary.get("system_metrics", {}).get("uptime_human", "unknown")
            },
            "services": {
                "transcriber": transcriber is not None,
                "forensics": forensics_engine is not None,
                "speaker_system": speaker_system is not None,
                "security": "enhanced" if ADVANCED_SECURITY_AVAILABLE else "basic"
            }
        }
        
    except ImportError:
        # Fallback to basic health check if health monitoring not available
        health_info = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "transcriber": transcriber is not None,
                "forensics": forensics_engine is not None,
                "speaker_system": speaker_system is not None,
                "security": "enhanced" if ADVANCED_SECURITY_AVAILABLE else "basic"
            },
            "health_monitoring": {
                "available": False,
                "message": "Health monitoring module not available"
            }
        }
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        health_info = {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "services": {
                "transcriber": transcriber is not None,
                "forensics": forensics_engine is not None,
                "speaker_system": speaker_system is not None,
                "security": "enhanced" if ADVANCED_SECURITY_AVAILABLE else "basic"
            }
        }
    
    # Add GPU information if available
    try:
        from src.gpu_manager import is_gpu_available, get_optimal_device, gpu_manager
        
        health_info["gpu"] = {
            "available": is_gpu_available(),
            "optimal_device": get_optimal_device(),
            "optimization_enabled": True
        }
        
        # Add basic GPU stats if available
        if is_gpu_available():
            gpu_stats = gpu_manager.get_performance_stats()
            health_info["gpu"]["device_count"] = gpu_stats.get("device_count", 0)
            health_info["gpu"]["cuda_version"] = gpu_stats.get("cuda_version", "Unknown")
            
            # Add memory info for optimal device
            memory_info = gpu_manager.get_memory_info()
            if memory_info.get("available", False):
                health_info["gpu"]["memory"] = {
                    "total_mb": memory_info.get("total_mb", 0),
                    "free_mb": memory_info.get("free_mb", 0),
                    "utilization_percent": memory_info.get("utilization_percent", 0)
                }
        
    except ImportError:
        health_info["gpu"] = {
            "available": False,
            "optimization_enabled": False,
            "message": "GPU optimization module not available"
        }
    except Exception as e:
        logger.warning(f"Error getting GPU information for health check: {e}")
        health_info["gpu"] = {
            "available": False,
            "optimization_enabled": False,
            "error": str(e)
        }
    
    # Add model validation information if available
    try:
        from src.model_validator import get_all_models_validation_status
        
        models_status = get_all_models_validation_status()
        health_info["models"] = {
            "validation_enabled": True,
            "whisper_version": models_status.get("whisper_version", "unknown"),
            "models_count": len(models_status.get("models", {})),
            "validation_interval_hours": models_status.get("validation_interval_hours", 24)
        }
        
        # Add basic model status summary
        models_summary = {"valid": 0, "invalid": 0, "not_validated": 0, "missing": 0}
        for model_name, model_status in models_status.get("models", {}).items():
            if not model_status.get("exists", False):
                models_summary["missing"] += 1
            elif model_status.get("validation_status") == "valid":
                models_summary["valid"] += 1
            elif model_status.get("validation_status") == "invalid":
                models_summary["invalid"] += 1
            else:
                models_summary["not_validated"] += 1
        
        health_info["models"]["summary"] = models_summary
        
        # Add transcriber model info if available
        if transcriber is not None:
            try:
                transcriber_validation = transcriber.get_model_validation_info()
                health_info["models"]["current_transcriber_model"] = {
                    "model_size": transcriber_validation.get("model_size"),
                    "model_loaded": transcriber_validation.get("model_loaded", False),
                    "validation_available": transcriber_validation.get("model_validation_available", False)
                }
            except Exception as e:
                logger.debug(f"Error getting transcriber validation info for health check: {e}")
        
    except ImportError:
        health_info["models"] = {
            "validation_enabled": False,
            "message": "Model validation module not available"
        }
    except Exception as e:
        logger.warning(f"Error getting model validation information for health check: {e}")
        health_info["models"] = {
            "validation_enabled": False,
            "error": str(e)
        }
    
    return health_info


@app.get("/api/security/status")
@limiter.limit("10/minute")
async def security_status(request: Request, api_key: str = Depends(verify_api_key)):
    """
    Get security system status and statistics - requires authentication
    
    Returns:
        Security configuration and statistics
    """
    if not ADVANCED_SECURITY_AVAILABLE:
        return {
            "security_enabled": False,
            "message": "Advanced security module not available"
        }
    
    try:
        security_report = security_validator.get_security_report()
        return {
            "security_enabled": True,
            "timestamp": datetime.now().isoformat(),
            **security_report
        }
    except Exception as e:
        logger.error(f"Error getting security status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving security status"
        )


@app.get("/api/gpu/status")
@limiter.limit("10/minute")
async def gpu_status(request: Request, api_key: str = Depends(verify_api_key)):
    """
    Get comprehensive GPU status and performance information - requires authentication
    
    Returns:
        GPU configuration, performance statistics, and optimization settings
    """
    try:
        from src.gpu_manager import is_gpu_available, gpu_manager
        
        if not is_gpu_available():
            return {
                "gpu_available": False,
                "message": "GPU not available or optimization disabled",
                "timestamp": datetime.now().isoformat()
            }
        
        # Get comprehensive GPU statistics
        gpu_stats = gpu_manager.get_performance_stats()
        
        # Get memory information for all devices
        memory_info = gpu_manager.get_memory_info()
        
        # Get optimization settings for different model sizes
        optimization_settings = {}
        for model_size in ["tiny", "small", "medium", "large"]:
            optimization_settings[model_size] = gpu_manager.optimize_model_loading(model_size)
        
        gpu_status_info = {
            "gpu_available": True,
            "timestamp": datetime.now().isoformat(),
            "performance_stats": gpu_stats,
            "memory_info": memory_info,
            "optimization_settings": optimization_settings,
            "environment_config": {
                "ENABLE_GPU": os.getenv("ENABLE_GPU", "true"),
                "FORCE_CPU": os.getenv("FORCE_CPU", "false"),
                "GPU_MEMORY_FRACTION": os.getenv("GPU_MEMORY_FRACTION", "0.8"),
                "GPU_MEMORY_GROWTH": os.getenv("GPU_MEMORY_GROWTH", "true")
            }
        }
        
        # Add transcriber GPU info if available
        if transcriber is not None:
            try:
                transcriber_gpu_info = transcriber.get_gpu_performance_info()
                gpu_status_info["transcriber_gpu_info"] = transcriber_gpu_info
            except Exception as e:
                logger.warning(f"Error getting transcriber GPU info: {e}")
                gpu_status_info["transcriber_gpu_info"] = {"error": str(e)}
        
        return gpu_status_info
        
    except ImportError:
        return {
            "gpu_available": False,
            "message": "GPU optimization module not available",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting GPU status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving GPU status"
        )


@app.get("/api/models/status")
@limiter.limit("10/minute")
async def models_status(request: Request, api_key: str = Depends(verify_api_key)):
    """
    Get comprehensive model validation status - requires authentication
    
    Returns:
        Model validation status for all models
    """
    try:
        # Import model validation features
        from src.model_validator import get_all_models_validation_status
        
        models_status_info = {
            "timestamp": datetime.now().isoformat(),
            "model_validation_available": True,
            **get_all_models_validation_status()
        }
        
        # Add transcriber model validation info if available
        if transcriber is not None:
            try:
                transcriber_validation_info = transcriber.get_model_validation_info()
                models_status_info["transcriber_model_validation"] = transcriber_validation_info
            except Exception as e:
                logger.warning(f"Error getting transcriber model validation info: {e}")
                models_status_info["transcriber_model_validation"] = {"error": str(e)}
        
        return models_status_info
        
    except ImportError:
        return {
            "timestamp": datetime.now().isoformat(),
            "model_validation_available": False,
            "message": "Model validation module not available"
        }
    except Exception as e:
        logger.error(f"Error getting models status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving models status"
        )


@app.post("/api/models/{model_name}/validate")
@limiter.limit("5/minute")
async def validate_model_endpoint(
    model_name: str,
    request: Request,
    api_key: str = Depends(verify_api_key),
    force: bool = False
):
    """
    Validate a specific model - requires authentication
    
    Args:
        model_name: Name of model to validate (tiny, small, medium, large)
        force: Force validation even if recently validated
    
    Returns:
        Model validation results
    """
    try:
        # Import model validation features
        from src.model_validator import validate_model
        
        # Validate model name
        valid_models = ["tiny", "base", "small", "medium", "large"]
        if model_name not in valid_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model name. Must be one of: {', '.join(valid_models)}"
            )
        
        logger.info(f"Validating model {model_name} (force={force})")
        
        # Perform validation
        validation_result = validate_model(model_name, force=force)
        
        validation_response = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "force_validation": force,
            "validation_result": {
                "is_valid": validation_result.is_valid,
                "checksum_valid": validation_result.checksum_valid,
                "version_compatible": validation_result.version_compatible,
                "file_exists": validation_result.file_exists,
                "file_readable": validation_result.file_readable,
                "size_valid": validation_result.size_valid,
                "corruption_detected": validation_result.corruption_detected,
                "security_valid": validation_result.security_valid,
                "performance_valid": validation_result.performance_valid,
                "validation_time": validation_result.validation_time.isoformat(),
                "errors": validation_result.errors,
                "warnings": validation_result.warnings,
                "details": validation_result.details
            }
        }
        
        logger.info(f"Model {model_name} validation completed: {'PASSED' if validation_result.is_valid else 'FAILED'}")
        
        return validation_response
        
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Model validation module not available"
        )
    except Exception as e:
        logger.error(f"Error validating model {model_name}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error validating model: {str(e)}"
        )


@app.get("/api/models/{model_name}/status")
@limiter.limit("10/minute")
async def model_status(
    model_name: str,
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    Get status for a specific model - requires authentication
    
    Args:
        model_name: Name of model to check
    
    Returns:
        Model status information
    """
    try:
        # Import model validation features
        from src.model_validator import get_model_validation_status
        
        # Validate model name
        valid_models = ["tiny", "base", "small", "medium", "large"]
        if model_name not in valid_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model name. Must be one of: {', '.join(valid_models)}"
            )
        
        model_status_info = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            **get_model_validation_status(model_name)
        }
        
        return model_status_info
        
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Model validation module not available"
        )
    except Exception as e:
        logger.error(f"Error getting model {model_name} status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model status: {str(e)}"
        )


@app.get("/api/health/detailed")
@limiter.limit("10/minute")
async def detailed_health_check(request: Request, api_key: str = Depends(verify_api_key)):
    """
    Get detailed health information with full system metrics - requires authentication
    
    Returns:
        Comprehensive health status with system metrics, trends, and component details
    """
    try:
        # Import health monitoring features
        from src.health_monitor import get_health_summary, check_component_health
        
        # Get comprehensive health summary
        health_summary = get_health_summary()
        
        # Perform detailed component health checks
        component_checks = {}
        
        # Check transcriber health with detailed info
        if transcriber is not None:
            def check_transcriber_detailed():
                try:
                    validation_info = transcriber.get_model_validation_info() if hasattr(transcriber, 'get_model_validation_info') else {}
                    gpu_info = transcriber.get_gpu_performance_info() if hasattr(transcriber, 'get_gpu_performance_info') else {}
                    
                    return {
                        "loaded": transcriber.model is not None,
                        "model_size": transcriber.model_size,
                        "available_models": transcriber.AVAILABLE_MODELS,
                        "validation_info": validation_info,
                        "gpu_info": gpu_info
                    }
                except Exception as e:
                    return {"error": str(e), "loaded": transcriber.model is not None}
            
            component_checks["transcriber"] = check_component_health("transcriber", check_transcriber_detailed)
        
        # Check forensics engine health
        if forensics_engine is not None:
            def check_forensics_detailed():
                return {
                    "available": True,
                    "engine_type": "forensics",
                    "features": ["audio_analysis", "statement_verification"]
                }
            component_checks["forensics"] = check_component_health("forensics", check_forensics_detailed)
        
        # Check speaker system health
        if speaker_system is not None:
            def check_speaker_system_detailed():
                return {
                    "available": True,
                    "system_type": "speaker_identification",
                    "features": ["voice_analysis", "speaker_recognition"]
                }
            component_checks["speaker_system"] = check_component_health("speaker_system", check_speaker_system_detailed)
        
        # Add GPU health check if available
        try:
            from src.gpu_manager import is_gpu_available, gpu_manager
            
            def check_gpu_health():
                if is_gpu_available():
                    return {
                        "available": True,
                        "performance_stats": gpu_manager.get_performance_stats(),
                        "memory_info": gpu_manager.get_memory_info()
                    }
                else:
                    return {"available": False, "reason": "No GPU detected"}
            
            component_checks["gpu"] = check_component_health("gpu", check_gpu_health)
        except ImportError:
            pass
        
        # Add model validation health check if available
        try:
            from src.model_validator import get_all_models_validation_status
            
            def check_model_validation():
                return get_all_models_validation_status()
            
            component_checks["model_validation"] = check_component_health("model_validation", check_model_validation)
        except ImportError:
            pass
        
        # Add security health check if available
        if ADVANCED_SECURITY_AVAILABLE:
            def check_security():
                try:
                    from src.security import security_validator
                    return security_validator.get_security_report()
                except Exception as e:
                    return {"error": str(e), "available": False}
            
            component_checks["security"] = check_component_health("security", check_security)
        
        # Combine detailed health information
        detailed_health = {
            **health_summary,
            "detailed_component_health": {
                name: {
                    "status": health.status,
                    "last_check": health.last_check.isoformat(),
                    "response_time_ms": health.response_time_ms,
                    "error_message": health.error_message,
                    "details": health.details
                }
                for name, health in component_checks.items()
            }
        }
        
        return detailed_health
        
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Health monitoring module not available"
        )
    except Exception as e:
        logger.error(f"Error getting detailed health: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving detailed health information"
        )


@app.get("/api/health/metrics")
@limiter.limit("20/minute")
async def health_metrics(request: Request, api_key: str = Depends(verify_api_key), limit: int = 50):
    """
    Get system metrics history - requires authentication
    
    Args:
        limit: Maximum number of metrics entries to return (default: 50, max: 200)
    
    Returns:
        Historical system metrics data
    """
    try:
        # Import health monitoring features
        from src.health_monitor import get_metrics_history
        
        # Validate limit
        if limit > 200:
            limit = 200
        elif limit < 1:
            limit = 1
        
        metrics_history = get_metrics_history(limit)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics_count": len(metrics_history),
            "limit": limit,
            "metrics": metrics_history
        }
        
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Health monitoring module not available"
        )
    except Exception as e:
        logger.error(f"Error getting health metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving health metrics"
        )


@app.post("/api/health/export")
@limiter.limit("5/minute")
async def export_health_report(request: Request, api_key: str = Depends(verify_api_key)):
    """
    Export comprehensive health report - requires authentication
    
    Returns:
        Health report export information
    """
    try:
        # Import health monitoring features
        from src.health_monitor import health_monitor
        
        # Export health report
        report_path = health_monitor.export_health_report()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "report_exported": True,
            "report_path": str(report_path),
            "report_size_bytes": report_path.stat().st_size if report_path.exists() else 0
        }
        
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Health monitoring module not available"
        )
    except Exception as e:
        logger.error(f"Error exporting health report: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error exporting health report"
        )


@app.post("/api/health/monitoring/start")
@limiter.limit("5/minute")
async def start_health_monitoring(request: Request, api_key: str = Depends(verify_api_key)):
    """
    Start background health monitoring - requires authentication
    
    Returns:
        Monitoring start status
    """
    try:
        # Import health monitoring features
        from src.health_monitor import start_health_monitoring
        
        start_health_monitoring()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "monitoring_started": True,
            "message": "Background health monitoring started"
        }
        
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Health monitoring module not available"
        )
    except Exception as e:
        logger.error(f"Error starting health monitoring: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error starting health monitoring"
        )


@app.post("/api/health/monitoring/stop")
@limiter.limit("5/minute")
async def stop_health_monitoring(request: Request, api_key: str = Depends(verify_api_key)):
    """
    Stop background health monitoring - requires authentication
    
    Returns:
        Monitoring stop status
    """
    try:
        # Import health monitoring features
        from src.health_monitor import stop_health_monitoring
        
        stop_health_monitoring()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "monitoring_stopped": True,
            "message": "Background health monitoring stopped"
        }
        
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Health monitoring module not available"
        )
    except Exception as e:
        logger.error(f"Error stopping health monitoring: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error stopping health monitoring"
        )


@app.post("/api/transcribe")
@limiter.limit("10/minute")
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(...),
    language: str = Form("af"),
    model_size: str = Form("small"),
    api_key: str = Depends(verify_api_key)
):
    """Transcribe audio file to text - requires authentication"""
    global transcriber
    
    tmp_path = None
    try:
        # Enhanced file validation with security checks
        await validate_file_security_and_size(file)
        
        # Sanitize and validate input parameters
        if ADVANCED_SECURITY_AVAILABLE:
            language = sanitize_user_input(language, max_length=10)
            model_size = sanitize_user_input(model_size, max_length=20)
        
        # Validate parameters
        if language not in ['af', 'en', 'nl']:
            raise HTTPException(status_code=400, detail="Invalid language. Must be af, en, or nl")
        
        if model_size not in ['tiny', 'small', 'medium', 'large']:
            raise HTTPException(status_code=400, detail="Invalid model size")
        
        # Initialize transcriber if needed
        if transcriber is None or transcriber.model_size != model_size:
            logger.info(f"Loading Whisper {model_size} model...")
            transcriber = WhisperTranscriber(model_size)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)
        
        # Transcribe with timeout
        logger.info(f"Transcribing {file.filename} with language={language}")
        
        # Use asyncio for timeout handling
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_event_loop()
        
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    transcriber.transcribe,
                    tmp_path,
                    language
                ),
                timeout=300  # 5 minutes timeout
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="Transcription timeout after 5 minutes"
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Always clean up temp file
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception as e:
                logger.error(f"Failed to delete temp file: {e}")


@app.post("/api/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """Perform forensic audio analysis"""
    global forensics_engine
    
    if forensics_engine is None:
        forensics_engine = ForensicsEngine(use_cache=True)
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)
        
        # Analyze
        logger.info(f"Analyzing {file.filename}")
        result = forensics_engine.analyze(tmp_path)
        
        # Clean up
        tmp_path.unlink()
        
        return result
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/speakers")
async def get_speakers():
    """Get all speaker profiles"""
    global speaker_system
    
    if speaker_system is None:
        speaker_system = SpeakerIdentificationSystem("MAIN_CASE")
    
    try:
        participants = speaker_system.get_all_participants()
        
        return [
            {
                "participant_id": p.participant_id,
                "primary_username": p.primary_username,
                "assigned_role": p.assigned_role.value,
                "voice_note_count": p.voice_note_count,
                "message_count": p.message_count,
                "confidence_score": p.confidence_score,
                "voice_fingerprints": len(p.voice_fingerprints)
            }
            for p in participants
        ]
        
    except Exception as e:
        logger.error(f"Get speakers error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/speakers/initialize")
async def initialize_speakers(data: dict):
    """Initialize investigation with two participants"""
    global speaker_system
    
    if speaker_system is None:
        speaker_system = SpeakerIdentificationSystem("MAIN_CASE")
    
    try:
        participant_a = data.get("participant_a")
        participant_b = data.get("participant_b")
        
        if not participant_a or not participant_b:
            raise HTTPException(status_code=400, detail="Both participant names required")
        
        logger.info(f"Initializing investigation: {participant_a} and {participant_b}")
        participant_a_id, participant_b_id = speaker_system.initialize_investigation(
            participant_a, participant_b
        )
        
        return {
            "success": True,
            "message": f"Investigation initialized: {participant_a_id}, {participant_b_id}",
            "participant_a_id": participant_a_id,
            "participant_b_id": participant_b_id
        }
        
    except Exception as e:
        logger.error(f"Initialize speakers error: {e}")
        return {
            "success": False,
            "message": str(e)
        }


@app.post("/api/speakers/train")
async def train_speaker(
    file: UploadFile = File(...),
    participant_id: str = Form(...)
):
    """Train speaker with voice sample"""
    global speaker_system
    
    if speaker_system is None:
        speaker_system = SpeakerIdentificationSystem("MAIN_CASE")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)
        
        # Get participant username
        participants = speaker_system.get_all_participants()
        participant = next((p for p in participants if p.participant_id == participant_id), None)
        
        if not participant:
            raise HTTPException(status_code=404, detail="Participant not found")
        
        # Train
        logger.info(f"Training {participant_id} with {file.filename}")
        result_id = speaker_system.register_voice_note(
            participant.primary_username,
            tmp_path,
            datetime.now()
        )
        
        # Clean up
        tmp_path.unlink()
        
        return {
            "success": True,
            "message": f"Voice sample trained successfully",
            "participant_id": result_id
        }
        
    except Exception as e:
        logger.error(f"Train speaker error: {e}")
        return {
            "success": False,
            "message": str(e)
        }


@app.get("/api/files/count")
async def get_file_count():
    """Get count of audio files"""
    try:
        count = 0
        for ext in ['mp3', 'wav', 'opus', 'ogg', 'm4a', 'aac']:
            count += len(list(AUDIO_DIR.rglob(f"*.{ext}")))
        
        return {"count": count}
        
    except Exception as e:
        logger.error(f"File count error: {e}")
        return {"count": 0}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
