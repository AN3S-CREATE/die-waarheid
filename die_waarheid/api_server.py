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

# File validation
async def validate_file_size(file: UploadFile) -> None:
    """Validate uploaded file size"""
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE/(1024*1024):.0f}MB"
        )
    await file.seek(0)  # Reset file pointer

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
    global transcriber, forensics_engine, speaker_system
    
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
    
    logger.info("Services initialized successfully")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Die Waarheid API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/api/health")
@limiter.limit("30/minute")
async def health_check(request: Request):
    """Health check endpoint - no authentication required"""
    return {
        "status": "healthy",
        "transcriber": transcriber is not None,
        "forensics": forensics_engine is not None,
        "speaker_system": speaker_system is not None,
        "timestamp": datetime.now().isoformat()
    }


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
        # Validate file size
        await validate_file_size(file)
        
        # Validate input parameters
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
