"""
FastAPI Backend Server
Exposes the AI pipeline as REST API endpoints

Features:
- Multi-language support (99+ languages)
- Long video support (10+ minutes)
- High-accuracy grammar correction
- REST API with interactive docs
"""

import os
import uuid
import shutil
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import our AI pipeline
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup FFmpeg path before importing any modules that use it
from utils.speech_to_text import setup_ffmpeg_path, SUPPORTED_LANGUAGES
setup_ffmpeg_path()

from backend.pipeline import AIPipeline
from utils.text_cleaner import clean_text
from utils.text_improver import TextImprover

# ============================================================
# Configuration
# ============================================================

UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".3gp"}
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}
MAX_FILE_SIZE = 1024 * 1024 * 1024  # 1GB for long videos

# ============================================================
# Initialize FastAPI App
# ============================================================

app = FastAPI(
    title="AI Video to Fluent Text API",
    description="""
    🎬 Convert video speech to fluent, professional text.
    
    ## Features
    - **Multi-language support**: 99+ languages with auto-detection
    - **Long video support**: Handles 10+ minute videos
    - **High accuracy**: Grammarly CoEdit model for grammar correction
    - **Speech-to-Text**: OpenAI Whisper neural network
    
    ## Supported Languages
    English, Urdu, Hindi, Arabic, Spanish, French, German, Chinese, and 90+ more!
    
    ## Neural Networks Used
    - **Whisper**: OpenAI's transformer-based speech recognition (680K hours training)
    - **CoEdit**: Grammarly's grammar correction model
    """,
    version="2.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Global Pipeline Instance (Lazy Loading)
# ============================================================

_pipeline: Optional[AIPipeline] = None
_text_improver: Optional[TextImprover] = None


def get_pipeline() -> AIPipeline:
    """Get or initialize the AI pipeline."""
    global _pipeline
    if _pipeline is None:
        print("🔄 Initializing AI Pipeline (first request may take a moment)...")
        _pipeline = AIPipeline(whisper_model="base", use_fast_grammar=False)
    return _pipeline


def get_text_improver() -> TextImprover:
    """Get or initialize the text improver."""
    global _text_improver
    if _text_improver is None:
        print("🔄 Initializing Text Improver...")
        _text_improver = TextImprover()
    return _text_improver


# ============================================================
# Request/Response Models
# ============================================================

class TextRequest(BaseModel):
    """Request model for text improvement."""
    text: str
    clean_first: bool = True
    task: str = "Fix grammar"  # Can be: "Fix grammar", "Make this more fluent", "Rewrite this professionally"


class TextResponse(BaseModel):
    """Response model for text improvement."""
    original_text: str
    cleaned_text: str
    improved_text: str


class VideoResponse(BaseModel):
    """Response model for video processing."""
    success: bool
    original_text: str
    cleaned_text: str
    improved_text: str
    detected_language: str = ""
    language_name: str = ""
    audio_duration: Optional[float] = None
    duration_formatted: str = ""
    message: str = ""


class LanguagesResponse(BaseModel):
    """Response model for supported languages."""
    languages: dict


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str
    supported_languages: int = len(SUPPORTED_LANGUAGES)


# ============================================================
# Helper Functions
# ============================================================

def validate_file_extension(filename: str, allowed: set) -> bool:
    """Check if file extension is allowed."""
    ext = Path(filename).suffix.lower()
    return ext in allowed


def cleanup_file(file_path: str):
    """Remove temporary file."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Warning: Could not cleanup file {file_path}: {e}")


async def save_upload_file(upload_file: UploadFile) -> str:
    """Save uploaded file to disk."""
    # Generate unique filename
    file_ext = Path(upload_file.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = UPLOAD_DIR / unique_filename
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    return str(file_path)


# ============================================================
# API Endpoints
# ============================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """API root - health check."""
    return HealthResponse(
        status="online",
        message="AI Video to Fluent Text API is running! Visit /docs for API documentation."
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="All systems operational"
    )


@app.post("/api/improve-text", response_model=TextResponse)
async def improve_text(request: TextRequest):
    """
    Improve text grammar and fluency.
    
    - **text**: The text to improve
    - **clean_first**: Whether to clean the text before improving (remove fillers, etc.)
    - **task**: Type of improvement:
        - "Fix grammar" (default)
        - "Make this more fluent"
        - "Rewrite this professionally"
    """
    try:
        improver = get_text_improver()
        
        original = request.text
        
        # Clean text if requested
        cleaned = clean_text(original) if request.clean_first else original
        
        # Improve text with specified task
        improved = improver.improve(cleaned, task=request.task)
        
        return TextResponse(
            original_text=original,
            cleaned_text=cleaned,
            improved_text=improved
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text improvement failed: {str(e)}")


@app.get("/api/languages", response_model=LanguagesResponse)
async def get_languages():
    """Get list of supported languages for speech recognition."""
    return LanguagesResponse(languages=SUPPORTED_LANGUAGES)


@app.post("/api/upload-video", response_model=VideoResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file to process"),
    language: Optional[str] = Form(None, description="Language code (e.g., 'en', 'ur', 'hi') or leave empty for auto-detect"),
    translate_to_english: bool = Form(False, description="Translate to English if not already in English"),
    task: str = Form("Fix grammar", description="Improvement type: 'Fix grammar', 'Make this more fluent', 'Rewrite this professionally'")
):
    """
    Upload a video file and convert speech to fluent text.
    
    **Supports 10+ minute videos!**
    
    Supported formats: MP4, AVI, MOV, MKV, WebM, FLV, WMV, 3GP
    
    Language codes:
    - 'en': English
    - 'ur': Urdu  
    - 'hi': Hindi
    - 'ar': Arabic
    - Leave empty for auto-detection
    
    Returns:
    - Original transcribed text
    - Cleaned text (fillers removed)
    - Improved fluent text
    - Detected language
    """
    # Validate file extension
    if not validate_file_extension(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
        )
    
    file_path = None
    try:
        # Save uploaded file
        file_path = await save_upload_file(file)
        
        # Process with pipeline
        pipeline = get_pipeline()
        result = pipeline.process_video(
            file_path, 
            keep_audio=False,
            language=language,
            translate_to_english=translate_to_english,
            task=task
        )
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_file, file_path)
        
        return VideoResponse(
            success=True,
            original_text=result["raw_text"],
            cleaned_text=result["cleaned_text"],
            improved_text=result["improved_text"],
            detected_language=result.get("detected_language", ""),
            language_name=result.get("language_name", ""),
            audio_duration=result.get("audio_info", {}).get("duration"),
            duration_formatted=result.get("audio_info", {}).get("duration_formatted", ""),
            message="Video processed successfully!"
        )
        
    except Exception as e:
        # Cleanup on error
        if file_path:
            cleanup_file(file_path)
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")


@app.post("/api/upload-audio", response_model=VideoResponse)
async def upload_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio file to process"),
    language: Optional[str] = Form(None, description="Language code or leave empty for auto-detect"),
    translate_to_english: bool = Form(False, description="Translate to English"),
    task: str = Form("Fix grammar", description="Improvement type: 'Fix grammar', 'Make this more fluent', 'Rewrite this professionally'")
):
    """
    Upload an audio file and convert speech to fluent text.
    
    **Supports long audio files!**
    
    Supported formats: MP3, WAV, M4A, FLAC, OGG, AAC, WMA
    """
    # Validate file extension
    if not validate_file_extension(file.filename, ALLOWED_AUDIO_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"
        )
    
    file_path = None
    try:
        # Save uploaded file
        file_path = await save_upload_file(file)
        
        # Process with pipeline (use video method which also handles audio)
        pipeline = get_pipeline()
        result = pipeline.process_video(
            file_path, 
            keep_audio=False,
            language=language,
            translate_to_english=translate_to_english,
            task=task
        )
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_file, file_path)
        
        return VideoResponse(
            success=True,
            original_text=result["raw_text"],
            cleaned_text=result["cleaned_text"],
            improved_text=result["improved_text"],
            detected_language=result.get("detected_language", ""),
            language_name=result.get("language_name", ""),
            audio_duration=result.get("audio_info", {}).get("duration"),
            duration_formatted=result.get("audio_info", {}).get("duration_formatted", ""),
            message="Audio processed successfully!"
        )
        
    except Exception as e:
        # Cleanup on error
        if file_path:
            cleanup_file(file_path)
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")


# ============================================================
# Run Server
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 Starting AI Video to Fluent Text API Server")
    print("=" * 60)
    print("\n📍 API Documentation: http://localhost:8000/docs")
    print("📍 Alternative docs: http://localhost:8000/redoc")
    print("\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload for production stability
        timeout_keep_alive=600,  # 10 minute timeout for long video processing
        workers=1  # Single worker to avoid memory issues
    )
