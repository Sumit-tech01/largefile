"""
REST API routes for Voice Sentiment Analysis application.
Provides endpoints for audio upload, analysis, and health checks.
"""

import logging
import uuid
import time
from pathlib import Path
from typing import Optional

from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename

from ..config import config
from ..models.schemas import (
    AudioUploadRequest,
    AudioAnalysisResponse,
    ErrorResponse,
    HealthResponse,
    SentimentLabel,
)
from ..services.whisper_service import whisper_service
from ..services.sentiment_service import sentiment_service
from ..utils.audio_utils import (
    validate_audio_file,
    convert_to_wav,
    preprocess_audio,
    AudioProcessingError,
)

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api/v1")


# ==================== Health & Status Endpoints ====================

@api_bp.route("/health", methods=["GET"])
def health_check() -> tuple[dict, int]:
    """
    Health check endpoint.
    
    Returns:
        JSON response with service health status
    """
    try:
        whisper_info = whisper_service.get_model_info()
        sentiment_info = sentiment_service.get_model_info()

        services = {
            "whisper": {
                "status": "healthy" if whisper_info.get("is_loaded") else "loading",
                "model": whisper_info.get("model_size", "unknown"),
                "device": whisper_info.get("device", "unknown"),
            },
            "sentiment": {
                "status": "healthy" if sentiment_info.get("is_loaded") else "loading",
                "model": sentiment_info.get("model_name", "unknown"),
                "device": sentiment_info.get("device", "unknown"),
            },
        }

        # Check if all services are ready
        all_healthy = all(
            s.get("status") == "healthy" for s in services.values()
        )

        response = HealthResponse(
            status="healthy" if all_healthy else "degraded",
            version=config.APP_VERSION,
            services=services,
        )

        return response.model_dump(), 200

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return ErrorResponse(
            error="Health check failed",
            error_code="HEALTH_CHECK_ERROR",
            details={"exception": str(e)},
        ).model_dump(), 500


@api_bp.route("/models", methods=["GET"])
def get_model_info() -> tuple[dict, int]:
    """
    Get information about loaded models.
    
    Returns:
        JSON response with model information
    """
    try:
        whisper_info = whisper_service.get_model_info()
        sentiment_info = sentiment_service.get_model_info()

        return {
            "whisper": whisper_info,
            "sentiment": sentiment_info,
        }, 200

    except Exception as e:
        logger.error(f"Model info request failed: {e}")
        return ErrorResponse(
            error="Failed to get model info",
            error_code="MODEL_INFO_ERROR",
        ).model_dump(), 500


# ==================== Analysis Endpoints ====================

@api_bp.route("/analyze/upload", methods=["POST"])
def analyze_upload() -> tuple[dict, int]:
    """
    Analyze uploaded audio file.
    
    Request:
        - multipart/form-data with 'audio' field
        - Optional 'language' field for language hint
        
    Returns:
        JSON response with transcription and sentiment analysis
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    # Validate file presence
    if "audio" not in request.files:
        return ErrorResponse(
            error="No audio file provided",
            error_code="NO_AUDIO_FILE",
            details={
                "hint": "Include 'audio' field in multipart/form-data request"
            },
        ).model_dump(), 400

    audio_file = request.files["audio"]
    
    if audio_file.filename == "":
        return ErrorResponse(
            error="No file selected",
            error_code="EMPTY_FILENAME",
        ).model_dump(), 400

    # Validate file
    filename = secure_filename(audio_file.filename)
    temp_path = Path(config.UPLOAD_FOLDER_ABSOLUTE) / f"{request_id}_{filename}"
    
    try:
        # Save uploaded file
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        audio_file.save(str(temp_path))
        
        logger.info(f"Processing upload: {filename}")

        # Validate audio file
        is_valid, error_msg = validate_audio_file(temp_path)
        if not is_valid:
            raise AudioProcessingError(error_msg)

        # Convert to WAV if needed
        if temp_path.suffix.lower() != ".wav":
            wav_path, duration = convert_to_wav(temp_path)
            temp_path.unlink()  # Remove original
            temp_path = wav_path
        else:
            duration = 0.0

        # Get language from request or auto-detect
        lang_hint = request.form.get("language") or None

        # Load models if not loaded
        if not whisper_service._model_loaded:
            whisper_service.load_model()
        if not sentiment_service._model_loaded:
            sentiment_service.load_model()

        # Transcribe
        transcription = whisper_service.transcribe(
            temp_path,
            language=lang_hint,
        )

        # Analyze sentiment
        dominant, all_emotions = sentiment_service.analyze_complete(
            transcription.text
        )

        # Calculate processing time
        processing_time = time.time() - start_time

        # Build response
        response = AudioAnalysisResponse(
            success=True,
            request_id=request_id,
            transcription=transcription,
            sentiment=dominant,
            all_emotions=all_emotions,
            processing_time_seconds=round(processing_time, 3),
        )

        return response.model_dump(), 200

    except AudioProcessingError as e:
        logger.warning(f"Audio processing error: {e}")
        return ErrorResponse(
            error=str(e),
            error_code="AUDIO_PROCESSING_ERROR",
        ).model_dump(), 400

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return ErrorResponse(
            error="Analysis failed",
            error_code="ANALYSIS_ERROR",
            details={"exception": str(e)},
        ).model_dump(), 500

    finally:
        # Cleanup temp file
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass


@api_bp.route("/analyze/text", methods=["POST"])
def analyze_text() -> tuple[dict, int]:
    """
    Analyze sentiment of provided text.
    
    Request:
        - JSON body with 'text' field
        
    Returns:
        JSON response with sentiment analysis
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    try:
        data = request.get_json()
        
        if not data or "text" not in data:
            return ErrorResponse(
                error="No text provided",
                error_code="NO_TEXT",
                details={"hint": "Include 'text' field in JSON body"},
            ).model_dump(), 400

        text = data["text"].strip()
        
        if not text:
            return ErrorResponse(
                error="Empty text provided",
                error_code="EMPTY_TEXT",
            ).model_dump(), 400

        # Load model if not loaded
        if not sentiment_service._model_loaded:
            sentiment_service.load_model()

        # Analyze sentiment
        dominant, all_emotions = sentiment_service.analyze_complete(text)
        processing_time = time.time() - start_time

        return {
            "success": True,
            "request_id": request_id,
            "text": text,
            "sentiment": {
                "label": dominant.label.value,
                "score": dominant.score,
            },
            "all_emotions": {
                label.value: score for label, score in all_emotions.items()
            },
            "processing_time_seconds": round(processing_time, 3),
        }, 200

    except Exception as e:
        logger.error(f"Text analysis failed: {e}", exc_info=True)
        return ErrorResponse(
            error="Text analysis failed",
            error_code="ANALYSIS_ERROR",
        ).model_dump(), 500


@api_bp.route("/analyze/stream/start", methods=["POST"])
def start_streaming_session() -> tuple[dict, int]:
    """
    Start a streaming analysis session.
    
    Returns:
        JSON response with session ID and configuration
    """
    session_id = str(uuid.uuid4())[:8]
    
    try:
        data = request.get_json() or {}
        language = data.get("language") or None

        # Initialize session data (could be stored in Redis for production)
        session_data = {
            "session_id": session_id,
            "language": language,
            "created_at": time.time(),
            "chunks_processed": 0,
            "status": "active",
        }

        return {
            "success": True,
            "session_id": session_id,
            "language": language,
            "websocket_url": f"/socket.io/?session_id={session_id}",
            "config": {
                "chunk_duration_ms": config.AUDIO_CHUNK_DURATION_MS,
                "max_duration_seconds": config.AUDIO_MAX_DURATION,
            },
        }, 200

    except Exception as e:
        logger.error(f"Failed to start streaming session: {e}")
        return ErrorResponse(
            error="Failed to start session",
            error_code="SESSION_ERROR",
        ).model_dump(), 500


# ==================== Documentation Endpoint ====================

@api_bp.route("/docs", methods=["GET"])
def get_api_docs() -> tuple[dict, int]:
    """
    Return API documentation.
    
    Returns:
        JSON response with API documentation
    """
    docs = {
        "title": "Voice Sentiment Analysis API",
        "version": config.APP_VERSION,
        "endpoints": {
            "GET /api/v1/health": {
                "description": "Health check endpoint",
                "response": "HealthResponse with service statuses",
            },
            "GET /api/v1/models": {
                "description": "Get information about loaded models",
                "response": "Model information",
            },
            "POST /api/v1/analyze/upload": {
                "description": "Upload and analyze audio file",
                "request": {
                    "content-type": "multipart/form-data",
                    "fields": {
                        "audio": "Audio file (WAV, MP3, WebM, OGG, FLAC)",
                        "language": "(optional) Language hint (e.g., 'en')",
                    },
                },
                "response": "AudioAnalysisResponse",
            },
            "POST /api/v1/analyze/text": {
                "description": "Analyze sentiment of text",
                "request": {
                    "content-type": "application/json",
                    "body": {"text": "Text to analyze"},
                },
                "response": "Sentiment analysis results",
            },
            "POST /api/v1/analyze/stream/start": {
                "description": "Start a streaming session",
                "response": "Session ID and WebSocket URL",
            },
        },
        "websocket": {
            "url": "ws://localhost:5000/socket.io/",
            "events": {
                "audio_chunk": "Send base64-encoded audio chunks",
                "sentiment_update": "Receive incremental results",
                "sentiment_complete": "Receive final results",
            },
        },
        "supported_emotions": [e.value for e in SentimentLabel],
    }

    return docs, 200


# ==================== Error Handlers ====================

@api_bp.errorhandler(400)
def bad_request(error) -> tuple[dict, int]:
    """Handle 400 Bad Request errors."""
    return ErrorResponse(
        error=str(error.description),
        error_code="BAD_REQUEST",
    ).model_dump(), 400


@api_bp.errorhandler(404)
def not_found(error) -> tuple[dict, int]:
    """Handle 404 Not Found errors."""
    return ErrorResponse(
        error="Endpoint not found",
        error_code="NOT_FOUND",
    ).model_dump(), 404


@api_bp.errorhandler(500)
def internal_error(error) -> tuple[dict, int]:
    """Handle 500 Internal Server errors."""
    return ErrorResponse(
        error="Internal server error",
        error_code="INTERNAL_ERROR",
    ).model_dump(), 500

