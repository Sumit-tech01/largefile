"""
Whisper Service for Speech-to-Text transcription.
Provides audio transcription using OpenAI's Whisper model.
"""

import os
import logging
import tempfile
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Union
import time

import numpy as np
import torch
import whisper
from scipy.io import wavfile

from ..config import config
from ..models.schemas import TranscriptionResult

logger = logging.getLogger(__name__)


class WhisperService:
    """
    Service for speech-to-text transcription using OpenAI Whisper.
    
    Features:
    - Load and cache Whisper model
    - Transcribe audio files
    - Support for multiple model sizes
    - Device optimization (CPU/CUDA)
    """

    _instance: Optional["WhisperService"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "WhisperService":
        """Singleton pattern to avoid reloading model."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the Whisper service."""
        if self._initialized:
            return

        self.model: Optional[whisper.Whisper] = None
        self.model_size: str = config.WHISPER_MODEL_SIZE
        self.device: str = config.WHISPER_DEVICE
        self._load_time: Optional[float] = None
        self._model_loaded: bool = False

        self._initialized = True
        logger.info(f"WhisperService initialized with model size: {self.model_size}")

    def load_model(self, force_reload: bool = False) -> None:
        """
        Load the Whisper model.
        
        Args:
            force_reload: If True, reload even if already loaded
        """
        if self._model_loaded and not force_reload:
            logger.info("Whisper model already loaded, skipping reload")
            return

        start_time = time.time()

        try:
            # Determine device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading Whisper '{self.model_size}' model on {self.device}")

            # Load model with specified options
            model_kwargs = {
                "model": self.model_size,
                "device": self.device,
                "download_root": str(config.MODEL_CACHE_DIR_ABSOLUTE),
            }

            # Add compute type for CUDA optimization
            if self.device == "cuda":
                compute_type = config.WHISPER_COMPUTE_TYPE
                if compute_type == "default":
                    compute_type = "float16" if torch.cuda.is_available() else "float32"
                model_kwargs["compute_type"] = compute_type

            self.model = whisper.load_model(**model_kwargs)

            self._load_time = time.time() - start_time
            self._model_loaded = True

            logger.info(
                f"Whisper model loaded successfully in {self._load_time:.2f}s"
            )

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise RuntimeError(f"Whisper model loading failed: {e}")

    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        verbose: bool = False,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to the audio file
            language: Language code (e.g., 'en'). Auto-detected if None
            verbose: Enable verbose logging
            
        Returns:
            TranscriptionResult with text and metadata
        """
        start_time = time.time()

        # Ensure model is loaded
        if not self._model_loaded:
            self.load_model()

        # Convert to Path if string
        audio_path = Path(audio_path)

        # Validate file exists
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Transcribing audio file: {audio_path.name}")

        try:
            # Run transcription
            options = {}
            if language:
                options["language"] = language

            result = self.model.transcribe(
                str(audio_path),
                verbose=verbose,
                **options
            )

            # Extract relevant information
            text = result.get("text", "").strip()
            language_detected = result.get("language", language or "unknown")

            # Calculate confidence from log probabilities if available
            confidence = None
            if "log_probs" in result and result["log_probs"] is not None:
                # Average log probability converted to confidence
                avg_log_prob = np.mean(result["log_probs"])
                confidence = float(np.exp(avg_log_prob))  # Convert to 0-1 scale

            processing_time = time.time() - start_time

            logger.info(
                f"Transcription complete: {len(text)} chars in {processing_time:.2f}s"
            )

            return TranscriptionResult(
                text=text,
                language=language_detected,
                confidence=confidence,
                duration=result.get("duration"),
            )

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Transcription error: {e}")

    def transcribe_numpy(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio from numpy array.
        
        Args:
            audio_data: NumPy array of audio samples
            sample_rate: Sample rate of audio
            language: Language code
            
        Returns:
            TranscriptionResult with text and metadata
        """
        start_time = time.time()

        if not self._model_loaded:
            self.load_model()

        logger.info(f"Transcribing numpy array: shape={audio_data.shape}")

        try:
            # Create temporary file for whisper
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as tmp_file:
                temp_path = tmp_file.name

            try:
                # Save numpy array to WAV file
                wavfile.write(temp_path, sample_rate, audio_data)

                # Transcribe using the saved file
                result = self.transcribe(temp_path, language)

                return result

            finally:
                # Cleanup temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        except Exception as e:
            logger.error(f"NumPy transcription failed: {e}")
            raise RuntimeError(f"NumPy transcription error: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self._model_loaded:
            return {"status": "not_loaded"}

        info = {
            "model_size": self.model_size,
            "device": self.device,
            "is_loaded": self._model_loaded,
            "load_time_seconds": self._load_time,
            "num_parameters": self.model.num_parameters if self.model else None,
        }

        # Add CUDA info if available
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_mb"] = torch.cuda.get_device_properties(
                0
            ).total_memory / (1024**2)

        return info

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self._model_loaded = False

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Whisper model unloaded")


# Global instance
whisper_service = WhisperService()

