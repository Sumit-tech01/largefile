"""
Audio processing utilities for Voice Sentiment Analysis.
Handles audio format conversion, preprocessing, and validation.
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union
import subprocess
import struct

import numpy as np
import scipy.io.wavfile as wav
from scipy import signal

logger = logging.getLogger(__name__)


# Supported audio formats with their MIME types
AUDIO_FORMATS = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".webm": "audio/webm",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
}

# Target sample rate for processing
TARGET_SAMPLE_RATE = 16000

# Max audio duration in seconds
MAX_AUDIO_DURATION = 300  # 5 minutes


class AudioProcessingError(Exception):
    """Exception raised for audio processing errors."""
    pass


def validate_audio_file(
    file_path: Union[str, Path],
    max_size_mb: int = 50,
    allowed_formats: Optional[list] = None,
) -> Tuple[bool, str]:
    """
    Validate an audio file.
    
    Args:
        file_path: Path to the audio file
        max_size_mb: Maximum file size in MB
        allowed_formats: List of allowed file extensions
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if allowed_formats is None:
        allowed_formats = list(AUDIO_FORMATS.keys())

    file_path = Path(file_path)

    # Check if file exists
    if not file_path.exists():
        return False, f"File not found: {file_path}"

    # Check file extension
    suffix = file_path.suffix.lower()
    if suffix not in allowed_formats:
        return False, f"Unsupported format: {suffix}. Allowed: {allowed_formats}"

    # Check file size
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return False, f"File too large: {file_size_mb:.1f}MB > {max_size_mb}MB limit"

    return True, ""


def convert_to_wav(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    sample_rate: int = TARGET_SAMPLE_RATE,
    mono: bool = True,
) -> Tuple[Path, float]:
    """
    Convert audio file to WAV format.
    
    Args:
        input_path: Path to input audio file
        output_path: Path for output WAV file (auto-generated if None)
        sample_rate: Target sample rate
        mono: Convert to mono channel
        
    Returns:
        Tuple of (output_path, duration_seconds)
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = Path(tempfile.gettempdir()) / f"{input_path.stem}_converted.wav"
    else:
        output_path = Path(output_path)

    try:
        # Use ffmpeg for conversion if available, else use scipy
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", str(input_path),
                "-ar", str(sample_rate),
                "-ac", "1" if mono else "2",
                "-codec:pcm_s16le",
                str(output_path),
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            
            if result.returncode != 0:
                raise AudioProcessingError(f"FFmpeg error: {result.stderr}")

        except FileNotFoundError:
            # Fallback to scipy if ffmpeg not available
            logger.warning("FFmpeg not found, using scipy for conversion")
            _convert_with_scipy(input_path, output_path, sample_rate, mono)

    except Exception as e:
        logger.error(f"Audio conversion failed: {e}")
        raise AudioProcessingError(f"Failed to convert audio: {e}")

    # Get duration
    duration = get_audio_duration(output_path)
    
    return output_path, duration


def _convert_with_scipy(
    input_path: Path,
    output_path: Path,
    sample_rate: int,
    mono: bool,
) -> None:
    """Convert audio using scipy (limited format support)."""
    try:
        import soundfile as sf
        import librosa

        # Load audio
        audio, orig_sr = sf.read(str(input_path))
        
        # Resample if needed
        if orig_sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sample_rate)
        
        # Convert to mono
        if mono and len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Normalize to prevent clipping
        if audio.max() > 0:
            audio = audio / audio.max()
        
        # Save as WAV
        sf.write(str(output_path), audio, sample_rate, subtype="PCM_16")

    except Exception as e:
        raise AudioProcessingError(f"Scipy conversion failed: {e}")


def get_audio_duration(file_path: Union[str, Path]) -> float:
    """
    Get the duration of an audio file in seconds.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Duration in seconds
    """
    file_path = Path(file_path)
    
    try:
        # Try using scipy
        try:
            import soundfile as sf
            info = sf.info(str(file_path))
            return float(info.duration)
        except ImportError:
            pass
        
        # Fallback: read with scipy
        sample_rate, data = wav.read(str(file_path))
        duration = len(data) / sample_rate
        return duration
        
    except Exception as e:
        logger.warning(f"Could not get audio duration: {e}")
        return 0.0


def normalize_audio(audio_data: np.ndarray) -> np.ndarray:
    """
    Normalize audio data to prevent clipping.
    
    Args:
        audio_data: NumPy array of audio samples
        
    Returns:
        Normalized audio array
    """
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        return audio_data / max_val
    return audio_data


def remove_silence(
    audio_data: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    threshold_db: float = -40,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Remove silence from beginning and end of audio.
    
    Args:
        audio_data: NumPy array of audio samples
        sample_rate: Sample rate of audio
        threshold_db: Silence threshold in dB
        frame_length: FFT window size
        hop_length: Hop length between frames
        
    Returns:
        Audio with silence removed
    """
    # Convert threshold from dB to amplitude
    threshold = 10 ** (threshold_db / 20)
    
    # Calculate silent frames
    envelope = np.abs(
        signal.hilbert(signal.medfilt(np.abs(audio_data), 101))
    )
    
    # Find non-silent regions
    non_silent = envelope > threshold
    
    # Get indices of non-silent regions
    indices = np.where(non_silent)[0]
    
    if len(indices) == 0:
        return audio_data
    
    # Get start and end points
    start = max(0, indices[0] - frame_length)
    end = min(len(audio_data), indices[-1] + frame_length)
    
    return audio_data[start:end]


def preprocess_audio(
    file_path: Union[str, Path],
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    normalize: bool = True,
    remove_silence_flag: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Preprocess audio file for model input.
    
    Args:
        file_path: Path to audio file
        target_sample_rate: Target sample rate
        normalize: Whether to normalize audio
        remove_silence_flag: Whether to remove silence
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    file_path = Path(file_path)
    
    try:
        # Read audio file
        try:
            import soundfile as sf
            audio_data, sample_rate = sf.read(str(file_path))
        except ImportError:
            sample_rate, audio_data = wav.read(str(file_path))
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample if needed
        if sample_rate != target_sample_rate:
            import librosa
            audio_data = librosa.resample(
                audio_data,
                orig_sr=sample_rate,
                target_sr=target_sample_rate
            )
            sample_rate = target_sample_rate
        
        # Normalize
        if normalize:
            audio_data = normalize_audio(audio_data)
        
        # Remove silence
        if remove_silence_flag:
            audio_data = remove_silence(audio_data, sample_rate)
        
        return audio_data, sample_rate
        
    except Exception as e:
        logger.error(f"Audio preprocessing failed: {e}")
        raise AudioProcessingError(f"Failed to preprocess audio: {e}")


def chunk_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    chunk_duration_ms: int = 250,
) -> list:
    """
    Split audio into chunks for streaming processing.
    
    Args:
        audio_data: NumPy array of audio samples
        sample_rate: Sample rate of audio
        chunk_duration_ms: Duration of each chunk in milliseconds
        
    Returns:
        List of audio chunks as numpy arrays
    """
    chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
    chunks = []
    
    for i in range(0, len(audio_data), chunk_samples):
        chunk = audio_data[i : i + chunk_samples]
        if len(chunk) > 0:
            # Pad last chunk if needed
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            chunks.append(chunk)
    
    return chunks


def audio_to_base64(audio_data: np.ndarray, sample_rate: int) -> str:
    """
    Convert numpy audio array to base64 encoded WAV string.
    
    Args:
        audio_data: NumPy array of audio samples
        sample_rate: Sample rate
        
    Returns:
        Base64 encoded WAV string
    """
    import base64
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav.write(tmp.name, sample_rate, audio_data.astype(np.int16))
        
        with open(tmp.name, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        os.unlink(tmp.name)
    
    return audio_b64


def base64_to_audio(base64_string: str) -> Tuple[np.ndarray, int]:
    """
    Convert base64 encoded audio to numpy array.
    
    Args:
        base64_string: Base64 encoded audio string
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    import base64
    import io
    
    audio_bytes = base64.b64decode(base64_string)
    
    # Read as WAV
    sample_rate, audio_data = wav.read(io.BytesIO(audio_bytes))
    
    return audio_data, sample_rate


class AudioBuffer:
    """
    Ring buffer for streaming audio processing.
    
    Maintains a rolling window of audio samples.
    """
    
    def __init__(
        self,
        sample_rate: int = TARGET_SAMPLE_RATE,
        max_duration: float = 10.0,
    ):
        """
        Initialize audio buffer.
        
        Args:
            sample_rate: Sample rate in Hz
            max_duration: Maximum buffer duration in seconds
        """
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_duration)
        self.buffer: np.ndarray = np.zeros(0)
        self._lock = False
    
    def append(self, data: np.ndarray) -> None:
        """Append new audio data to buffer."""
        if self._lock:
            return
        
        self.buffer = np.concatenate([self.buffer, data])
        
        # Trim if exceeding max size
        if len(self.buffer) > self.max_samples:
            self.buffer = self.buffer[-self.max_samples:]
    
    def get_chunk(
        self,
        start_offset: float = 0.0,
        duration: float = 2.0,
    ) -> np.ndarray:
        """
        Get a chunk of audio from the buffer.
        
        Args:
            start_offset: Start offset from end of buffer (seconds)
            duration: Duration of chunk (seconds)
            
        Returns:
            Audio chunk as numpy array
        """
        start_sample = int((len(self.buffer) - start_offset * self.sample_rate))
        end_sample = int(start_sample + duration * self.sample_rate)
        
        start_sample = max(0, start_sample)
        end_sample = min(len(self.buffer), end_sample)
        
        if start_sample >= end_sample:
            return np.zeros(0)
        
        return self.buffer[start_sample:end_sample]
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer = np.zeros(0)
    
    def get_all(self) -> np.ndarray:
        """Get all buffered audio."""
        return self.buffer.copy()
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self.buffer) == 0
    
    def length_seconds(self) -> float:
        """Get current buffer duration in seconds."""
        return len(self.buffer) / self.sample_rate

