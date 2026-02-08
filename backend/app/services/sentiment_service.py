"""
Sentiment Analysis Service using Transformers.
Provides emotion detection using pre-trained transformer models.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Dict, Optional, List, Any

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from ..config import config
from ..models.schemas import SentimentLabel, SentimentResult

logger = logging.getLogger(__name__)


class SentimentService:
    """
    Service for sentiment/emotion analysis using Transformers.
    
    Uses j-hartmann/emotion-english-distilroberta-base which detects:
    - joy, sadness, anger, fear, love, surprise
    """

    _instance: Optional["SentimentService"] = None
    _lock: threading.Lock = threading.Lock()

    # Mapping from model labels to our enum
    EMOTION_MAPPING = {
        "joy": SentimentLabel.JOY,
        "sadness": SentimentLabel.SADNESS,
        "anger": SentimentLabel.ANGER,
        "fear": SentimentLabel.FEAR,
        "love": SentimentLabel.LOVE,
        "surprise": SentimentLabel.SURPRISE,
    }

    def __new__(cls) -> "SentimentService":
        """Singleton pattern to avoid reloading model."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the sentiment service."""
        if self._initialized:
            return

        self.model_name: str = config.SENTIMENT_MODEL_NAME
        self.device: str = config.SENTIMENT_DEVICE
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.classifier: Optional[Any] = None
        self._load_time: Optional[float] = None
        self._model_loaded: bool = False
        self._labels: List[str] = []

        self._initialized = True
        logger.info(f"SentimentService initialized with model: {self.model_name}")

    def load_model(self, force_reload: bool = False) -> None:
        """
        Load the sentiment analysis model.
        
        Args:
            force_reload: If True, reload even if already loaded
        """
        if self._model_loaded and not force_reload:
            logger.info("Sentiment model already loaded, skipping reload")
            return

        start_time = time.time()

        try:
            # Determine device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(
                f"Loading sentiment model '{self.model_name}' on {self.device}"
            )

            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self.model.to(self.device)
            self.model.eval()

            # Get labels from model config
            self._labels = self.model.config.id2label.values()  # type: ignore

            # Create pipeline for easier inference
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device if self.device != "cpu" else -1,
                top_k=None,  # Return all scores
            )

            self._load_time = time.time() - start_time
            self._model_loaded = True

            logger.info(
                f"Sentiment model loaded successfully in {self._load_time:.2f}s"
            )

        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            raise RuntimeError(f"Sentiment model loading failed: {e}")

    def analyze(
        self,
        text: str,
        return_all_scores: bool = True,
    ) -> SentimentResult:
        """
        Analyze sentiment/emotion in text.
        
        Args:
            text: Input text to analyze
            return_all_scores: If True, return all emotion scores
            
        Returns:
            SentimentResult with dominant emotion and score
        """
        if not self._model_loaded:
            self.load_model()

        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        logger.debug(f"Analyzing sentiment for text: {text[:50]}...")

        try:
            # Run classification
            results = self.classifier(text)

            # Handle different return formats
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], list):
                    # Multiple inputs, take first
                    results = results[0]

                # Extract scores
                if isinstance(results[0], dict) and "score" in results[0]:
                    # Format: [{"label": "joy", "score": 0.9}, ...]
                    scores = {
                        r["label"]: r["score"]
                        for r in sorted(results, key=lambda x: x["score"], reverse=True)
                    }
                else:
                    # Already processed format
                    scores = results if isinstance(results, dict) else {}
            else:
                scores = {}

            # Find dominant emotion
            if scores:
                dominant_label = max(scores, key=scores.get)  # type: ignore
                dominant_score = scores[dominant_label]  # type: ignore
            else:
                dominant_label = "neutral"
                dominant_score = 1.0

            # Map to our enum
            mapped_label = self.EMOTION_MAPPING.get(
                dominant_label, SentimentLabel.NEUTRAL
            )

            return SentimentResult(
                label=mapped_label,
                score=dominant_score,
            )

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            raise RuntimeError(f"Sentiment analysis error: {e}")

    def analyze_with_all_scores(
        self,
        text: str,
    ) -> Dict[SentimentLabel, float]:
        """
        Analyze text and return scores for all emotions.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary mapping emotion labels to confidence scores
        """
        if not self._model_loaded:
            self.load_model()

        if not text or not text.strip():
            return {label: 0.0 for label in SentimentLabel}

        try:
            results = self.classifier(text)

            # Handle multiple inputs
            if isinstance(results, list):
                results = results[0]

            # Extract all scores
            all_scores: Dict[SentimentLabel, float] = {}
            for result in results:
                label = result.get("label", "unknown")
                score = result.get("score", 0.0)
                mapped = self.EMOTION_MAPPING.get(label, SentimentLabel.NEUTRAL)
                all_scores[mapped] = score

            # Ensure all labels present
            for label in SentimentLabel:
                if label not in all_scores:
                    all_scores[label] = 0.0

            return all_scores

        except Exception as e:
            logger.error(f"Full sentiment analysis failed: {e}")
            raise RuntimeError(f"Full sentiment analysis error: {e}")

    def analyze_complete(
        self,
        text: str,
    ) -> tuple[SentimentResult, Dict[SentimentLabel, float]]:
        """
        Complete sentiment analysis with all scores.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (dominant_result, all_scores_dict)
        """
        dominant = self.analyze(text)
        all_scores = self.analyze_with_all_scores(text)
        return dominant, all_scores

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self._model_loaded:
            return {"status": "not_loaded"}

        info = {
            "model_name": self.model_name,
            "device": self.device,
            "is_loaded": self._model_loaded,
            "load_time_seconds": self._load_time,
            "labels": list(self._labels),
        }

        # Add memory info
        if torch.cuda.is_available():
            info["gpu_memory_mb"] = torch.cuda.get_device_properties(
                0
            ).total_memory / (1024**2)

        return info

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            del self.classifier
            self.model = None
            self.tokenizer = None
            self.classifier = None
            self._model_loaded = False

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Sentiment model unloaded")


# Global instance
sentiment_service = SentimentService()

