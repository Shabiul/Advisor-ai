"""
Emotions Engine — Multimodal Real-Time Emotion Intelligence
============================================================
Acoustic emotion detection (Wav2Vec2) + Speech understanding (Canary-Qwen).
Importable as a package for integration with the Advisor-AI pipeline.
"""

from .emo import (
    EmotionSmoother,
    SharedState,
    analyse_text_emotion,
    fuse_emotions,
    preprocess_audio,
    compute_rms,
    sharpen_scores,
    parse_dims,
    vad_quadrant,
    load_emotion_models,
    check_gpu,
    ACOUSTIC_WEIGHT,
    TEXT_WEIGHT,
    SAMPLE_RATE,
    NOISE_GATE_RMS,
    EMA_ALPHA,
    TEMPERATURE,
    DIMENSIONAL_MODEL,
    CATEGORICAL_MODEL,
)

__all__ = [
    "EmotionSmoother",
    "SharedState",
    "analyse_text_emotion",
    "fuse_emotions",
    "preprocess_audio",
    "compute_rms",
    "sharpen_scores",
    "parse_dims",
    "vad_quadrant",
    "load_emotion_models",
    "check_gpu",
    "ACOUSTIC_WEIGHT",
    "TEXT_WEIGHT",
    "SAMPLE_RATE",
    "NOISE_GATE_RMS",
    "EMA_ALPHA",
    "TEMPERATURE",
    "DIMENSIONAL_MODEL",
    "CATEGORICAL_MODEL",
]
