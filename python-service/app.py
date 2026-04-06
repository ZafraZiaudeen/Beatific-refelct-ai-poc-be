from __future__ import annotations

import base64
import inspect
import io
import logging
import os
import wave
from functools import lru_cache
from typing import Dict, List

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    import huggingface_hub
except ModuleNotFoundError:
    huggingface_hub = None  # type: ignore[assignment]

try:
    from speechbrain.inference.speaker import EncoderClassifier
    SPEECHBRAIN_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as error:
    EncoderClassifier = None  # type: ignore[assignment]
    SPEECHBRAIN_IMPORT_ERROR = error

TARGET_SAMPLE_RATE = 16000
FALLBACK_EMBEDDING_DIM = 45
MIN_ACTIVE_SPEECH_SECONDS = 0.35
SHORT_UTTERANCE_SECONDS = 1.6
SPEECHBRAIN_MIN_CONFIDENCE = 0.62
SPEECHBRAIN_MIN_MARGIN = 0.08
FALLBACK_MIN_CONFIDENCE = 0.42
FALLBACK_MIN_MARGIN = 0.015
FALLBACK_SHORT_MIN_CONFIDENCE = 0.5
FALLBACK_SHORT_MIN_MARGIN = 0.025

logger = logging.getLogger("reflect-ai-poc-speaker-id")
logging.basicConfig(level=logging.INFO)


def apply_huggingface_compat_patch() -> bool:
    if huggingface_hub is None:
        return False

    parameters = inspect.signature(huggingface_hub.hf_hub_download).parameters
    if "use_auth_token" in parameters:
        return False

    original_download = huggingface_hub.hf_hub_download

    def compat_hf_hub_download(*args, use_auth_token=None, token=None, **kwargs):
        effective_token = token if token is not None else use_auth_token
        return original_download(*args, token=effective_token, **kwargs)

    huggingface_hub.hf_hub_download = compat_hf_hub_download  # type: ignore[assignment]
    logger.info(
        "Applied Hugging Face compatibility shim so SpeechBrain can call hf_hub_download(use_auth_token=...)."
    )
    return True


HUGGINGFACE_COMPAT_PATCHED = apply_huggingface_compat_patch()


class EmbedRequest(BaseModel):
    audioBase64: str


class ProfileInput(BaseModel):
    role: str
    displayName: str | None = None
    embedding: List[float]


class IdentifyRequest(BaseModel):
    audioBase64: str
    profiles: List[ProfileInput]


app = FastAPI(title="Reflect AI POC Speaker ID Service")


@lru_cache(maxsize=1)
def get_classifier_bundle() -> tuple["EncoderClassifier | None", str | None]:
    if EncoderClassifier is None:
        message = (
            "SpeechBrain is unavailable in this environment. "
            f"Original import error: {SPEECHBRAIN_IMPORT_ERROR}"
        )
        return None, message

    model_name = os.getenv("SPEAKER_MODEL", "speechbrain/spkrec-ecapa-voxceleb")
    savedir = os.getenv("SPEAKER_MODEL_CACHE", "pretrained_models/spkrec-ecapa-voxceleb")

    try:
        classifier = EncoderClassifier.from_hparams(source=model_name, savedir=savedir)
        return classifier, None
    except Exception as error:  # pragma: no cover - runtime dependency failure path
        message = (
            "SpeechBrain model initialization failed. "
            f"Falling back to local acoustic embeddings. Cause: {error}"
        )
        logger.warning(message)
        return None, str(error)


def decode_wav_base64(audio_base64: str) -> torch.Tensor:
    raw = base64.b64decode(audio_base64)

    with wave.open(io.BytesIO(raw), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        pcm = wav_file.readframes(frame_count)

    if sample_width != 2:
        raise HTTPException(status_code=400, detail="Expected 16-bit PCM WAV input.")

    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    waveform = torch.from_numpy(audio).unsqueeze(0)

    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, TARGET_SAMPLE_RATE)

    return waveform


def trim_to_active_speech(waveform: torch.Tensor) -> torch.Tensor:
    mono = waveform.squeeze(0)
    frame_length = 400
    hop_length = 160

    if mono.numel() < frame_length:
        return waveform

    frames = mono.unfold(0, frame_length, hop_length)
    frame_rms = frames.pow(2).mean(dim=1).sqrt()

    if frame_rms.numel() == 0:
        return waveform

    max_rms = float(frame_rms.max().item())
    if max_rms <= 1e-4:
        return waveform

    median_rms = float(frame_rms.median().item())
    threshold = max(max_rms * 0.18, median_rms * 2.0, 0.01)
    active_frames = (frame_rms >= threshold).nonzero(as_tuple=False).squeeze(1)

    if active_frames.numel() == 0:
        return waveform

    start_frame = max(int(active_frames[0].item()) - 2, 0)
    end_frame = min(int(active_frames[-1].item()) + 3, frame_rms.numel())
    start_sample = start_frame * hop_length
    end_sample = min(mono.numel(), end_frame * hop_length + frame_length)
    trimmed = mono[start_sample:end_sample]

    if trimmed.numel() < int(MIN_ACTIVE_SPEECH_SECONDS * TARGET_SAMPLE_RATE):
        return waveform

    return trimmed.unsqueeze(0)


def extract_fallback_embedding(waveform: torch.Tensor) -> torch.Tensor:
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=TARGET_SAMPLE_RATE,
        n_mfcc=20,
        melkwargs={
            "n_fft": 400,
            "hop_length": 160,
            "n_mels": 40,
        },
    )
    mfcc = mfcc_transform(waveform)
    mfcc_mean = mfcc.mean(dim=2).squeeze(0)
    mfcc_std = mfcc.std(dim=2, unbiased=False).squeeze(0)

    mono = waveform.squeeze(0)
    abs_wave = mono.abs()
    zero_crossings = (
        ((mono[:-1] * mono[1:]) < 0).float().mean().unsqueeze(0)
        if mono.numel() > 1
        else torch.zeros(1)
    )
    energy = mono.pow(2).mean().sqrt().unsqueeze(0)
    abs_mean = abs_wave.mean().unsqueeze(0)
    abs_std = abs_wave.std(unbiased=False).unsqueeze(0)
    peak = abs_wave.max().unsqueeze(0)

    embedding = torch.cat([mfcc_mean, mfcc_std, energy, abs_mean, abs_std, peak, zero_crossings])
    return torch.nn.functional.normalize(embedding, dim=0)


def extract_embedding_from_waveform(
    waveform: torch.Tensor,
    force_backend: str | None = None,
) -> tuple[torch.Tensor, str]:
    prepared_waveform = trim_to_active_speech(waveform)

    if force_backend == "fallback":
        return extract_fallback_embedding(prepared_waveform).cpu(), "fallback"

    classifier, classifier_error = get_classifier_bundle()

    if classifier is None:
        logger.warning(
            "Using fallback acoustic embedding backend. "
            f"SpeechBrain classifier unavailable: {classifier_error}"
        )
        return extract_fallback_embedding(prepared_waveform).cpu(), "fallback"

    try:
        with torch.no_grad():
            embedding = classifier.encode_batch(prepared_waveform)

        return embedding.squeeze().cpu(), "speechbrain"
    except Exception as error:  # pragma: no cover - runtime inference failure path
        logger.warning(
            "SpeechBrain embedding failed during inference; using fallback acoustic embedding instead. "
            f"Cause: {error}"
        )
        return extract_fallback_embedding(prepared_waveform).cpu(), "fallback"


def extract_embedding(audio_base64: str) -> tuple[torch.Tensor, str, float]:
    waveform = decode_wav_base64(audio_base64)
    duration_seconds = waveform.shape[1] / TARGET_SAMPLE_RATE if waveform.shape[1] else 0.0
    embedding, backend = extract_embedding_from_waveform(waveform)
    return embedding, backend, duration_seconds


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0)).item())


def choose_backend_for_profiles(profiles: List[ProfileInput]) -> str | None:
    dimensions = [len(profile.embedding) for profile in profiles if profile.embedding]
    if not dimensions:
        return None

    if all(dimension == FALLBACK_EMBEDDING_DIM for dimension in dimensions):
        return "fallback"

    return None


def resolve_thresholds(backend: str, duration_seconds: float) -> tuple[float, float]:
    if backend == "fallback":
        if duration_seconds < SHORT_UTTERANCE_SECONDS:
            return FALLBACK_SHORT_MIN_CONFIDENCE, FALLBACK_SHORT_MIN_MARGIN

        return FALLBACK_MIN_CONFIDENCE, FALLBACK_MIN_MARGIN

    return SPEECHBRAIN_MIN_CONFIDENCE, SPEECHBRAIN_MIN_MARGIN


@app.get("/health")
def health() -> Dict[str, object]:
    classifier, classifier_error = get_classifier_bundle()

    return {
        "ok": True,
        "backend": "speechbrain" if classifier is not None else "fallback",
        "speechbrain_import_ok": EncoderClassifier is not None,
        "speechbrain_import_error": str(SPEECHBRAIN_IMPORT_ERROR) if SPEECHBRAIN_IMPORT_ERROR else None,
        "speechbrain_runtime_error": classifier_error,
        "huggingface_compat_patch_applied": HUGGINGFACE_COMPAT_PATCHED,
    }


@app.post("/embed")
def embed(request: EmbedRequest) -> Dict[str, List[float]]:
    embedding, _, _ = extract_embedding(request.audioBase64)
    return {"embedding": embedding.tolist()}


@app.post("/identify")
def identify(request: IdentifyRequest) -> Dict[str, object]:
    if not request.profiles:
        return {"role": "unknown", "confidence": 0.0, "scores": {}}

    profile_dimensions = {len(profile.embedding) for profile in request.profiles if profile.embedding}
    if len(profile_dimensions) > 1:
        logger.warning(
            "Speaker identification skipped because profile embeddings use mixed dimensions %s. "
            "Reset the local browser data and re-enroll both partners together.",
            sorted(profile_dimensions),
        )
        return {"role": "unknown", "confidence": 0.0, "scores": {}}

    waveform = decode_wav_base64(request.audioBase64)
    duration_seconds = waveform.shape[1] / TARGET_SAMPLE_RATE if waveform.shape[1] else 0.0
    preferred_backend = choose_backend_for_profiles(request.profiles)
    utterance, backend = extract_embedding_from_waveform(waveform, force_backend=preferred_backend)
    compatible_profiles = [
        profile for profile in request.profiles if len(profile.embedding) == int(utterance.shape[0])
    ]

    if not compatible_profiles and backend != "fallback":
        utterance, backend = extract_embedding_from_waveform(waveform, force_backend="fallback")
        compatible_profiles = [
            profile for profile in request.profiles if len(profile.embedding) == int(utterance.shape[0])
        ]

    if not compatible_profiles:
        logger.warning(
            "Speaker identification skipped because stored profile embeddings are incompatible with the active backend. "
            "Re-enroll both partners so their embeddings are regenerated together."
        )
        return {"role": "unknown", "confidence": 0.0, "scores": {}}

    scores: Dict[str, float] = {}

    for profile in compatible_profiles:
        profile_embedding = torch.tensor(profile.embedding, dtype=torch.float32)
        scores[profile.role] = cosine_similarity(utterance, profile_embedding)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_role, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else -1.0
    margin = best_score - second_score
    min_confidence, min_margin = resolve_thresholds(backend, duration_seconds)

    logger.info(
        "Speaker identification backend=%s duration=%.2fs scores=%s best=%s confidence=%.4f margin=%.4f thresholds=(%.3f, %.3f)",
        backend,
        duration_seconds,
        {key: round(value, 4) for key, value in scores.items()},
        best_role,
        best_score,
        margin,
        min_confidence,
        min_margin,
    )

    if best_score < min_confidence or (len(ranked) > 1 and margin < min_margin):
        return {
            "role": "unknown",
            "confidence": round(max(best_score, 0.0), 4),
            "scores": {key: round(value, 4) for key, value in scores.items()},
            "backend": backend,
        }

    return {
        "role": best_role,
        "confidence": round(best_score, 4),
        "scores": {key: round(value, 4) for key, value in scores.items()},
        "backend": backend,
    }
