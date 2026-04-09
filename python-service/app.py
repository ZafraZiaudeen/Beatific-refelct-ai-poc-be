from __future__ import annotations

import base64
import inspect
import io
import logging
import os
import wave
from contextlib import asynccontextmanager
from typing import Dict, List

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── resemblyzer (primary backend) ─────────────────────────────────────────────
# The GE2E pretrained model is bundled inside the resemblyzer wheel (15.7 MB)
# so it is available immediately after `pip install resemblyzer --no-deps`.
# We intentionally skip `preprocess_wav` (which needs webrtcvad) and do our
# own audio normalisation instead.
try:
    from resemblyzer import VoiceEncoder as _ResemblyzerEncoder
    RESEMBLYZER_IMPORT_ERROR: Exception | None = None
except Exception as _err:
    _ResemblyzerEncoder = None          # type: ignore[assignment,misc]
    RESEMBLYZER_IMPORT_ERROR = _err

# ── SpeechBrain (optional second-tier, kept for backwards compat) ──────────────
try:
    import huggingface_hub as _hf_hub
except ModuleNotFoundError:
    _hf_hub = None  # type: ignore[assignment]

try:
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
    SPEECHBRAIN_IMPORT_ERROR: Exception | None = None
except Exception as _sberr:
    EncoderClassifier = None            # type: ignore[assignment]
    LocalStrategy = None                # type: ignore[assignment]
    SPEECHBRAIN_IMPORT_ERROR = _sberr

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET_SAMPLE_RATE = 16_000

# resemblyzer GE2E — 256-dim embeddings, high accuracy
RESEMBLYZER_EMBEDDING_DIM  = 256
RESEMBLYZER_MIN_CONFIDENCE = 0.62   # same speaker typically 0.85-0.97; lowered for conversational speech
RESEMBLYZER_MIN_MARGIN     = 0.06

# MFCC fallback — 45-dim, low accuracy (last resort only)
FALLBACK_EMBEDDING_DIM          = 45
FALLBACK_MIN_CONFIDENCE         = 0.58
FALLBACK_MIN_MARGIN             = 0.09
FALLBACK_SHORT_MIN_CONFIDENCE   = 0.64
FALLBACK_SHORT_MIN_MARGIN       = 0.12

SHORT_UTTERANCE_SECONDS    = 1.6
MIN_ACTIVE_SPEECH_SECONDS  = 0.35

# Set FORCE_MFCC_ONLY=1 to skip resemblyzer (offline / no-download mode).
# Default is OFF so resemblyzer is always tried first.
FORCE_MFCC_ONLY: bool = os.getenv("FORCE_MFCC_ONLY", "0").strip().lower() not in {
    "0", "false", "no", "off"
}

logger = logging.getLogger("reflect-ai-poc-speaker-id")
logging.basicConfig(level=logging.INFO)

if os.name == "nt":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ── HuggingFace compat shim (needed only if SpeechBrain path is used) ──────────
def _apply_hf_compat_patch() -> bool:
    if _hf_hub is None:
        return False
    params = inspect.signature(_hf_hub.hf_hub_download).parameters
    if "use_auth_token" in params:
        return False
    _orig = _hf_hub.hf_hub_download

    def _compat(*args, use_auth_token=None, token=None, **kwargs):
        return _orig(*args, token=token if token is not None else use_auth_token, **kwargs)

    _hf_hub.hf_hub_download = _compat  # type: ignore[assignment]
    return True

_HF_COMPAT_PATCHED = _apply_hf_compat_patch()

# ── resemblyzer encoder (singleton, loaded at startup) ─────────────────────────
_resemblyzer_encoder: "_ResemblyzerEncoder | None" = None
_resemblyzer_error: str | None = None

def _load_resemblyzer() -> tuple["_ResemblyzerEncoder | None", str | None]:
    global _resemblyzer_encoder, _resemblyzer_error

    if _resemblyzer_encoder is not None:
        return _resemblyzer_encoder, None
    if _resemblyzer_error is not None:
        return None, _resemblyzer_error

    if FORCE_MFCC_ONLY:
        _resemblyzer_error = "FORCE_MFCC_ONLY is enabled"
        return None, _resemblyzer_error

    if _ResemblyzerEncoder is None:
        msg = f"resemblyzer not installed: {RESEMBLYZER_IMPORT_ERROR}"
        _resemblyzer_error = msg
        logger.warning(msg)
        return None, msg

    try:
        logger.info("Loading resemblyzer GE2E speaker encoder (downloads ~14 MB on first run)…")
        _resemblyzer_encoder = _ResemblyzerEncoder()
        logger.info("resemblyzer encoder ready — speaker identification active.")
        return _resemblyzer_encoder, None
    except Exception as exc:
        _resemblyzer_error = str(exc)
        logger.warning("resemblyzer failed to load: %s", exc)
        return None, _resemblyzer_error


# ── SpeechBrain helpers (only used when resemblyzer unavailable) ────────────────
def _resolve_sb_model_name() -> str:
    return os.getenv("SPEAKER_MODEL", "speechbrain/spkrec-ecapa-voxceleb")

def _resolve_sb_cache_dir() -> str:
    return os.path.abspath(
        os.getenv("SPEAKER_MODEL_CACHE", "pretrained_models/spkrec-ecapa-voxceleb")
    )

def _resolve_sb_local_strategy():
    if LocalStrategy is None:
        return None
    raw = os.getenv("SPEAKER_MODEL_LOCAL_STRATEGY", "").strip().upper()
    strategy = getattr(LocalStrategy, raw, None) if raw else None
    if raw and strategy is None:
        logger.warning("Unknown SPEAKER_MODEL_LOCAL_STRATEGY=%s — using platform default.", raw)
    if strategy is not None:
        return strategy
    return LocalStrategy.COPY if os.name == "nt" else LocalStrategy.SYMLINK

_SB_LOCAL_STRATEGY = _resolve_sb_local_strategy()
_sb_classifier = None
_sb_error: str | None = None

def _load_speechbrain():
    global _sb_classifier, _sb_error
    if _sb_classifier is not None:
        return _sb_classifier, None
    if _sb_error is not None:
        return None, _sb_error
    if EncoderClassifier is None:
        _sb_error = f"SpeechBrain import failed: {SPEECHBRAIN_IMPORT_ERROR}"
        return None, _sb_error
    try:
        savedir = _resolve_sb_cache_dir()
        os.makedirs(savedir, exist_ok=True)
        _sb_classifier = EncoderClassifier.from_hparams(
            source=_resolve_sb_model_name(),
            savedir=savedir,
            local_strategy=_SB_LOCAL_STRATEGY,
        )
        return _sb_classifier, None
    except Exception as exc:
        _sb_error = str(exc)
        logger.warning("SpeechBrain model failed: %s", exc)
        return None, _sb_error


# ── FastAPI models ─────────────────────────────────────────────────────────────
class EmbedRequest(BaseModel):
    audioBase64: str

class ProfileInput(BaseModel):
    role: str
    displayName: str | None = None
    embedding: List[float]

class IdentifyRequest(BaseModel):
    audioBase64: str
    profiles: List[ProfileInput]


# ── Audio helpers ──────────────────────────────────────────────────────────────
def _decode_wav(audio_base64: str) -> torch.Tensor:
    raw = base64.b64decode(audio_base64)
    with wave.open(io.BytesIO(raw), "rb") as wf:
        channels    = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate  = wf.getframerate()
        frames       = wf.getnframes()
        pcm          = wf.readframes(frames)

    if sample_width != 2:
        raise HTTPException(status_code=400, detail="Expected 16-bit PCM WAV.")

    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    waveform = torch.from_numpy(audio).unsqueeze(0)
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, TARGET_SAMPLE_RATE)
    return waveform


def _trim_active_speech(waveform: torch.Tensor) -> torch.Tensor:
    mono = waveform.squeeze(0)
    frame_length, hop_length = 400, 160
    if mono.numel() < frame_length:
        return waveform
    frames     = mono.unfold(0, frame_length, hop_length)
    frame_rms  = frames.pow(2).mean(dim=1).sqrt()
    if frame_rms.numel() == 0:
        return waveform
    max_rms = float(frame_rms.max().item())
    if max_rms <= 1e-4:
        return waveform
    median_rms = float(frame_rms.median().item())
    threshold  = max(max_rms * 0.18, median_rms * 2.0, 0.01)
    active     = (frame_rms >= threshold).nonzero(as_tuple=False).squeeze(1)
    if active.numel() == 0:
        return waveform
    start_sample = max(int(active[0].item()) - 2, 0) * hop_length
    end_sample   = min(mono.numel(), (min(int(active[-1].item()) + 3, frame_rms.numel())) * hop_length + frame_length)
    trimmed = mono[start_sample:end_sample]
    if trimmed.numel() < int(MIN_ACTIVE_SPEECH_SECONDS * TARGET_SAMPLE_RATE):
        return waveform
    return trimmed.unsqueeze(0)


def _mfcc_embedding(waveform: torch.Tensor) -> torch.Tensor:
    """45-dim MFCC fallback — only used when resemblyzer is unavailable."""
    mfcc_t = torchaudio.transforms.MFCC(
        sample_rate=TARGET_SAMPLE_RATE,
        n_mfcc=20,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40},
    )
    mfcc = mfcc_t(waveform)
    mean = mfcc.mean(dim=2).squeeze(0)
    std  = mfcc.std(dim=2, unbiased=False).squeeze(0)
    mono = waveform.squeeze(0)
    zcr  = ((mono[:-1] * mono[1:]) < 0).float().mean().unsqueeze(0) if mono.numel() > 1 else torch.zeros(1)
    energy   = mono.pow(2).mean().sqrt().unsqueeze(0)
    abs_mean = mono.abs().mean().unsqueeze(0)
    abs_std  = mono.abs().std(unbiased=False).unsqueeze(0)
    peak     = mono.abs().max().unsqueeze(0)
    vec = torch.cat([mean, std, energy, abs_mean, abs_std, peak, zcr])
    return torch.nn.functional.normalize(vec, dim=0)


def _resemblyzer_embedding(waveform: torch.Tensor) -> torch.Tensor:
    """256-dim GE2E embedding — accurate speaker identity.

    We skip resemblyzer's preprocess_wav (needs webrtcvad) and normalise
    the audio ourselves — the VoiceEncoder accepts any float32 array at 16 kHz.
    """
    encoder, err = _load_resemblyzer()
    if encoder is None:
        raise RuntimeError(f"resemblyzer unavailable: {err}")
    audio_np = waveform.squeeze(0).numpy().astype(np.float32)
    # Normalise to [-1, 1] so the model receives the right dynamic range
    peak = np.abs(audio_np).max()
    if peak > 1e-6:
        audio_np = audio_np / peak
    emb = encoder.embed_utterance(audio_np)
    return torch.tensor(emb, dtype=torch.float32)


def _extract_embedding(
    waveform: torch.Tensor,
    force_backend: str | None = None,
) -> tuple[torch.Tensor, str]:
    prepared = _trim_active_speech(waveform)

    # Explicit fallback override
    if force_backend == "fallback":
        return _mfcc_embedding(prepared).cpu(), "fallback"

    # ── 1. resemblyzer (best) ──────────────────────────────────────────────────
    if force_backend in (None, "resemblyzer") and not FORCE_MFCC_ONLY:
        try:
            emb = _resemblyzer_embedding(prepared)
            return emb.cpu(), "resemblyzer"
        except Exception as exc:
            logger.warning("resemblyzer embedding failed, trying SpeechBrain: %s", exc)

    # ── 2. SpeechBrain ECAPA (second tier) ────────────────────────────────────
    if not FORCE_MFCC_ONLY:
        classifier, _ = _load_speechbrain()
        if classifier is not None:
            try:
                with torch.no_grad():
                    emb = classifier.encode_batch(prepared)
                return emb.squeeze().cpu(), "speechbrain"
            except Exception as exc:
                logger.warning("SpeechBrain inference failed: %s", exc)

    # ── 3. MFCC fallback (last resort) ────────────────────────────────────────
    logger.info("Using MFCC fallback embedding (accuracy limited).")
    return _mfcc_embedding(prepared).cpu(), "fallback"


def _choose_backend(profiles: List[ProfileInput]) -> str | None:
    dims = [len(p.embedding) for p in profiles if p.embedding]
    if not dims:
        return None
    if all(d == FALLBACK_EMBEDDING_DIM for d in dims):
        return "fallback"
    if all(d == RESEMBLYZER_EMBEDDING_DIM for d in dims):
        return "resemblyzer"
    return None  # let the system decide


def _thresholds(backend: str, duration_s: float) -> tuple[float, float]:
    if backend == "resemblyzer":
        # GE2E model is robust — same margin for short and normal
        return RESEMBLYZER_MIN_CONFIDENCE, RESEMBLYZER_MIN_MARGIN
    if backend == "fallback":
        if duration_s < SHORT_UTTERANCE_SECONDS:
            return FALLBACK_SHORT_MIN_CONFIDENCE, FALLBACK_SHORT_MIN_MARGIN
        return FALLBACK_MIN_CONFIDENCE, FALLBACK_MIN_MARGIN
    # speechbrain ECAPA
    return 0.62, 0.08


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


# ── App lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the speaker model before accepting any requests."""
    if not FORCE_MFCC_ONLY:
        encoder, err = _load_resemblyzer()
        if encoder is not None:
            logger.info("Speaker identification ready (resemblyzer GE2E).")
        else:
            logger.warning(
                "resemblyzer not ready (%s). Trying SpeechBrain fallback…", err
            )
            _load_speechbrain()
    else:
        logger.info("FORCE_MFCC_ONLY=1 — using MFCC spectrogram embeddings.")
    yield


app = FastAPI(title="Reflect AI Speaker ID", lifespan=lifespan)


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> Dict[str, object]:
    enc, enc_err   = _load_resemblyzer()
    cls, cls_err   = _load_speechbrain()

    active = "resemblyzer" if enc else ("speechbrain" if cls else "fallback")
    return {
        "ok": True,
        "active_backend": active,
        "resemblyzer_ready": enc is not None,
        "resemblyzer_error": enc_err,
        "speechbrain_ready": cls is not None,
        "speechbrain_error": cls_err,
        "force_mfcc_only": FORCE_MFCC_ONLY,
    }


@app.post("/embed")
def embed(request: EmbedRequest) -> Dict[str, List[float]]:
    waveform = _decode_wav(request.audioBase64)
    emb, backend = _extract_embedding(waveform)
    logger.info("Enrolled voice sample — backend=%s dim=%d", backend, len(emb))
    return {"embedding": emb.tolist()}


@app.post("/identify")
def identify(request: IdentifyRequest) -> Dict[str, object]:
    if not request.profiles:
        return {"role": "unknown", "confidence": 0.0, "scores": {}}

    # Reject mixed-dimension profiles (enrolled with different backends)
    dims = {len(p.embedding) for p in request.profiles if p.embedding}
    if len(dims) > 1:
        logger.warning(
            "Mixed embedding dimensions %s — re-enroll both partners together.", sorted(dims)
        )
        return {"role": "unknown", "confidence": 0.0, "scores": {}}

    waveform   = _decode_wav(request.audioBase64)
    duration_s = waveform.shape[1] / TARGET_SAMPLE_RATE if waveform.shape[1] else 0.0

    preferred = _choose_backend(request.profiles)
    utterance, backend = _extract_embedding(waveform, force_backend=preferred)

    # Filter to profiles whose embedding dim matches what we extracted
    compatible = [p for p in request.profiles if len(p.embedding) == int(utterance.shape[0])]

    # If no match and we used a neural backend, fall back to MFCC
    if not compatible and backend != "fallback":
        utterance, backend = _extract_embedding(waveform, force_backend="fallback")
        compatible = [p for p in request.profiles if len(p.embedding) == int(utterance.shape[0])]

    if not compatible:
        logger.warning(
            "Stored embeddings (dim=%s) incompatible with extracted embedding (dim=%d). "
            "Re-enroll both partners.",
            sorted(dims), int(utterance.shape[0]),
        )
        return {"role": "unknown", "confidence": 0.0, "scores": {}}

    # Cosine similarity against each enrolled profile
    scores: Dict[str, float] = {
        p.role: _cosine(utterance, torch.tensor(p.embedding, dtype=torch.float32))
        for p in compatible
    }

    ranked         = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_role, best_score = ranked[0]
    second_score   = ranked[1][1] if len(ranked) > 1 else -1.0
    margin         = best_score - second_score
    min_conf, min_margin = _thresholds(backend, duration_s)

    logger.info(
        "identify backend=%s dur=%.2fs scores=%s best=%s conf=%.4f margin=%.4f thresholds=(%.3f,%.3f)",
        backend, duration_s,
        {k: round(v, 4) for k, v in scores.items()},
        best_role, best_score, margin, min_conf, min_margin,
    )

    if best_score < min_conf or (len(ranked) > 1 and margin < min_margin):
        return {
            "role": "unknown",
            "confidence": round(max(best_score, 0.0), 4),
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "backend": backend,
        }

    return {
        "role": best_role,
        "confidence": round(best_score, 4),
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "backend": backend,
    }
