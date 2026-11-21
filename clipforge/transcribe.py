"""Speech-to-text backends: local faster-whisper, OpenAI-compatible API, or a file."""

from __future__ import annotations

# Make the pip-installed CUDA DLLs (cublas, cudnn) findable before faster_whisper
# or ctranslate2 are imported. Must stay above any faster_whisper/ctranslate2 import.
import glob
import os
import sys
from pathlib import Path
from typing import Callable

_nvidia = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
_dll_dirs = glob.glob(os.path.join(_nvidia, "*", "bin"))
for _bin in _dll_dirs:
    os.add_dll_directory(_bin)
os.environ["PATH"] = os.pathsep.join(_dll_dirs) + os.pathsep + os.environ.get("PATH", "")

from .config import Config
from .transcript import Segment, Transcript, Word, _transcript_from_words, load_transcript
from .util import ClipForgeError, http_multipart

Progress = Callable[[float, str], None]

LOCAL_SIZES = {
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large-v1", "large-v2", "large-v3",
    "distil-small.en", "distil-medium.en", "distil-large-v3",
}


def local_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except Exception:
        return False
    return True


def api_credentials() -> tuple[str, str]:
    """Return (base_url, api_key) for an OpenAI-compatible transcribe endpoint."""
    base = os.environ.get("CLIPFORGE_STT_BASE") or os.environ.get("OPENAI_BASE_URL") or ""
    key = os.environ.get("CLIPFORGE_STT_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    if not base and key:
        base = "https://api.openai.com/v1"
    return base.rstrip("/"), key


def resolve_backend(cfg: Config) -> str:
    backend = (cfg.transcribe_backend or "auto").lower()
    if backend != "auto":
        return backend
    if local_available():
        return "local"
    if api_credentials()[0]:
        return "api"
    raise ClipForgeError(
        "No speech-to-text backend available.\n\n"
        "Pick one of these:\n"
        "  1. Local & free:   pip install faster-whisper\n"
        "  2. Cloud API:      set OPENAI_API_KEY (works with OpenAI, Groq, ...)\n"
        "  3. Bring your own: clipforge run video.mp4 --transcript subs.srt"
    )


def transcribe(
    audio_wav: str | Path,
    cfg: Config,
    progress: Progress | None = None,
) -> Transcript:
    backend = resolve_backend(cfg)
    if backend == "local":
        return transcribe_local(audio_wav, cfg, progress)
    if backend == "api":
        return transcribe_api(audio_wav, cfg, progress)
    if backend == "file":
        raise ClipForgeError("Backend 'file' needs an explicit --transcript path.")
    raise ClipForgeError(f"Unknown transcription backend: {backend}")


# --------------------------------------------------------------------------- #
# Local (faster-whisper)
# --------------------------------------------------------------------------- #
def _pick_device(cfg: Config) -> tuple[str, str]:
    device = (cfg.whisper_device or "auto").lower()
    compute = (cfg.whisper_compute_type or "auto").lower()
    if device == "auto":
        try:
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"
    return device, compute


def transcribe_local(audio_wav: str | Path, cfg: Config, progress: Progress | None = None) -> Transcript:
    try:
        from faster_whisper import WhisperModel
    except Exception as err:  # pragma: no cover - depends on user install
        raise ClipForgeError(
            "faster-whisper is not installed. Run:  pip install faster-whisper"
        ) from err

    size = cfg.whisper_model or "small"
    if size not in LOCAL_SIZES and not Path(size).exists():
        raise ClipForgeError(
            f"Unknown whisper model '{size}'. Choose from: {', '.join(sorted(LOCAL_SIZES))}"
        )

    device, compute = _pick_device(cfg)
    if progress:
        progress(2, f"Loading Whisper '{size}' on {device} ({compute})")

    def build(dev: str, comp: str):
        return WhisperModel(size, device=dev, compute_type=comp)

    try:
        model = build(device, compute)
    except Exception:
        # A CUDA build without a usable GPU is common; CPU is slower but works.
        if device != "cpu":
            if progress:
                progress(4, "GPU unavailable, falling back to CPU")
            device, compute = "cpu", "int8"
            model = build(device, compute)
        else:
            raise

    try:
        segments_iter, info = model.transcribe(
            str(audio_wav),
            language=cfg.whisper_language or None,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
            word_timestamps=True,
        )
    except Exception as err:
        raise ClipForgeError(f"Whisper failed to transcribe the audio: {err}") from err

    total = float(getattr(info, "duration", 0) or 0)
    collected: list[Segment] = []
    for seg in segments_iter:
        words = [
            Word(float(w.start), float(w.end), str(w.word).strip())
            for w in (getattr(seg, "words", None) or [])
            if str(w.word).strip()
        ]
        collected.append(Segment(float(seg.start), float(seg.end), str(seg.text).strip(), words))
        if progress and total:
            progress(min(99.0, 100.0 * float(seg.end) / total), f"Transcribing {int(100 * seg.end / total)}%")

    if not collected:
        raise ClipForgeError("No speech was detected in that video.")

    return Transcript(collected, getattr(info, "language", "") or "", "faster-whisper:" + size)


# --------------------------------------------------------------------------- #
# OpenAI-compatible API
# --------------------------------------------------------------------------- #
def _api_model(cfg: Config, base: str) -> str:
    override = os.environ.get("CLIPFORGE_STT_MODEL")
    if override:
        return override
    model = (cfg.whisper_model or "").strip()
    if model and model not in LOCAL_SIZES:
        return model
    if "groq" in base:
        return "whisper-large-v3"
    return "whisper-1"


def transcribe_api(audio_wav: str | Path, cfg: Config, progress: Progress | None = None) -> Transcript:
    base, key = api_credentials()
    if not base:
        raise ClipForgeError(
            "No transcription API configured. Set OPENAI_API_KEY or CLIPFORGE_STT_BASE."
        )

    url = base + ("/audio/transcriptions" if not base.endswith("/audio/transcriptions") else "")
    model = _api_model(cfg, base)
    data = Path(audio_wav).read_bytes()
    if progress:
        progress(5, f"Uploading audio to {model} ({len(data) / 1e6:.1f} MB)")

    fields = {"model": model, "response_format": "verbose_json", "temperature": "0"}
    if cfg.whisper_language:
        fields["language"] = cfg.whisper_language

    reply = http_multipart(
        url,
        fields,
        [("file", "audio.wav", data, "audio/wav")],
        headers={"Authorization": f"Bearer {key}"} if key else None,
        timeout=1800,
    )
    if progress:
        progress(70, "Transcription received")

    raw_segments = reply.get("segments") or []
    segments = [
        Segment(
            float(s.get("start", 0)),
            float(s.get("end", 0)),
            str(s.get("text", "")).strip(),
            [
                Word(float(w["start"]), float(w["end"]), str(w["word"]).strip())
                for w in (s.get("words") or [])
            ],
        )
        for s in raw_segments
        if float(s.get("end", 0)) > float(s.get("start", 0))
    ]

    if not segments and reply.get("text"):
        # Some providers only honour verbose_json partially.
        return Transcript([Segment(0.0, 0.0, str(reply["text"]).strip())], reply.get("language", ""), "api:" + model)

    if not segments:
        raise ClipForgeError("The transcription API returned no segments.")
    return Transcript(segments, reply.get("language", "") or "", "api:" + model)


def from_file(path: str | Path) -> Transcript:
    transcript = load_transcript(path)
    # A flat word list still needs grouping.
    if transcript.segments and not any(s.text for s in transcript.segments):
        words = [w for s in transcript.segments for w in s.words]
        return _transcript_from_words(words, str(path))
    return transcript