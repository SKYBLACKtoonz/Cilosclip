"""Probing and extracting from media files with ffmpeg/ffprobe."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .util import ClipForgeError, find_binary, run


@dataclass
class MediaInfo:
    path: Path
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    rotation: int = 0

    @property
    def display_width(self) -> int:
        """Width as the viewer sees it (phones store rotated landscape frames)."""
        return self.height if self.rotation % 180 else self.width

    @property
    def display_height(self) -> int:
        return self.width if self.rotation % 180 else self.height

    @property
    def aspect(self) -> float:
        return self.display_width / max(1, self.display_height)


def ffmpeg_bin() -> str:
    exe = find_binary("ffmpeg")
    if not exe:
        raise ClipForgeError(
            "ffmpeg was not found.\n"
            "Windows:  winget install Gyan.FFmpeg   (then reopen your terminal)\n"
            "macOS:    brew install ffmpeg\n"
            "Linux:    sudo apt install ffmpeg"
        )
    return exe


def ffprobe_bin() -> str:
    exe = find_binary("ffprobe")
    if not exe:
        raise ClipForgeError("ffprobe was not found (it ships with ffmpeg).")
    return exe


def probe(path: str | Path) -> MediaInfo:
    path = Path(path)
    if not path.exists():
        raise ClipForgeError(f"Input file not found: {path}")

    proc = run(
        [
            ffprobe_bin(),
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        check=False,
    )
    if proc.returncode != 0:
        raise ClipForgeError(f"Could not read {path.name}. Is it a valid video?\n{proc.stderr[:400]}")

    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise ClipForgeError(f"{path.name} has no video stream.")

    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0)
    if duration <= 0:
        raise ClipForgeError(f"Could not determine the length of {path.name}.")

    fps = 30.0
    rate = video.get("avg_frame_rate") or video.get("r_frame_rate") or "30/1"
    try:
        num, den = (int(x) for x in str(rate).split("/"))
        if den:
            fps = num / den
    except (ValueError, ZeroDivisionError):
        pass
    if not 1 <= fps <= 240:
        fps = 30.0

    rotation = 0
    for side in video.get("side_data_list", []) or []:
        if "rotation" in side:
            try:
                rotation = int(float(side["rotation"])) % 360
            except (TypeError, ValueError):
                rotation = 0
    if not rotation:
        try:
            rotation = int(float(video.get("tags", {}).get("rotate", 0))) % 360
        except (TypeError, ValueError):
            rotation = 0

    return MediaInfo(
        path=path,
        duration=duration,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=round(fps, 3),
        has_audio=audio is not None,
        rotation=rotation,
    )


def extract_audio(video: str | Path, out_wav: str | Path, log=None) -> Path:
    """Mono 16 kHz PCM - what every speech model wants."""
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", str(video),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(out_wav),
        ],
        log=log,
    )
    return out_wav


def extract_thumbnail(video: str | Path, at: float, out_jpg: str | Path, width: int = 360) -> Path:
    out_jpg = Path(out_jpg)
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-ss", f"{max(0.0, at):.3f}",
            "-i", str(video),
            "-frames:v", "1",
            "-vf", f"scale={width}:-2",
            "-q:v", "3",
            str(out_jpg),
        ],
        check=False,
    )
    return out_jpg
