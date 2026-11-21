"""Turning one candidate window into a finished vertical clip with ffmpeg."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .captions import write_ass
from .config import Config
from .media import MediaInfo
from .track import available as faces_available, track_face_x
from .transcript import Transcript, Word
from .util import ClipForgeError, run


@dataclass
class RenderResult:
    path: Path
    command: list[str]


def _layout_for(info: MediaInfo, cfg: Config) -> str:
    """Blur-padding an already-vertical video just blurs nothing; fill it instead."""
    if cfg.layout == "blur" and abs(info.aspect - cfg.width / cfg.height) < 0.12:
        return "crop"
    return cfg.layout if cfg.layout in {"blur", "crop", "pad"} else "blur"


def crop_fraction(frame_w: int, frame_h: int, out_w: int, out_h: int) -> float:
    """Width of the vertical crop as a fraction of the source frame's width."""
    if frame_w <= 0 or frame_h <= 0:
        return 1.0
    factor = max(out_w / frame_w, out_h / frame_h)
    return min(1.0, out_w / (frame_w * factor))


def keyframe_value(points: list[tuple[float, float]], t: float) -> float:
    """Where the crop sits at time t: straight lines between keyframes, held at both ends.

    Mirrors the expression _piecewise() builds for ffmpeg.
    """
    pts = sorted(points)
    if not pts:
        return 0.5
    if t <= pts[0][0]:
        return pts[0][1]
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if t <= t1:
            return v0 if t1 - t0 <= 1e-6 else v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return pts[-1][1]


def _piecewise(t_value: str, points: list[tuple[float, float]], clamp_max: float) -> str:
    """Build an ffmpeg expression interpolating linearly between keyframes."""
    points = sorted(points)
    last_v = points[-1][1]
    expr = f"{last_v:.1f}"
    for i in range(len(points) - 2, -1, -1):
        t0, v0 = points[i]
        t1, v1 = points[i + 1]
        if t1 - t0 <= 1e-6:
            continue
        slope = (v1 - v0) / (t1 - t0)
        segment = f"{v0:.1f}+({slope:.4f})*({t_value}-{t0:.2f})"
        expr = f"if(lt({t_value},{t1:.2f}),{segment},{expr})"
    # Never show past the first keyframe (no extrapolating backwards).
    if len(points) > 1:
        t_first, v_first = points[0]
        expr = f"if(lt({t_value},{t_first:.2f}),{v_first:.1f},{expr})"
    return f"max(0,min({clamp_max:.1f},{expr}))"


def build_filtergraph(
    info: MediaInfo,
    cfg: Config,
    layout: str,
    caption_file: str | None,
    crop_keyframes: list[tuple[float, float]] | None,
) -> str:
    w, h = cfg.width, cfg.height
    parts: list[str] = []
    common = f"fps={cfg.fps},setpts=PTS-STARTPTS"

    if layout == "blur":
        sigma = max(12, round(w * 0.028))
        parts.append(
            f"[0:v]{common},split=2[bgsrc][fgsrc];"
            f"[bgsrc]scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h},gblur=sigma={sigma},eq=brightness=-0.10:saturation=0.75[bg];"
            f"[fgsrc]scale={w}:-2:flags=lanczos[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[v0]"
        )
    elif layout == "pad":
        parts.append(
            f"[0:v]{common},scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1[v0]"
        )
    else:  # crop: fill the frame, optionally following a face
        crop = f"crop={w}:{h}"
        if crop_keyframes:
            # The frame is scaled to *cover* w x h; work out the scaled width so
            # the face-centre fractions can be turned into pixel offsets.
            factor = max(w / max(1, info.display_width), h / max(1, info.display_height))
            scaled_w = info.display_width * factor
            max_x = max(0.0, scaled_w - w)
            points = [
                (t, min(max_x, max(0.0, frac * scaled_w - w / 2)))
                for t, frac in crop_keyframes
            ]
            crop = f"crop={w}:{h}:x='{_piecewise('t', points, max_x)}':y=0"
        parts.append(
            f"[0:v]{common},scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"{crop},setsar=1[v0]"
        )

    if caption_file:
        parts.append(f"[v0]subtitles={caption_file}[v]")
    else:
        parts.append("[v0]null[v]")

    return ";".join(parts)


def render_clip(
    info: MediaInfo,
    start: float,
    end: float,
    cfg: Config,
    transcript: Transcript,
    workdir: str | Path,
    out_path: str | Path,
    ffmpeg: str,
    log=None,
    dry_run: bool = False,
    verbose: bool = False,
    crop_keyframes: list[tuple[float, float]] | None = None,
) -> RenderResult:
    """Render one clip.

    crop_keyframes: optional hand-picked crop path, (seconds_from_clip_start,
    crop_centre_x as a fraction of frame width). When given, face tracking is
    skipped. Only used with the "crop" layout.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    duration = max(0.4, end - start)
    layout = _layout_for(info, cfg)

    # ---- captions ------------------------------------------------------- #
    caption_name: str | None = None
    if cfg.captions:
        words = _clip_words(transcript, start, end)
        if words:
            write_ass(words, cfg, workdir / "captions.ass", cfg.width, cfg.height)
            caption_name = "captions.ass"   # relative: avoids Windows path escaping

    # ---- auto reframe --------------------------------------------------- #
    keyframes = crop_keyframes
    if keyframes is None and layout == "crop" and cfg.reframe == "face" and faces_available():
        if log:
            log(f"Tracking faces in clip {duration:.0f}s ...")
        keyframes = track_face_x(info.path, start, end, ffmpeg, workdir, log=log)

    graph = build_filtergraph(info, cfg, layout, caption_name, keyframes)

    cmd: list[str] = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-ss", f"{start:.3f}",
        "-i", str(info.path.resolve()),
        "-t", f"{duration:.3f}",
        "-filter_complex", graph,
        "-map", "[v]",
    ]
    if info.has_audio:
        cmd += [
            "-map", "0:a:0?",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        ]
    else:
        cmd += ["-an"]

    cmd += [
        "-c:v", "libx264",
        "-preset", cfg.preset,
        "-crf", str(cfg.crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-map_metadata", "-1",
        str(out_path.resolve()),
    ]

    if dry_run:
        if log:
            log(" ".join(cmd))
        return RenderResult(out_path, cmd)

    proc = run(cmd, cwd=workdir, log=log if verbose else None, check=False)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-10:])
        raise ClipForgeError(f"ffmpeg could not render this clip:\n{tail}")
    if not out_path.exists() or out_path.stat().st_size < 1024:
        raise ClipForgeError("ffmpeg reported success but produced no usable video.")

    return RenderResult(out_path, cmd)


def _clip_words(transcript: Transcript, start: float, end: float) -> list[Word]:
    words: list[Word] = []
    for word in transcript.slice(start, end).words():
        s = max(0.0, word.start - start)
        e = max(s + 0.02, min(end - start, word.end - start))
        words.append(Word(s, e, word.text))
    return words
