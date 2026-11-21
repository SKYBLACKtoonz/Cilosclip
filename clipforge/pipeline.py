"""The whole job, end to end: probe -> audio -> transcript -> pick -> render -> caption.

Each clip is rendered twice:
  1. a base clip (reframed, no captions) kept in <output>/_base/
  2. the final clip, with captions burned onto the base

Because the base is kept, recaption() can restyle any clip later (new animation
or font) without re-running transcription or reframing, and reframe() can
re-crop a clip by hand.
"""

from __future__ import annotations

import copy
import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from .analyze import Candidate, analyse
from .captions import burn_captions, validate_style, write_ass
from .config import Config
from .media import MediaInfo, extract_audio, extract_thumbnail, ffmpeg_bin, probe
from .render import render_clip
from .transcribe import from_file, transcribe
from .transcript import Transcript, Word, save_json, save_srt
from .util import ClipForgeError, run, slugify

Progress = Callable[[float, str], None]
Log = Callable[[str], None]
Cancel = Callable[[], bool]
ClipReady = Callable[["ClipOutput"], None]

BASE_DIR = "_base"


@dataclass
class ClipOutput:
    candidate: Candidate
    path: Path
    thumbnail: Path | None = None
    base: Path | None = None
    caption_animation: str = ""
    caption_font: str = ""

    def to_dict(self) -> dict:
        return {
            **self.candidate.to_dict(),
            "file": self.path.name,
            "thumbnail": self.thumbnail.name if self.thumbnail else "",
            "base": self.base.name if self.base else "",
            "caption_animation": self.caption_animation,
            "caption_font": self.caption_font,
        }


@dataclass
class RunResult:
    source: Path
    output_dir: Path
    media: MediaInfo
    transcript: Transcript
    clips: list[ClipOutput] = field(default_factory=list)
    manifest: Path | None = None

    @property
    def total_duration(self) -> float:
        return sum(c.candidate.duration for c in self.clips)


def _noop_progress(pct: float, msg: str) -> None:
    pass


def _noop_log(msg: str) -> None:
    pass


def _all_words(transcript: Transcript) -> list[Word]:
    """Every word in the transcript. Works whether .words is a method or a list."""
    source = transcript.words() if callable(transcript.words) else transcript.words
    return list(source)


def clip_words(transcript: Transcript, start: float, end: float) -> list[Word]:
    """Words overlapping [start, end], with times relative to the clip start."""
    out: list[Word] = []
    for w in _all_words(transcript):
        if w.end <= start or w.start >= end:
            continue
        out.append(Word(max(0.0, w.start - start), min(end, w.end) - start, w.text))
    return out


def run_pipeline(
    input_path: str | Path,
    cfg: Config,
    progress: Progress | None = None,
    log: Log | None = None,
    cancel: Cancel | None = None,
    on_clip: ClipReady | None = None,
    transcript_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> RunResult:
    progress = progress or _noop_progress
    log = log or _noop_log
    cancelled = cancel or (lambda: False)

    def checkpoint() -> None:
        if cancelled():
            raise ClipForgeError("Cancelled.")

    source = Path(input_path).expanduser()
    ffmpeg = ffmpeg_bin()
    validate_style(cfg.caption_animation, cfg.caption_font)

    # ---- 1. inspect ----------------------------------------------------- #
    progress(2, "Reading video")
    info = probe(source)
    log(f"{source.name}: {info.display_width}x{info.display_height}, "
        f"{info.duration:.1f}s, {info.fps:g} fps, audio={'yes' if info.has_audio else 'no'}")

    out_dir = Path(output_dir) if output_dir else (
        Path(cfg.output_dir).expanduser() / f"{slugify(source.stem, 'video', 40)}-{_stamp()}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    workdir = out_dir / "_work"
    workdir.mkdir(exist_ok=True)
    checkpoint()

    # ---- 2. speech ------------------------------------------------------ #
    if transcript_path:
        progress(10, "Loading your transcript")
        transcript = from_file(transcript_path)
    else:
        if not info.has_audio:
            raise ClipForgeError("This video has no audio track, so there is nothing to transcribe.")
        progress(6, "Extracting audio")
        wav = extract_audio(source, workdir / "audio.wav", log=log)
        checkpoint()
        transcript = transcribe(
            wav, cfg, progress=lambda p, m: progress(10 + p * 0.45, m)
        )
        try:
            wav.unlink()
        except OSError:
            pass

    if not transcript.segments:
        raise ClipForgeError("The transcript is empty - no speech found.")

    save_json(transcript, out_dir / "transcript.json")
    save_srt(transcript, out_dir / "transcript.srt")
    log(f"Transcript: {len(transcript.segments)} segments, {len(transcript.full_text().split())} words")
    checkpoint()

    # ---- 3. pick the moments -------------------------------------------- #
    progress(57, "Finding the best moments")
    candidates = analyse(transcript, cfg, info.duration,
                         progress=lambda p, m: progress(57 + p * 0.05, m))
    for c in candidates:
        log(f"  [{c.score:5.1f}] {c.start:7.2f}-{c.end:7.2f}  {c.title[:60]}")
    checkpoint()

    # ---- 4. render base clips, then caption ------------------------------ #
    from . import track

    if cfg.reframe == "face" and cfg.layout != "pad" and not track.available():
        log("Note: OpenCV is not installed, so clips use a centred crop.")
        log("      Install opencv-python-headless for face-aware framing.")

    base_dir = out_dir / BASE_DIR
    base_dir.mkdir(exist_ok=True)

    # Same settings, but the reframe step writes no captions.
    base_cfg = copy.copy(cfg)
    base_cfg.captions = False

    result = RunResult(source=source, output_dir=out_dir, media=info, transcript=transcript)
    total = len(candidates)
    pad = max(2, len(str(total)))

    for index, candidate in enumerate(candidates, 1):
        checkpoint()
        pct = 62 + (index - 1) / max(1, total) * 37
        progress(pct, f"Rendering clip {index}/{total}")
        name = f"{index:0{pad}d}-{slugify(candidate.title, 'clip')}.mp4"
        clip_path = out_dir / name
        base_path = base_dir / name
        log(f"Rendering {index}/{total}: {candidate.title} "
            f"({candidate.start:.1f}s - {candidate.end:.1f}s)")

        try:
            render_clip(
                info, candidate.start, candidate.end, base_cfg, transcript,
                workdir, base_path, ffmpeg, log=log, dry_run=dry_run, verbose=verbose,
            )
        except ClipForgeError as err:
            log(f"  ! skipped: {err}")
            continue

        if not dry_run:
            if cfg.captions:
                words = clip_words(transcript, candidate.start, candidate.end)
                ass = write_ass(words, cfg, workdir / f"{index:0{pad}d}.ass",
                                cfg.width, cfg.height)
                burn_captions(base_path, ass, clip_path, ffmpeg)
            else:
                shutil.copyfile(base_path, clip_path)

        thumb = None
        if not dry_run:
            thumb = out_dir / f"{index:0{pad}d}-thumb.jpg"
            extract_thumbnail(clip_path, 0.5, thumb)

        clip_out = ClipOutput(
            candidate, clip_path, thumb,
            base=None if dry_run else base_path,
            caption_animation=cfg.caption_animation if cfg.captions else "",
            caption_font=cfg.caption_font if cfg.captions else "",
        )
        result.clips.append(clip_out)
        if on_clip is not None:
            on_clip(clip_out)

    if not dry_run:
        result.manifest = _write_manifest(result, cfg)
    progress(99, "Cleaning up")
    shutil.rmtree(workdir, ignore_errors=True)
    progress(100, f"Done - {len(result.clips)} clip(s) in {out_dir}")
    return result


def recaption(
    output_dir: str | Path,
    clip_file: str,
    cfg: Config,
    *,
    animation: str,
    font: str,
    log: Log | None = None,
) -> Path:
    """Re-burn captions on an existing clip in a new style or font.

    Uses the kept base clip and the saved transcript, so no transcription or
    reframing is repeated. Updates the clip, its thumbnail and clips.json.
    """
    log = log or _noop_log
    animation, font = validate_style(animation, font)
    out_dir = Path(output_dir)
    manifest_path = out_dir / "clips.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    entry = next((c for c in manifest["clips"] if c["file"] == clip_file), None)
    if entry is None:
        raise ClipForgeError(f"{clip_file} is not in this job's clips.json")
    if not entry.get("base"):
        raise ClipForgeError("This clip has no base copy, so it cannot be recaptioned. "
                             "Re-run the job to create it.")

    base_path = out_dir / BASE_DIR / entry["base"]
    if not base_path.exists():
        raise ClipForgeError(f"Base clip is missing: {base_path.name}")

    transcript_json = out_dir / "transcript.json"
    if not transcript_json.exists():
        raise ClipForgeError("transcript.json is missing from this job folder.")
    transcript = from_file(transcript_json)

    settings = manifest["settings"]
    run_cfg = copy.copy(cfg)
    run_cfg.caption_animation = animation
    run_cfg.caption_font = font

    work = out_dir / "_work"
    work.mkdir(exist_ok=True)
    words = clip_words(transcript, float(entry["start"]), float(entry["end"]))
    ass = write_ass(words, run_cfg, work / f"{Path(clip_file).stem}.ass",
                    settings["width"], settings["height"])

    final = out_dir / clip_file
    tmp = final.with_name(final.stem + ".tmp.mp4")
    log(f"Recaptioning {clip_file}: {animation}, {font}")
    burn_captions(base_path, ass, tmp, ffmpeg_bin())
    tmp.replace(final)

    if entry.get("thumbnail"):
        extract_thumbnail(final, 0.5, out_dir / entry["thumbnail"])

    entry["caption_animation"] = animation
    entry["caption_font"] = font
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    return final


def clip_info(output_dir: str | Path, clip_file: str) -> dict:
    """What the manual reframe window needs to know about one finished clip."""
    out_dir = Path(output_dir)
    manifest_path = out_dir / "clips.json"
    if not manifest_path.exists():
        raise ClipForgeError("clips.json is missing from this job folder.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next((c for c in manifest["clips"] if c["file"] == clip_file), None)
    if entry is None:
        raise ClipForgeError(f"{clip_file} is not in this job's clips.json")
    if not entry.get("base"):
        raise ClipForgeError("This clip has no base copy, so it cannot be reframed. "
                             "Re-run the job to create it.")
    source = Path(manifest["source"])
    if not source.exists():
        raise ClipForgeError(f"The original video is missing, so the clip cannot be reframed:\n{source}")
    settings = manifest["settings"]
    return {
        "source": str(source),
        "start": float(entry["start"]),
        "end": float(entry["end"]),
        "width": int(settings["width"]),
        "height": int(settings["height"]),
        "keyframes": [(float(t), float(x)) for t, x in entry.get("crop_keyframes", [])],
    }


def grab_frame(source: str | Path, at: float, out: str | Path, width: int = 960) -> Path:
    """Save one still from the original video, for the manual reframe preview."""
    out = Path(out)
    proc = run(
        [
            ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-ss", f"{max(0.0, at):.3f}", "-i", str(source),
            "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "3", str(out),
        ],
        check=False,
    )
    if proc.returncode != 0 or not out.exists():
        raise ClipForgeError("Could not read a frame from the original video.")
    return out


def reframe(
    output_dir: str | Path,
    clip_file: str,
    cfg: Config,
    keyframes: list[tuple[float, float]],
    *,
    log: Log | None = None,
) -> Path:
    """Re-render one clip with a hand-picked crop instead of the automatic one.

    keyframes are (seconds_from_clip_start, crop_centre_x as a fraction of the
    frame width). One keyframe means a fixed crop; several mean the crop glides
    between them. The clip is rendered from the original video again, then the
    current captions are burned back on. Nothing is replaced until everything
    has rendered, so a failure leaves the old clip as it was.
    """
    log = log or _noop_log
    if not keyframes:
        raise ClipForgeError("No crop position was given.")
    out_dir = Path(output_dir)
    info_for_clip = clip_info(out_dir, clip_file)   # also validates the manifest entry

    manifest_path = out_dir / "clips.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(c for c in manifest["clips"] if c["file"] == clip_file)

    transcript_json = out_dir / "transcript.json"
    if not transcript_json.exists():
        raise ClipForgeError("transcript.json is missing from this job folder.")
    transcript = from_file(transcript_json)

    start, end = info_for_clip["start"], info_for_clip["end"]
    duration = max(0.4, end - start)
    points = sorted(
        (min(duration, max(0.0, float(t))), min(1.0, max(0.0, float(x))))
        for t, x in keyframes
    )

    ffmpeg = ffmpeg_bin()
    media = probe(Path(info_for_clip["source"]))

    render_cfg = copy.copy(cfg)
    render_cfg.captions = False          # captions go on afterwards, like a normal job
    render_cfg.layout = "crop"           # a manual crop only means something in crop layout
    render_cfg.width = info_for_clip["width"]
    render_cfg.height = info_for_clip["height"]

    work = out_dir / "_work"
    work.mkdir(exist_ok=True)
    base_path = out_dir / BASE_DIR / entry["base"]
    base_tmp = base_path.with_name(base_path.stem + ".tmp.mp4")
    final = out_dir / clip_file
    final_tmp = final.with_name(final.stem + ".tmp.mp4")

    try:
        log(f"Reframing {clip_file} ({len(points)} crop point(s))")
        render_clip(
            media, start, end, render_cfg, transcript, work, base_tmp, ffmpeg,
            log=log, crop_keyframes=points,
        )

        animation = entry.get("caption_animation") or ""
        if animation:
            ass_cfg = copy.copy(cfg)
            ass_cfg.caption_animation = animation
            ass_cfg.caption_font = entry.get("caption_font") or cfg.caption_font
            words = clip_words(transcript, start, end)
            ass = write_ass(words, ass_cfg, work / f"{Path(clip_file).stem}.ass",
                            info_for_clip["width"], info_for_clip["height"])
            burn_captions(base_tmp, ass, final_tmp, ffmpeg)
        else:
            shutil.copyfile(base_tmp, final_tmp)

        base_tmp.replace(base_path)
        final_tmp.replace(final)
    finally:
        for leftover in (base_tmp, final_tmp):
            try:
                leftover.unlink()
            except OSError:
                pass
        shutil.rmtree(work, ignore_errors=True)

    if entry.get("thumbnail"):
        extract_thumbnail(final, 0.5, out_dir / entry["thumbnail"])

    entry["reframe"] = "manual"
    entry["crop_keyframes"] = [[round(t, 3), round(x, 4)] for t, x in points]
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    return final


def _write_manifest(result: RunResult, cfg: Config) -> Path:
    manifest = result.output_dir / "clips.json"
    payload = {
        "source": str(result.source.resolve()),
        "created": datetime.now().isoformat(timespec="seconds"),
        "video": {
            "duration": round(result.media.duration, 3),
            "width": result.media.display_width,
            "height": result.media.display_height,
            "fps": result.media.fps,
        },
        "settings": {
            "clip_count": cfg.clip_count,
            "min_duration": cfg.min_duration,
            "max_duration": cfg.max_duration,
            "layout": cfg.layout,
            "reframe": cfg.reframe,
            "captions": cfg.captions,
            "caption_animation": cfg.caption_animation,
            "caption_font": cfg.caption_font,
            "width": cfg.width,
            "height": cfg.height,
        },
        "transcript": "transcript.srt",
        "transcript_json": "transcript.json",
        "clips": [c.to_dict() for c in result.clips],
    }
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")
