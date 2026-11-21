"""Command line interface: ``clipforge run video.mp4``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .analyze import analyse, has_llm
from .config import CONFIG_FILE, Config, load_config
from .media import ffmpeg_bin, probe
from .pipeline import run_pipeline
from .transcribe import api_credentials, from_file, local_available, transcribe
from .util import ClipForgeError, fmt_ts, show_error_dialog

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clipforge",
        description="Turn long videos into captioned vertical clips, locally.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  clipforge run podcast.mp4
  clipforge run talk.mp4 -n 8 --layout crop --reframe face
  clipforge run talk.mp4 --transcript subtitles.srt --no-captions
  clipforge run talk.mp4 --llm-base https://api.openai.com/v1 --llm-model gpt-4o-mini
  clipforge analyze talk.mp4          # just show the picks, render nothing
  clipforge check                     # what is installed / configured
""",
    )
    parser.add_argument("--version", action="version", version=f"clipforge {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("input", type=Path, help="source video file")
        p.add_argument("-o", "--out", type=Path, help="output folder")
        p.add_argument("-n", "--count", type=int, help="how many clips (default 5)")
        p.add_argument("--min-duration", type=float, metavar="SEC", help="shortest clip (default 15)")
        p.add_argument("--max-duration", type=float, metavar="SEC", help="longest clip (default 60)")
        p.add_argument("--layout", choices=["blur", "crop", "pad"], help="how to fill the 9:16 frame")
        p.add_argument("--reframe", choices=["face", "center"], help="crop strategy (default face)")
        p.add_argument("--no-captions", action="store_true", help="do not burn in subtitles")
        p.add_argument("--font", help="caption font name (default Arial)")
        p.add_argument("--transcript", type=Path, help="reuse an existing .srt/.vtt/.json transcript")
        p.add_argument("--stt", choices=["auto", "local", "api"], help="speech-to-text backend")
        p.add_argument("--whisper-model", help="local model size: tiny/base/small/medium/large-v3")
        p.add_argument("--language", help="spoken language code, e.g. en (default auto)")
        p.add_argument("--llm-base", help="OpenAI-compatible base URL for highlight picking")
        p.add_argument("--llm-model", help="model name for highlight picking")
        p.add_argument("--llm-key", help="API key for the LLM endpoint")
        p.add_argument("--no-llm", action="store_true", help="use the built-in scorer, no network")
        p.add_argument("-v", "--verbose", action="store_true", help="show ffmpeg and ffprobe output")

    run_p = sub.add_parser("run", help="make clips from a video")
    add_common(run_p)
    run_p.add_argument("--dry-run", action="store_true", help="plan the work, render nothing")
    run_p.add_argument("--json", action="store_true", help="print a JSON summary at the end")

    an_p = sub.add_parser("analyze", help="transcribe and show the chosen moments, no rendering")
    add_common(an_p)
    an_p.add_argument("--json", action="store_true", help="print the picks as JSON")
    an_p.add_argument("--save", type=Path, help="write the picks to a JSON file")

    sub.add_parser("check", help="report on ffmpeg, models and settings")

    cfg_p = sub.add_parser("config", help="show or change saved settings")
    cfg_p.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="update settings, e.g. llm_model=gpt-4o-mini")
    cfg_p.add_argument("--path", action="store_true", help="print the config file location only")

    gui_p = sub.add_parser("gui", help="open the desktop app")
    gui_p.add_argument("input", nargs="?", type=Path, help="video to load on start")

    return parser


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    cfg = cfg.copy()
    mapping = {
        "count": "clip_count",
        "min_duration": "min_duration",
        "max_duration": "max_duration",
        "layout": "layout",
        "reframe": "reframe",
        "font": "font",
        "stt": "transcribe_backend",
        "whisper_model": "whisper_model",
        "language": "whisper_language",
        "llm_base": "llm_base_url",
        "llm_model": "llm_model",
        "llm_key": "llm_api_key",
    }
    for attr, target in mapping.items():
        value = getattr(args, attr, None)
        if value is not None:
            setattr(cfg, target, value)
    if getattr(args, "no_captions", False):
        cfg.captions = False
    if getattr(args, "no_llm", False):
        cfg.llm_base_url = ""
    if cfg.clip_count < 1:
        raise ClipForgeError("--count must be at least 1.")
    if cfg.min_duration <= 0 or cfg.max_duration < cfg.min_duration:
        raise ClipForgeError("--max-duration must be greater than --min-duration.")
    return cfg


def _progress_printer(quiet: bool = False):
    state = {"last": ""}

    def progress(pct: float, message: str) -> None:
        if quiet:
            return
        line = f"[{pct:5.1f}%] {message}"
        if sys.stdout.isatty():
            width = max(len(state["last"]), len(line))
            sys.stdout.write("\r" + line.ljust(width))
            sys.stdout.flush()
            state["last"] = line
        elif message != state["last"]:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
        state["last"] = message

    return progress


def _log_printer(verbose: bool):
    def log(message: str) -> None:
        if sys.stdout.isatty():
            sys.stdout.write("\r" + " " * 100 + "\r")
        print(("  " if not message.startswith("$ ") else "") + message, flush=True)

    return log


def _load_source(args, cfg: Config):
    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        raise ClipForgeError(f"Video not found: {input_path}")
    return input_path


def _ensure_speech_ready(cfg: Config, args) -> None:
    if getattr(args, "transcript", None):
        return
    if cfg.transcribe_backend == "auto" and not local_available() and not api_credentials()[0]:
        raise ClipForgeError(
            "No speech-to-text backend available.\n\n"
            "Choose one:\n"
            "  pip install faster-whisper              (local, free, recommended)\n"
            "  set OPENAI_API_KEY=...                  (cloud)\n"
            "  clipforge run video.mp4 --transcript subs.srt"
        )


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_run(args, cfg: Config) -> int:
    input_path = _load_source(args, cfg)
    _ensure_speech_ready(cfg, args)
    ffmpeg_bin()  # fail fast with a friendly message

    progress = _progress_printer(quiet=args.json)
    log = _log_printer(args.verbose)

    try:
        result = run_pipeline(
            input_path,
            cfg,
            progress=progress,
            log=log,
            transcript_path=args.transcript,
            output_dir=args.out,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130

    if sys.stdout.isatty():
        print()

    if args.json:
        payload = {
            "output_dir": str(result.output_dir),
            "clips": [
                {**c.to_dict(), "path": str(c.path)} for c in result.clips
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        if args.dry_run:
            print(f"\nWould write {len(result.clips)} clip(s) to {result.output_dir}")
        else:
            print(f"\n{len(result.clips)} clip(s) written to {result.output_dir}")
        for clip in result.clips:
            c = clip.candidate
            print(
                f"  {clip.path.name:<50} {fmt_ts(c.start)}-{fmt_ts(c.end)} "
                f"({c.duration:.0f}s, score {c.score:.0f})"
            )
        if result.manifest:
            print(f"\nManifest: {result.manifest}")

    return 0 if result.clips or args.dry_run else 1


def cmd_analyze(args, cfg: Config) -> int:
    input_path = _load_source(args, cfg)
    _ensure_speech_ready(cfg, args)
    ffmpeg_bin()

    progress = _progress_printer()
    info = probe(input_path)

    if args.transcript:
        transcript = from_file(args.transcript)
    else:
        from .media import extract_audio
        import tempfile

        with tempfile.TemporaryDirectory(prefix="clipforge-") as tmp:
            wav = extract_audio(input_path, Path(tmp) / "audio.wav")
            transcript = transcribe(wav, cfg, progress=lambda p, m: progress(10 + p * 0.6, m))

    if sys.stdout.isatty():
        print()
    progress(80, "Choosing moments")
    clips = analyse(transcript, cfg, info.duration)
    progress(100, "Done")

    payload = {
        "source": str(input_path),
        "duration": round(info.duration, 2),
        "engine": "llm" if has_llm(cfg) else "heuristic",
        "clips": [c.to_dict() for c in clips],
    }

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"\n{len(clips)} candidate clip(s) from {fmt_ts(info.duration)} of video:\n")
        for i, c in enumerate(clips, 1):
            print(f"  {i}. [{c.score:5.1f}] {fmt_ts(c.start)} - {fmt_ts(c.end)}  ({c.duration:.0f}s)")
            print(f"     {c.title}")
            if c.hook:
                print(f"     hook: \"{c.hook}\"")
    return 0


def cmd_check(args, cfg: Config) -> int:
    from . import track

    def mark(ok: bool) -> str:
        return "ok  " if ok else "--  "

    print(f"ClipForge {__version__}")
    print(f"Python      {sys.version.split()[0]}  ({sys.executable})")
    print(f"Config      {CONFIG_FILE}")

    print()
    ffmpeg = None
    try:
        ffmpeg = ffmpeg_bin()
        from .util import run as run_cmd

        version = run_cmd([ffmpeg, "-version"], check=False).stdout.splitlines()
        print(f"{mark(True)}ffmpeg    {version[0] if version else ffmpeg}")

        filters = run_cmd([ffmpeg, "-hide_banner", "-filters"], check=False).stdout
        has_ass = " subtitles " in filters
        print(f"{mark(has_ass)}libass    {'captions available' if has_ass else 'MISSING - ffmpeg build has no subtitles filter'}")
    except ClipForgeError as err:
        print(f"{mark(False)}ffmpeg    {err}")

    print(f"{mark(local_available())}whisper   "
          f"{'faster-whisper installed' if local_available() else 'not installed (pip install faster-whisper)'}")
    print(f"{mark(track.available())}opencv    "
          f"{'face tracking available' if track.available() else 'not installed (pip install opencv-python-headless)'}")

    try:
        import PySide6  # noqa: F401

        print(f"{mark(True)}gui       PySide6 installed")
    except Exception:
        print(f"{mark(False)}gui       PySide6 not installed (pip install PySide6)")

    print()
    if has_llm(cfg):
        key = "set" if cfg.llm_api_key else "none"
        print(f"Highlight picker: LLM  {cfg.llm_model} at {cfg.llm_base_url} (key: {key})")
    else:
        base, key = api_credentials()
        print("Highlight picker: built-in heuristic scorer (no network)")
        if base:
            print(f"  (speech-to-text API at {base}, key {'set' if key else 'none'})")
    print(f"Speech-to-text   : {cfg.transcribe_backend}"
          + (f", whisper {cfg.whisper_model}" if cfg.transcribe_backend != "api" else ""))
    print(f"Output folder    : {cfg.output_dir}")
    return 0


def cmd_config(args, cfg: Config) -> int:
    if args.path:
        print(CONFIG_FILE)
        return 0

    cfg = cfg.copy()
    if args.set:
        from dataclasses import fields

        known = {f.name: f for f in fields(Config)}
        for pair in args.set:
            if "=" not in pair:
                raise ClipForgeError(f"Expected KEY=VALUE, got '{pair}'")
            key, _, raw = pair.partition("=")
            key = key.strip()
            if key not in known:
                raise ClipForgeError(f"Unknown setting '{key}'. Try: {', '.join(sorted(known))}")
            default = known[key].default
            if isinstance(default, bool):
                value: object = raw.strip().lower() in {"1", "true", "yes", "on"}
            elif isinstance(default, int):
                value = int(raw)
            elif isinstance(default, float):
                value = float(raw)
            else:
                value = raw.strip()
            setattr(cfg, key, value)
        cfg.save()
        print(f"Saved {CONFIG_FILE}")

    data = {k: v for k, v in cfg.__dict__.items()}
    if data.get("llm_api_key"):
        data["llm_api_key"] = "***" + str(data["llm_api_key"])[-4:]
    print(json.dumps(data, indent=2))
    return 0


def cmd_gui(args, cfg: Config) -> int:
    try:
        from .gui import launch
    except ImportError:
        message = (
            "The desktop app needs PySide6.\n\n"
            "Install it with:\n  pip install PySide6\n\n"
            "Or run setup.bat again, which installs everything."
        )
        show_error_dialog("ClipForge", message)
        raise ClipForgeError(message) from None
    return launch(args.input, cfg)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        cfg = load_config()
        if args.command in {"run", "analyze"}:
            cfg = apply_overrides(cfg, args)
        if args.command == "run":
            return cmd_run(args, cfg)
        if args.command == "analyze":
            return cmd_analyze(args, cfg)
        if args.command == "check":
            return cmd_check(args, cfg)
        if args.command == "config":
            return cmd_config(args, cfg)
        if args.command == "gui":
            return cmd_gui(args, cfg)
        parser.error("unknown command")
    except ClipForgeError as err:
        print(f"\nError: {err}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
