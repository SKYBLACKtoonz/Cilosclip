"""Transcript data model plus SRT / VTT / JSON readers and writers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .util import ClipForgeError


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Transcript:
    segments: list[Segment]
    language: str = ""
    source: str = ""

    # -- derived views ---------------------------------------------------- #
    def words(self) -> list[Word]:
        """Every word, synthesised from segment timings when the model gave none."""
        out: list[Word] = []
        for seg in self.segments:
            if seg.words:
                out.extend(seg.words)
                continue
            out.extend(_synthesise_words(seg))
        return out

    def full_text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    def timed_text(self, start: float = 0.0, end: float | None = None) -> str:
        """Transcript formatted as `[m:ss] text` lines, for LLM consumption."""
        end = float("inf") if end is None else end
        lines = []
        for seg in self.segments:
            if seg.end < start or seg.start > end:
                continue
            lines.append(f"[{_mmss(seg.start)}] {seg.text.strip()}")
        return "\n".join(lines)

    def slice(self, start: float, end: float) -> "Transcript":
        """Segments overlapping a window, clipped to it."""
        out = []
        for seg in self.segments:
            if seg.end <= start or seg.start >= end:
                continue
            words = [
                Word(max(w.start, start), min(w.end, end), w.text)
                for w in seg.words
                if w.end > start and w.start < end
            ]
            out.append(
                Segment(
                    start=max(seg.start, start),
                    end=min(seg.end, end),
                    text=seg.text.strip(),
                    words=words,
                )
            )
        return Transcript(out, self.language, self.source)

    @property
    def duration(self) -> float:
        return self.segments[-1].end if self.segments else 0.0


def _mmss(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


_TERMINAL = re.compile(r"[.!?…]([\"')\]]*)$")


def _synthesise_words(seg: Segment) -> list[Word]:
    """Spread a segment's text across its duration, weighted by word length.

    Whisper often returns sentence-level timings only. Captions still need
    per-word timing to do the karaoke highlight, and length-weighting is a
    surprisingly good approximation.
    """
    tokens = seg.text.split()
    if not tokens:
        return []
    weights = [max(1, len(tok.strip(".,!?;:"))) for tok in tokens]
    total = sum(weights)
    span = max(0.05, seg.end - seg.start)
    words: list[Word] = []
    cursor = seg.start
    for tok, weight in zip(tokens, weights):
        dur = span * weight / total
        words.append(Word(cursor, min(seg.end, cursor + dur), tok))
        cursor += dur
    return words


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_transcript(path: str | Path) -> Transcript:
    path = Path(path)
    if not path.exists():
        raise ClipForgeError(f"Transcript file not found: {path}")
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".srt":
        return Transcript(parse_srt(text), source=str(path))
    if suffix == ".vtt":
        return Transcript(parse_vtt(text), source=str(path))
    if suffix == ".json":
        return _load_json(text, path)
    raise ClipForgeError(f"Unsupported transcript format: {suffix}. Use .srt, .vtt or .json")


_SRT_TIME = re.compile(
    r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{1,3})"
)


def _time(parts) -> float:
    h, m, s, ms = parts
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def parse_srt(text: str) -> list[Segment]:
    segments: list[Segment] = []
    blocks = re.split(r"\r?\n\s*\r?\n", text.strip())
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        stamp_idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if stamp_idx is None:
            continue
        match = _SRT_TIME.search(lines[stamp_idx])
        if not match:
            continue
        start = _time(match.groups()[:4])
        end = _time(match.groups()[4:])
        body = " ".join(lines[stamp_idx + 1 :]).strip()
        body = re.sub(r"</?[a-zA-Z][^>]*>", "", body)
        if body:
            segments.append(Segment(start, end, body))
    if not segments:
        raise ClipForgeError("No subtitle cues could be read from that file.")
    return segments


def parse_vtt(text: str) -> list[Segment]:
    # VTT shares SRT's cue shape once headers/NOTE blocks are stripped.
    cleaned = re.sub(r"^WEBVTT.*?(\r?\n\r?\n)", "", text, count=1, flags=re.S)
    cleaned = re.sub(r"(?m)^(NOTE|STYLE|REGION).*?(?=\r?\n\r?\n)", "", cleaned, flags=re.S)
    return parse_srt(cleaned)


def _load_json(text: str, path: Path) -> Transcript:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as err:
        raise ClipForgeError(f"{path.name} is not valid JSON: {err}") from err

    # Our own transcript.json format.
    if isinstance(data, dict) and "segments" in data:
        segments = [
            Segment(
                float(s["start"]),
                float(s["end"]),
                str(s.get("text", "")).strip(),
                [
                    Word(float(w["start"]), float(w["end"]), str(w["text"]))
                    for w in s.get("words", []) or []
                ],
            )
            for s in data["segments"]
        ]
        return Transcript(segments, data.get("language", ""), str(path))

    # Whisper.cpp / generic word list.
    if isinstance(data, list):
        words = [Word(float(w["start"]), float(w["end"]), str(w["text"])) for w in data]
        return _transcript_from_words(words, str(path))

    raise ClipForgeError(f"{path.name} is not a transcript ClipForge understands.")


def _transcript_from_words(words: list[Word], source: str = "") -> Transcript:
    """Group a flat word list into sentence-ish segments."""
    segments: list[Segment] = []
    current: list[Word] = []
    for word in words:
        current.append(word)
        text = " ".join(w.text for w in current)
        long_pause = len(current) > 1 and word.start - current[-2].end > 0.6
        if _TERMINAL.search(word.text) or len(text) > 180 or long_pause:
            segments.append(
                Segment(current[0].start, current[-1].end, text.strip(), current[:])
            )
            current = []
    if current:
        segments.append(Segment(current[0].start, current[-1].end, " ".join(w.text for w in current), current[:]))
    return Transcript(segments, "", source)


def save_json(transcript: Transcript, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "language": transcript.language,
        "source": transcript.source,
        "segments": [
            {
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "text": s.text,
                "words": [{"start": round(w.start, 3), "end": round(w.end, 3), "text": w.text} for w in s.words],
            }
            for s in transcript.segments
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_srt(transcript: Transcript, path: str | Path) -> Path:
    path = Path(path)

    def stamp(t: float) -> str:
        ms = int(round(t * 1000))
        h, ms = divmod(ms, 3_600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    for i, seg in enumerate(transcript.segments, 1):
        lines += [str(i), f"{stamp(seg.start)} --> {stamp(seg.end)}", seg.text.strip(), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
