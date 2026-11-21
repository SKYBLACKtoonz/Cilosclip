"""Burned-in captions: ASS subtitle generation plus a burn-in step.

Caption styles (cfg.caption_animation):
  - "word_pop":   the spoken word pops and takes the highlight colour (default)
  - "typewriter": the line types itself out letter by letter
  - "karaoke":    the line is filled with the highlight colour word by word

Fonts (cfg.caption_font):
  - "montserrat"  Montserrat, bold   (default)
  - "bebas"       Bebas Neue, condensed display
  - "bangers"     Bangers, comic display

The fonts must be installed on the machine, because libass uses the system font list.

burn_captions() takes an already-rendered clip and writes a new file with the
subtitles on it. Keeping the caption pass separate from the reframe pass means
a clip can be recaptioned in a new style without re-running the whole render.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .transcript import Word
from .util import ClipForgeError, ass_time

MAX_LINE_SECONDS = 2.6
PAUSE_BREAK = 0.9

ANIMATIONS = ("word_pop", "typewriter", "karaoke")

# font key -> (installed font family name, ASS bold flag)
FONTS: dict[str, tuple[str, int]] = {
    "montserrat": ("Montserrat", -1),
    "bebas": ("Bebas Neue", 0),
    "bangers": ("Bangers", 0),
}


@dataclass
class CaptionLine:
    words: list[Word]

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end


def validate_style(animation: str, font: str) -> tuple[str, str]:
    """Return normalised (animation, font) or raise ValueError."""
    anim = (animation or "word_pop").lower()
    key = (font or "montserrat").lower()
    if anim not in ANIMATIONS:
        raise ValueError(f"Unknown caption animation {anim!r}; expected one of {ANIMATIONS}")
    if key not in FONTS:
        raise ValueError(f"Unknown caption font {key!r}; expected one of {tuple(FONTS)}")
    return anim, key


def _hex_to_ass(rgb: str, alpha: str = "00") -> str:
    """'RRGGBB' -> ASS &HAABBGGRR& (BGR channel order, as ASS expects)."""
    rgb = (rgb or "FFFFFF").strip().lstrip("#")
    if len(rgb) != 6:
        rgb = "FFFFFF"
    rr, gg, bb = rgb[0:2], rgb[2:4], rgb[4:6]
    return f"&H{alpha}{bb}{gg}{rr}&".upper()


def _clean(text: str) -> str:
    """Remove characters ASS or the caption font cannot render."""
    out = []
    for ch in text:
        code = ord(ch)
        if code > 0xFFFF:              # emoji / astral plane
            continue
        if 0x1F000 <= code <= 0x1FAFF:
            continue
        if 0x2600 <= code <= 0x27BF:   # misc symbols, dingbats
            continue
        if ch in "{}":
            continue
        if ch == "\\":
            out.append("/")
            continue
        out.append(ch)
    return " ".join("".join(out).split())


def chunk_words(words: list[Word], cfg: Config) -> list[CaptionLine]:
    lines: list[CaptionLine] = []
    current: list[Word] = []

    def flush() -> None:
        nonlocal current
        if current:
            lines.append(CaptionLine(current))
            current = []

    for word in words:
        if not word.text.strip():
            continue
        text = _clean(word.text)
        if not text:
            continue
        clean_word = Word(word.start, word.end, text)

        if current:
            proposed = " ".join(w.text for w in current + [clean_word])
            too_long = len(proposed) > cfg.caption_max_chars
            too_many = len(current) >= cfg.caption_max_words
            too_slow = clean_word.start - current[-1].end > PAUSE_BREAK
            too_wide = clean_word.end - current[0].start > MAX_LINE_SECONDS
            ends_sentence = current[-1].text.rstrip().endswith((".", "!", "?", "\u2026", ","))
            if too_long or too_many or too_slow or too_wide or (ends_sentence and len(current) >= 2):
                flush()
        current.append(clean_word)

    flush()
    return lines


# ---- per-animation event builders -----------------------------------------
# Each returns a list of (start, end, ass_text) for one caption line.

def _word_pop_events(line: CaptionLine, hot: str) -> list[tuple[float, float, str]]:
    out = []
    words = line.words
    for idx, active in enumerate(words):
        start = max(0.0, active.start)
        end = words[idx + 1].start if idx + 1 < len(words) else line.end
        if end <= start:
            end = start + 0.12

        pieces = []
        for j, word in enumerate(words):
            if j == idx:
                pieces.append(
                    f"{{\\c{hot}\\fscx108\\fscy108\\t(0,120,\\fscx100\\fscy100)}}{word.text}{{\\r}}"
                )
            else:
                pieces.append(word.text)
        out.append((start, end, " ".join(pieces)))
    return out


def _typewriter_events(line: CaptionLine) -> list[tuple[float, float, str]]:
    text = " ".join(w.text for w in line.words)
    n = len(text)
    if n == 0:
        return []
    step = (line.end - line.start) / n
    out = []
    for k in range(1, n + 1):
        start = max(0.0, line.start + (k - 1) * step)
        end = line.end if k == n else max(0.0, line.start + k * step)
        if end <= start:
            end = start + 0.05
        out.append((start, end, text[:k]))
    return out


def _karaoke_events(line: CaptionLine) -> list[tuple[float, float, str]]:
    # \kf fills each word from SecondaryColour to PrimaryColour over its duration.
    # A word's duration runs until the next word starts, so the sweep stays in sync
    # with the speech, including short pauses between words.
    parts = []
    words = line.words
    for i, w in enumerate(words):
        nxt = words[i + 1].start if i + 1 < len(words) else line.end
        cs = max(1, round((nxt - w.start) * 100))
        parts.append(f"{{\\kf{cs}}}{w.text}")
    return [(max(0.0, line.start), line.end, " ".join(parts))]


def build_ass(words: list[Word], cfg: Config, width: int, height: int) -> str:
    animation, font_key = validate_style(
        getattr(cfg, "caption_animation", "word_pop"),
        getattr(cfg, "caption_font", "montserrat"),
    )
    family, bold = FONTS[font_key]

    size = max(16, round(cfg.font_size * height / 1920))
    margin_v = {"lower": round(height * 0.20), "center": 0, "upper": round(height * 0.14)}.get(
        cfg.caption_position, round(height * 0.20)
    )
    alignment = {"lower": 2, "center": 5, "upper": 8}.get(cfg.caption_position, 2)

    base = _hex_to_ass(cfg.caption_color)
    hot = _hex_to_ass(cfg.highlight_color)
    outline = "&H00000000&"

    # Karaoke fills unsung text with SecondaryColour and sung text with PrimaryColour.
    if animation == "karaoke":
        primary, secondary = hot, base
    else:
        primary, secondary = base, base

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{family},{size},{primary},{secondary},{outline},&H64000000,{bold},0,0,0,100,100,0,0,1,{max(2, round(size * 0.055))},{max(1, round(size * 0.03))},{alignment},{round(width * 0.06)},{round(width * 0.06)},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events: list[str] = []
    for line in chunk_words(words, cfg):
        if not line.words:
            continue

        if animation == "word_pop":
            pairs = _word_pop_events(line, hot)
        elif animation == "typewriter":
            pairs = _typewriter_events(line)
        else:
            pairs = _karaoke_events(line)

        for start, end, text in pairs:
            events.append(
                f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,{text}"
            )

    return header + "\n".join(events) + ("\n" if events else "")


def write_ass(words: list[Word], cfg: Config, path: str | Path, width: int, height: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_ass(words, cfg, width, height), encoding="utf-8")
    return path


def _filter_path(path: Path) -> str:
    """Escape a path for use inside ffmpeg's subtitles= filter argument."""
    p = path.resolve().as_posix()
    return p.replace(":", "\\:").replace("'", "\\'")


def burn_captions(src: str | Path, ass_path: str | Path, out: str | Path,
                  ffmpeg: str, *, crf: int = 20) -> Path:
    """Render `src` with the ASS subtitles burned in, writing `out`."""
    src, ass_path, out = Path(src), Path(ass_path), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-v", "error",
        "-i", str(src),
        "-vf", f"subtitles='{_filter_path(ass_path)}'",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip()[-400:]
        raise ClipForgeError(f"Caption burn-in failed: {tail}")
    return out
