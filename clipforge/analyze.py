"""Deciding which seconds of a video are worth turning into clips.

Two strategies that produce the same output shape:

* ``analyse_llm``       - ask an OpenAI-compatible chat model, with a carefully
                          tuned editor prompt. Best quality.
* ``analyse_heuristic`` - score a sliding window over the transcript with
                          hand-tuned signals. No API, no network, instant.

Both run through the same boundary snapping and overlap removal, so clips are
always sentence-aligned and never duplicate each other.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable

from .config import Config
from .prompts import SYSTEM_PROMPT, USER_PROMPT
from .transcript import Segment, Transcript
from .util import ClipForgeError, extract_json, fmt_ts, http_json

Progress = Callable[[float, str], None]


@dataclass
class Candidate:
    start: float
    end: float
    title: str = ""
    hook: str = ""
    score: float = 0.0
    reason: str = ""
    source: str = "heuristic"

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "title": self.title,
            "hook": self.hook,
            "score": round(self.score, 1),
            "reason": self.reason,
            "source": self.source,
        }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def has_llm(cfg: Config) -> bool:
    return bool((cfg.llm_base_url or "").strip())


def analyse(
    transcript: Transcript,
    cfg: Config,
    duration: float,
    progress: Progress | None = None,
) -> list[Candidate]:
    if not transcript.segments:
        raise ClipForgeError("The transcript is empty, so there is nothing to clip.")

    if has_llm(cfg):
        clips = analyse_llm(transcript, cfg, duration, progress)
    else:
        if progress:
            progress(0, "No LLM configured - using the built-in highlight scorer")
        clips = analyse_heuristic(transcript, cfg, duration)

    clips = [snap(c, transcript, cfg, duration) for c in clips]
    clips = [c for c in clips if c.duration >= max(3.0, cfg.min_duration * 0.6)]
    clips = dedupe(clips)

    if not clips:
        # Very short source, or everything got filtered: fall back to the whole thing.
        end = min(duration, max(cfg.max_duration, transcript.duration))
        clips = [
            Candidate(
                0.0, end, "Full clip",
                transcript.segments[0].text[:80] if transcript.segments else "",
                50.0, "Source was too short for highlight selection.",
            )
        ]

    clips.sort(key=lambda c: c.score, reverse=True)
    return clips[: max(1, cfg.clip_count)]


# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #
def _chat_url(base: str) -> str:
    base = base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def analyse_llm(
    transcript: Transcript,
    cfg: Config,
    duration: float,
    progress: Progress | None = None,
) -> list[Candidate]:
    if progress:
        progress(0, f"Asking {cfg.llm_model} which moments to clip")

    timed = transcript.timed_text()
    # Keep the request inside a sane context budget; 120k chars ~ 30k tokens.
    max_chars = int(os.environ.get("CLIPFORGE_MAX_TRANSCRIPT_CHARS", "120000"))
    if len(timed) > max_chars:
        timed = timed[:max_chars] + "\n[... transcript truncated ...]"

    prompt = USER_PROMPT.format(
        duration=fmt_ts(duration),
        count=cfg.clip_count,
        min_dur=int(cfg.min_duration),
        max_dur=int(cfg.max_duration),
        transcript=timed,
    )

    payload = {
        "model": cfg.llm_model,
        "temperature": cfg.llm_temperature,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {}
    if cfg.llm_api_key:
        headers["Authorization"] = f"Bearer {cfg.llm_api_key}"

    def call(with_json_mode: bool) -> dict:
        body = dict(payload)
        if with_json_mode:
            body["response_format"] = {"type": "json_object"}
        return http_json(_chat_url(cfg.llm_base_url), body, headers, timeout=cfg.llm_timeout)

    try:
        reply = call(cfg.llm_json_mode)
    except ClipForgeError:
        # Many self-hosted servers reject response_format; retry plain.
        if not cfg.llm_json_mode:
            raise
        if progress:
            progress(20, "Server rejected JSON mode, retrying without it")
        reply = call(False)

    try:
        content = reply["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as err:
        raise ClipForgeError(f"Unexpected reply from the LLM endpoint:\n{str(reply)[:400]}") from err

    data = extract_json(content)
    raw = data.get("clips", data) if isinstance(data, dict) else data
    if not isinstance(raw, list):
        raise ClipForgeError("The model replied with JSON, but not a list of clips.")

    clips: list[Candidate] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        try:
            score = float(item.get("score", 0))
        except (TypeError, ValueError):
            score = 0.0
        clips.append(
            Candidate(
                start=max(0.0, start),
                end=min(duration, end),
                title=str(item.get("title", "")).strip(),
                hook=str(item.get("hook", "")).strip(),
                score=max(0.0, min(100.0, score)),
                reason=str(item.get("reason", "")).strip(),
                source="llm",
            )
        )

    if not clips:
        if progress:
            progress(30, "Model returned nothing usable, falling back to the heuristic scorer")
        return analyse_heuristic(transcript, cfg, duration)
    return clips


# --------------------------------------------------------------------------- #
# Heuristic
# --------------------------------------------------------------------------- #
HOOK_WORDS = {
    "how", "why", "what", "when", "never", "always", "nobody", "everyone",
    "secret", "truth", "mistake", "biggest", "worst", "best", "crazy", "insane",
    "honestly", "actually", "listen", "imagine", "stop", "don't", "you",
}
PAYOFF_WORDS = {
    "because", "reason", "result", "turns out", "the point", "so that's",
    "here's", "the trick", "the key", "the thing is", "which means",
    "that's why", "in the end", "lesson",
}
EMOTION_WORDS = {
    "love", "hate", "amazing", "terrible", "shocked", "scared", "excited",
    "furious", "hilarious", "painful", "incredible", "unbelievable", "wild",
    "brutal", "perfect", "disaster", "obsessed", "proud", "embarrassing",
}
FILLER_WORDS = {
    "um", "uh", "erm", "like,", "you know", "i mean", "kind of", "sort of",
    "anyway", "so yeah", "basically", "actually", "literally",
}
_PUNCT = re.compile(r"[^\w\s']+", re.UNICODE)


def _norm(text: str) -> str:
    return _PUNCT.sub(" ", text.lower())


def _count_matches(text: str, vocab: set[str]) -> int:
    norm = " " + _norm(text) + " "
    hits = 0
    for term in vocab:
        if " " in term:
            if term in norm:
                hits += 1
        elif f" {term} " in norm:
            hits += 1
    return hits


def analyse_heuristic(
    transcript: Transcript,
    cfg: Config,
    duration: float,
    progress: Progress | None = None,
) -> list[Candidate]:
    if progress:
        progress(5, "Scoring candidate moments")

    segments = [s for s in transcript.segments if s.text.strip()]
    if not segments:
        raise ClipForgeError("The transcript has no spoken text to analyse.")

    min_dur = max(5.0, cfg.min_duration)
    max_dur = max(min_dur + 5.0, cfg.max_duration)
    windows: list[Candidate] = []

    for i in range(len(segments)):
        acc: list[Segment] = []
        for j in range(i, len(segments)):
            acc.append(segments[j])
            span = acc[-1].end - acc[0].start
            if span > max_dur:
                break
            if span < min_dur:
                continue
            windows.append(_score_window(acc, duration))

    if not windows:
        # Everything is shorter than min_duration: score whole speaking span.
        span = segments[-1].end - segments[0].start
        if span >= 3.0:
            windows.append(_score_window(segments, duration))

    windows = dedupe(windows, overlap=0.35)
    windows.sort(key=lambda c: c.score, reverse=True)
    if progress:
        progress(100, f"Scored {len(windows)} candidates")
    return windows[: max(cfg.clip_count * 3, 12)]


def _score_window(segs: list[Segment], duration: float) -> Candidate:
    start, end = segs[0].start, segs[-1].end
    span = max(0.1, end - start)
    text = " ".join(s.text for s in segs).strip()
    if not text:
        return Candidate(start, end, score=0.0)

    words = text.split()
    wps = len(words) / span  # words per second: energy / talking speed

    first = " ".join(s.text for s in segs[: clamp_idx(len(segs), 3)]).strip()
    last = segs[-1].text.strip()

    hook = _count_matches(first, HOOK_WORDS)
    payoff = _count_matches(text, PAYOFF_WORDS)
    emotion = _count_matches(text, EMOTION_WORDS)
    filler = _count_matches(text, FILLER_WORDS)

    score = 40.0
    score += min(14.0, hook * 5.0)
    score += min(12.0, payoff * 4.0)
    score += min(10.0, emotion * 3.5)
    score -= min(12.0, filler * 3.0)

    # Fast, dense speech reads as confident and holds attention.
    if 2.2 <= wps <= 4.2:
        score += 10.0
    elif wps < 1.4:
        score -= 8.0

    # Numbers make hooks concrete ("I lost 40k in a week").
    score += 5.0 if re.search(r"\b\d[\d,.]*\b", text) else 0.0

    if _starts_question(first):
        score += 7.0
    if first.rstrip().endswith("?"):
        score += 3.0

    # Reward clips that finish a thought and avoid trailing-off ends.
    if last.rstrip().endswith((".", "!", "?", "…")):
        score += 6.0
    else:
        score -= 4.0

    # Intros and outros are usually weaker than the meat of the video.
    if duration > 120:
        if start < 15:
            score -= 6.0
        if end > duration - 15:
            score -= 5.0

    # Sweet spot is 25-45 seconds.
    ideal = 35.0
    score -= abs(span - ideal) * 0.18

    score = max(0.0, min(100.0, score))
    title = " ".join(words[:7]).strip(" ,.!?-")
    return Candidate(
        start=start,
        end=end,
        title=_title_case(title),
        hook=" ".join(first.split()[:14]),
        score=score,
        reason="Strong opening, complete thought, high speech energy.",
        source="heuristic",
    )


def clamp_idx(n: int, limit: int) -> int:
    return min(n, limit)


def _starts_question(text: str) -> bool:
    return bool(re.match(r"\s*(how|why|what|when|where|who|do|does|did|can|could|would|should|is|are|was|were|have|has)\b", text.lower()))


def _title_case(text: str) -> str:
    if not text:
        return ""
    return text[0].upper() + text[1:]


# --------------------------------------------------------------------------- #
# Boundary snapping + overlap removal
# --------------------------------------------------------------------------- #
def snap(candidate: Candidate, transcript: Transcript, cfg: Config, duration: float) -> Candidate:
    """Nudge clip edges onto sentence boundaries, then enforce the length budget."""
    start = max(0.0, min(candidate.start, duration))
    end = max(start + 0.5, min(candidate.end, duration))

    start = _snap_start(start, transcript.segments)
    end = _snap_end(end, transcript.segments)

    # A little air on both sides sounds natural but not enough to be dead space.
    start = max(0.0, start - 0.15)
    end = min(duration, end + 0.25)

    if end - start < cfg.min_duration:
        end = min(duration, start + cfg.min_duration)
        if end - start < cfg.min_duration:
            start = max(0.0, end - cfg.min_duration)
    if end - start > cfg.max_duration:
        end = start + cfg.max_duration

    candidate.start = round(start, 3)
    candidate.end = round(end, 3)
    if not candidate.title:
        candidate.title = _title_from_text(candidate, transcript)
    return candidate


def _snap_start(start: float, segments: list[Segment]) -> float:
    for seg in segments:
        if seg.start <= start <= seg.end:
            # Only pull back if we are close to the beginning of the sentence,
            # otherwise we would swallow unrelated content.
            if start - seg.start <= 2.5:
                return seg.start
            return start
    return start


def _snap_end(end: float, segments: list[Segment]) -> float:
    for seg in segments:
        if seg.start <= end <= seg.end:
            if seg.end - end <= 2.5:
                return seg.end
            return end
    return end


def _title_from_text(candidate: Candidate, transcript: Transcript) -> str:
    window = transcript.slice(candidate.start, min(candidate.end, candidate.start + 12))
    words = window.full_text().split()
    title = " ".join(words[:7]).strip(" ,.!?-")
    return _title_case(title) or f"Clip at {fmt_ts(candidate.start)}"


def dedupe(clips: list[Candidate], overlap: float = 0.25) -> list[Candidate]:
    """Drop clips that cover substantially the same ground as a better one."""
    kept: list[Candidate] = []
    for clip in sorted(clips, key=lambda c: c.score, reverse=True):
        if any(_overlap(clip, other) > overlap for other in kept):
            continue
        kept.append(clip)
    return kept


def _overlap(a: Candidate, b: Candidate) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    if inter <= 0:
        return 0.0
    # Fraction of the shorter clip that is duplicated.
    shortest = min(a.duration, b.duration)
    return inter / shortest if shortest else 0.0
