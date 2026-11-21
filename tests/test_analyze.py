from __future__ import annotations

import unittest

from clipforge.analyze import (
    Candidate,
    analyse,
    analyse_heuristic,
    dedupe,
    snap,
)
from clipforge.config import Config
from clipforge.transcript import Segment, Transcript


def build_transcript(count: int = 20, seg_len: float = 5.0) -> Transcript:
    filler = [
        "So here is the thing most people get wrong about this.",
        "I tracked every single hour for twelve months to find out.",
        "The answer was almost never the thing I was excited about.",
        "That one question saved me more time than any app ever did.",
        "You have to protect it before anyone books a meeting with you.",
        "Honestly the first week was brutal and I felt behind.",
        "By the end of the month I had shipped twice as much work.",
        "Stop optimising the small stuff and find the one thing.",
        "That is how you get more done in six weeks than in a year.",
        "Here is the biggest mistake I see people make every day.",
    ]
    segments = [
        Segment(i * seg_len, (i + 1) * seg_len, filler[i % len(filler)])
        for i in range(count)
    ]
    return Transcript(segments)


class HeuristicTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(clip_count=3, min_duration=20, max_duration=40)
        self.transcript = build_transcript()
        self.duration = 100.0

    def test_returns_requested_number(self):
        clips = analyse(self.transcript, self.cfg, self.duration)
        self.assertLessEqual(len(clips), 3)
        self.assertGreaterEqual(len(clips), 1)

    def test_respects_length_budget(self):
        for clip in analyse(self.transcript, self.cfg, self.duration):
            self.assertLessEqual(clip.duration, self.cfg.max_duration + 0.5)
            self.assertGreaterEqual(clip.duration, self.cfg.min_duration - 0.5)

    def test_clips_do_not_overlap_much(self):
        clips = analyse(self.transcript, self.cfg, self.duration)
        for i, a in enumerate(clips):
            for b in clips[i + 1 :]:
                overlap = max(0.0, min(a.end, b.end) - max(a.start, b.start))
                self.assertLess(overlap, min(a.duration, b.duration) * 0.3)

    def test_sorted_by_score(self):
        scores = [c.score for c in analyse(self.transcript, self.cfg, self.duration)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_every_clip_has_a_title(self):
        for clip in analyse(self.transcript, self.cfg, self.duration):
            self.assertTrue(clip.title.strip())

    def test_stays_inside_the_video(self):
        for clip in analyse(self.transcript, self.cfg, self.duration):
            self.assertGreaterEqual(clip.start, 0.0)
            self.assertLessEqual(clip.end, self.duration)

    def test_short_video_still_produces_a_clip(self):
        short = build_transcript(count=3, seg_len=4.0)
        clips = analyse_heuristic(short, self.cfg, 12.0)
        self.assertGreaterEqual(len(clips), 1)

    def test_empty_transcript_raises(self):
        with self.assertRaises(Exception):
            analyse(Transcript([]), self.cfg, 100.0)


class SnapTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(min_duration=10, max_duration=60)
        self.transcript = Transcript(
            [
                Segment(0.0, 5.0, "First sentence here."),
                Segment(5.0, 10.0, "Second sentence here."),
                Segment(10.0, 15.0, "Third sentence here."),
            ]
        )

    def test_start_is_pulled_back_to_sentence_start(self):
        snapped = snap(Candidate(6.2, 14.0), self.transcript, self.cfg, 15.0)
        self.assertAlmostEqual(snapped.start, 5.0 - 0.15, places=2)

    def test_end_is_pushed_forward_to_sentence_end(self):
        snapped = snap(Candidate(5.0, 13.8), self.transcript, self.cfg, 15.0)
        self.assertAlmostEqual(snapped.end, 15.0, places=2)

    def test_short_clip_is_stretched_to_minimum(self):
        snapped = snap(Candidate(5.0, 10.0), self.transcript, self.cfg, 60.0)
        self.assertGreaterEqual(snapped.duration, 10.0)

    def test_long_clip_is_trimmed_to_maximum(self):
        cfg = Config(min_duration=5, max_duration=8)
        snapped = snap(Candidate(0.0, 15.0), self.transcript, cfg, 15.0)
        self.assertLessEqual(snapped.duration, 8.0)

    def test_never_exceeds_video_length(self):
        snapped = snap(Candidate(14.0, 40.0), self.transcript, self.cfg, 15.0)
        self.assertLessEqual(snapped.end, 15.0)


class DedupeTests(unittest.TestCase):
    def test_identical_clips_collapse(self):
        clips = [
            Candidate(10, 40, score=90),
            Candidate(11, 41, score=80),
            Candidate(100, 130, score=70),
        ]
        kept = dedupe(clips)
        self.assertEqual(len(kept), 2)

    def test_best_score_wins(self):
        clips = [Candidate(10, 40, score=50), Candidate(12, 42, score=90)]
        self.assertEqual(dedupe(clips)[0].score, 90)


if __name__ == "__main__":
    unittest.main()
