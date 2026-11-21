from __future__ import annotations

import unittest

from clipforge.transcript import (
    Segment,
    Transcript,
    Word,
    load_transcript,
    parse_srt,
    save_json,
    save_srt,
)


SAMPLE_SRT = """1
00:00:01,000 --> 00:00:04,500
Hello there, this is the first line.

2
00:00:04,500 --> 00:00:09,000
And here is a second line that runs a bit longer.
"""


class ParseTests(unittest.TestCase):
    def test_parse_srt_handles_crlf(self):
        segments = parse_srt(SAMPLE_SRT.replace("\n", "\r\n"))
        self.assertEqual(len(segments), 2)
        self.assertAlmostEqual(segments[0].start, 1.0)
        self.assertAlmostEqual(segments[1].end, 9.0)
        self.assertIn("second line", segments[1].text)

    def test_empty_srt_raises(self):
        with self.assertRaises(Exception):
            parse_srt("")


class WordTests(unittest.TestCase):
    def setUp(self):
        self.transcript = Transcript(
            [
                Segment(0.0, 2.0, "one two three four"),
                Segment(2.0, 4.0, "five six"),
            ]
        )

    def test_synthesised_words_cover_the_segment_in_order(self):
        words = self.transcript.words()
        self.assertEqual(len(words), 6)
        for prev, nxt in zip(words, words[1:]):
            self.assertLessEqual(prev.start, nxt.start)
        self.assertGreaterEqual(words[-1].end, 3.99)

    def test_words_are_weighted_by_length(self):
        words = self.transcript.words()
        # "three" is longer than "one", so it should be held longer.
        one = next(w for w in words if w.text == "one")
        three = next(w for w in words if w.text == "three")
        self.assertGreater(three.end - three.start, one.end - one.start)

    def test_slice_clips_to_window(self):
        sliced = self.transcript.slice(1.0, 3.0)
        self.assertEqual(len(sliced.segments), 2)
        self.assertAlmostEqual(sliced.segments[0].start, 1.0)
        self.assertAlmostEqual(sliced.segments[-1].end, 3.0)

    def test_timed_text_uses_m_ss(self):
        self.assertIn("[0:02] five six", self.transcript.timed_text())


class RoundTripTests(unittest.TestCase):
    def test_json_round_trip(self):
        import tempfile
        from pathlib import Path

        original = Transcript([Segment(0.5, 1.5, "hi there", [Word(0.5, 0.9, "hi")])], "en")
        with tempfile.TemporaryDirectory() as tmp:
            path = save_json(original, Path(tmp) / "t.json")
            loaded = load_transcript(path)
        self.assertEqual(loaded.language, "en")
        self.assertEqual(loaded.segments[0].text, "hi there")
        self.assertEqual(loaded.segments[0].words[0].text, "hi")

    def test_srt_round_trip(self):
        import tempfile
        from pathlib import Path

        original = Transcript([Segment(65.25, 67.0, "later on")])
        with tempfile.TemporaryDirectory() as tmp:
            path = save_srt(original, Path(tmp) / "t.srt")
            loaded = load_transcript(path)
        self.assertAlmostEqual(loaded.segments[0].start, 65.25, places=2)
        self.assertEqual(loaded.segments[0].text, "later on")


if __name__ == "__main__":
    unittest.main()
