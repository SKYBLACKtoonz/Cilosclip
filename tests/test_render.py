from __future__ import annotations

import unittest
from pathlib import Path

from clipforge.config import Config
from clipforge.media import MediaInfo
from clipforge.render import _clip_words, _layout_for, _piecewise, build_filtergraph
from clipforge.transcript import Segment, Transcript, Word


HORIZONTAL = MediaInfo(Path("a.mp4"), 120, 1920, 1080, 30, True)
VERTICAL = MediaInfo(Path("b.mp4"), 60, 1080, 1920, 30, True)


class LayoutTests(unittest.TestCase):
    def test_blur_fills_vertically(self):
        graph = build_filtergraph(HORIZONTAL, Config(), "blur", None, None)
        self.assertIn("gblur", graph)
        self.assertIn("overlay", graph)

    def test_pad_letterboxes(self):
        graph = build_filtergraph(HORIZONTAL, Config(), "pad", None, None)
        self.assertIn("pad=1080:1920", graph)

    def test_crop_fills_frame(self):
        graph = build_filtergraph(HORIZONTAL, Config(), "crop", None, None)
        self.assertIn("crop=1080:1920", graph)

    def test_tracked_crop_emits_expression(self):
        graph = build_filtergraph(HORIZONTAL, Config(), "crop", None, [(0.0, 0.2), (1.0, 0.8)])
        self.assertIn("crop=1080:1920:x='", graph)
        self.assertIn("if(lt(t,", graph)

    def test_captions_added_to_graph(self):
        graph = build_filtergraph(HORIZONTAL, Config(), "blur", "captions.ass", None)
        self.assertIn("subtitles=captions.ass", graph)

    def test_vertical_source_skips_blur(self):
        self.assertEqual(_layout_for(VERTICAL, Config(layout="blur")), "crop")

    def test_horizontal_source_keeps_blur(self):
        self.assertEqual(_layout_for(HORIZONTAL, Config(layout="blur")), "blur")

    def test_unknown_layout_falls_back_to_blur(self):
        self.assertEqual(_layout_for(HORIZONTAL, Config(layout="weird")), "blur")


class ExpressionTests(unittest.TestCase):
    def test_expression_is_clamped(self):
        expr = _piecewise("t", [(0.0, 10.0), (1.0, 50.0)], clamp_max=100.0)
        self.assertTrue(expr.startswith("max(0,min(100.0,"))
        self.assertIn("if(lt(t,", expr)

    def test_single_point_is_constant(self):
        expr = _piecewise("t", [(0.0, 42.0)], clamp_max=100.0)
        self.assertEqual(expr, "max(0,min(100.0,42.0))")


class ClipWordTests(unittest.TestCase):
    def test_words_are_shifted_to_clip_zero(self):
        transcript = Transcript([Segment(10.0, 12.0, "one two")])
        words = _clip_words(transcript, 10.0, 12.0)
        self.assertAlmostEqual(words[0].start, 0.0, places=2)
        self.assertLessEqual(words[-1].end, 2.0 + 1e-6)

    def test_words_never_go_negative(self):
        transcript = Transcript(
            [Segment(9.0, 12.0, "one two", [Word(9.0, 9.5, "one"), Word(9.5, 10.2, "two")])]
        )
        words = _clip_words(transcript, 9.5, 12.0)
        for word in words:
            self.assertGreaterEqual(word.start, 0.0)
            self.assertGreater(word.end, word.start)


if __name__ == "__main__":
    unittest.main()
