from __future__ import annotations

import unittest

from clipforge.captions import build_ass, chunk_words
from clipforge.config import Config
from clipforge.transcript import Word
from clipforge.util import ass_time


def make_words(text: str, start: float = 0.0, per_word: float = 0.4) -> list[Word]:
    words = []
    cursor = start
    for token in text.split():
        words.append(Word(cursor, cursor + per_word, token))
        cursor += per_word
    return words


class ChunkTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(caption_max_words=3, caption_max_chars=20)

    def test_respects_word_limit(self):
        lines = chunk_words(make_words("one two three four five six seven"), self.cfg)
        self.assertTrue(all(len(line.words) <= 3 for line in lines))
        self.assertEqual(sum(len(line.words) for line in lines), 7)

    def test_respects_char_limit(self):
        lines = chunk_words(make_words("extraordinarily comprehensive explanation"), self.cfg)
        for line in lines:
            self.assertLessEqual(len(" ".join(w.text for w in line.words)), 20)

    def test_breaks_on_long_pause(self):
        words = make_words("before after")
        words[1] = Word(words[0].end + 1.5, words[1].end + 1.5, "after")
        lines = chunk_words(words, self.cfg)
        self.assertEqual(len(lines), 2)

    def test_strips_text_ass_cannot_render(self):
        words = make_words("hello \U0001F600 world {brace}")
        lines = chunk_words(words, self.cfg)
        text = " ".join(w.text for line in lines for w in line.words)
        self.assertNotIn("\U0001F600", text)
        self.assertNotIn("{", text)


class AssTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.words = make_words("the secret is consistency")

    def test_header_matches_output_size(self):
        ass = build_ass(self.words, self.cfg, 1080, 1920)
        self.assertIn("PlayResX: 1080", ass)
        self.assertIn("PlayResY: 1920", ass)

    def test_one_event_per_word(self):
        ass = build_ass(self.words, self.cfg, 1080, 1920)
        events = [ln for ln in ass.splitlines() if ln.startswith("Dialogue:")]
        self.assertEqual(len(events), len(self.words))

    def test_every_event_draws_its_whole_line(self):
        lines = chunk_words(self.words, self.cfg)
        events = [ln for ln in build_ass(self.words, self.cfg, 1080, 1920).splitlines()
                  if ln.startswith("Dialogue:")]
        self.assertEqual(len(events), len(self.words))
        cursor = 0
        for line in lines:
            for event in events[cursor:cursor + len(line.words)]:
                for word in line.words:
                    self.assertIn(word.text, event)
            cursor += len(line.words)

    def test_active_word_is_highlighted(self):
        cfg = Config(highlight_color="00E5FF")
        ass = build_ass(self.words, cfg, 1080, 1920)
        # White -> &H00FFFFFF, cyan -> &H00FFE500 (BGR order).
        self.assertIn("\\c&H00FFE500&", ass)

    def test_timestamps_increase(self):
        ass = build_ass(self.words, self.cfg, 1080, 1920)
        times = [ln.split(",")[1] for ln in ass.splitlines() if ln.startswith("Dialogue:")]
        self.assertEqual(times, sorted(times))

    def test_font_size_scales_with_height(self):
        small = build_ass(self.words, Config(font_size=76), 720, 1280)
        big = build_ass(self.words, Config(font_size=76), 1080, 1920)
        self.assertIn(f",{round(76 * 1280 / 1920)},", small)
        self.assertIn(",76,", big)


class TimeTests(unittest.TestCase):
    def test_ass_time(self):
        self.assertEqual(ass_time(0), "0:00:00.00")
        self.assertEqual(ass_time(61.5), "0:01:01.50")
        self.assertEqual(ass_time(3723.07), "1:02:03.07")


if __name__ == "__main__":
    unittest.main()
