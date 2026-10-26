"""Unit tests for scripts/timeline.py on synthetic signals (no model, no GPU)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import timeline as tl  # noqa: E402

SR = 24000


def burst(sec, amp=0.3, hz=200):
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def quiet(sec):
    return np.zeros(int(sec * SR), np.float32)


class CleanTest(unittest.TestCase):
    def test_trim_removes_edges_and_keeps_pad(self):
        x = np.concatenate([quiet(0.8), burst(1.0), quiet(0.9)])
        y = tl.trim(x, SR, pad=0.04)
        self.assertAlmostEqual(len(y) / SR, 1.08, delta=0.03)

    def test_collapse_shortens_only_long_inner_silences(self):
        x = np.concatenate([burst(0.5), quiet(1.2), burst(0.5), quiet(0.3), burst(0.5)])
        y, removed = tl.collapse_pauses(x, SR, max_pause=0.45, keep=0.30)
        self.assertAlmostEqual(removed, 0.9, delta=0.05)
        self.assertAlmostEqual(len(y) / SR, len(x) / SR - removed, delta=0.03)
        y2, removed2 = tl.collapse_pauses(np.concatenate([burst(0.5), quiet(0.3), burst(0.5)]), SR)
        self.assertEqual(removed2, 0.0)

    def test_valley_lands_in_the_silence_and_silence_len(self):
        x = np.concatenate([burst(1.0), quiet(0.4), burst(1.0)])
        v = tl.valley(x, SR, 1.15, window=0.15)
        self.assertTrue(1.0 <= v <= 1.4, v)
        self.assertAlmostEqual(tl.silence_len_at(x, SR, 1.2), 0.4, delta=0.03)
        self.assertEqual(tl.silence_len_at(x, SR, 0.5), 0.0)

    def test_active_db_orders_levels(self):
        loud, soft = burst(1.0, 0.3), burst(1.0, 0.03)
        self.assertAlmostEqual(tl.active_db(loud, SR) - tl.active_db(soft, SR), 20.0, delta=1.0)


class AlignTest(unittest.TestCase):
    def test_first_word_times_and_ratio(self):
        lines = [["the", "best", "of", "it"], ["lives", "in", "a", "few", "heads"]]
        heard = [("the", 0.1, 0.2), ("best", 0.2, 0.5), ("of", 0.5, 0.6), ("it", 0.6, 0.8), ("lives", 1.9, 2.2),
                 ("in", 2.2, 2.3), ("a", 2.3, 2.4), ("few", 2.4, 2.7), ("heads", 2.7, 3.1)]
        firsts, ratio = tl.align_boundaries(heard, lines)
        self.assertEqual(firsts, [0.1, 1.9])
        self.assertEqual(ratio, 1.0)

    def test_missing_first_word_uses_next_matched_and_ratio_drops(self):
        lines = [["a", "b"], ["c", "d", "e"]]
        heard = [("a", 0.0, 0.1), ("b", 0.1, 0.2), ("d", 1.0, 1.1), ("e", 1.1, 1.2)]
        firsts, ratio = tl.align_boundaries(heard, lines)
        self.assertEqual(firsts, [0.0, 1.0])
        self.assertAlmostEqual(ratio, 0.8)
        self.assertEqual(tl.align_boundaries([], lines), ([None, None], 0.0))


class PlanTest(unittest.TestCase):
    def test_overrun_is_sped_up_within_the_window(self):
        r = tl.plan([4.5], [0], [10.0], 14.0)
        self.assertGreater(r["f"], 1.1)
        self.assertEqual(r["over"], 0.0)

    def test_short_sentence_with_big_tail_is_slowed_a_little_not_stretched_to_fit(self):
        r = tl.plan([2.5], [0], [10.0], 14.0)
        self.assertLess(r["f"], 1.0)
        self.assertGreaterEqual(r["f"], 0.88)

    def test_fitting_sentence_keeps_its_pace(self):
        r = tl.plan([3.6], [0], [10.0], 14.0)
        self.assertEqual(r["f"], 1.0)

    def test_internal_break_lands_on_its_slot_and_pause_stays_reasonable(self):
        r = tl.plan([3.0, 5.0], [0, 0.3], [7.5, 11.5], 18.0)
        self.assertAlmostEqual(r["starts"][1], 11.5, delta=0.01)
        pause = r["starts"][1] - (r["starts"][0] + 3.0 / r["f"])
        self.assertLessEqual(pause, 1.0)

    def test_overrunning_first_segment_pushes_the_second_back_without_overlap(self):
        r = tl.plan([5.0, 3.0], [0, 0.2], [0.0, 4.0], 9.0)
        end0 = r["starts"][0] + 5.0 / r["f"]
        self.assertGreaterEqual(r["starts"][1], end0 + 0.07)
        self.assertLessEqual(r["over"], 0.05)

    def test_impossible_window_reports_the_overrun(self):
        r = tl.plan([9.0], [0], [0.0], 3.0)
        self.assertGreater(r["over"], 0.5)
        self.assertEqual(r["f"], 1.3)


class GroupTest(unittest.TestCase):
    def test_sentences_span_lines_that_do_not_end_in_terminal_punctuation(self):
        rows = [{"text": t} for t in ["@neutral One thing here.", "@serious The best of it lives in a few heads:", "@serious how to read it, all of it.",
                                      "Is it so?", "Last"]]
        g = tl.sentence_groups(rows)
        self.assertEqual([len(x) for x in g], [1, 2, 1, 1])
        self.assertEqual(tl.strip_marks("@serious how  to @warm read"), "how to read")


if __name__ == "__main__":
    unittest.main()
