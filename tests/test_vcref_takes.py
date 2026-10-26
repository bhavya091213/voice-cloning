"""python -m unittest discover -s tests   (run with the venv python; needs numpy + soundfile only)"""
import json
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import takes  # noqa: E402
import vcref  # noqa: E402


def make_refs(d):
    clips = {}
    for tag, hz in (("neutral", 220), ("sad", 180), ("question", 260)):
        t = np.arange(48000) / 48000
        sf.write(d / f"{tag}.wav", (0.2 * np.sin(2 * np.pi * hz * t)).astype("float32"), 48000, subtype="PCM_24")
        clips[tag] = {"text": f"{tag} text.", "seconds": 1.0, "label": tag, "source": "A"}
    (d / "index.json").write_text(json.dumps({"default": "neutral", "clips": clips}), encoding="utf-8")


class TakesTest(unittest.TestCase):
    def test_punctuation_rules_and_merge(self):
        t = takes.split("One. Two. Is it three? Four! Five.")
        self.assertEqual([(x.tag, x.text) for x in t],
                         [("neutral", "One. Two."), ("question", "Is it three?"), ("excited", "Four!"), ("neutral", "Five.")])
        self.assertEqual(t[-1].gap, 0.0)

    def test_explicit_prefix_and_fallback(self):
        missing = []
        t = takes.split("@sad I miss it. Really.\n@nope Hello there.", available={"neutral", "sad"}, missing=missing)
        self.assertEqual([x.tag for x in t], ["sad", "neutral"])
        self.assertEqual(missing, ["nope"])

    def test_paragraph_gap_and_custom_rules(self):
        import re
        rules = [(re.compile(r"(?i)\bsorry\b"), "sad")]
        t = takes.split("Fine. Sorry about that.\n\nNext paragraph.", rules=rules)
        self.assertEqual([x.tag for x in t], ["neutral", "sad", "neutral"])
        self.assertEqual(t[1].gap, takes.GAP_PARA)

    def test_marker_mid_line_is_not_spoken(self):
        t = takes.split("Fine. @playful Oh, you think so? @serious Listen up. Done.", available={"neutral", "playful", "serious"})
        self.assertEqual([(x.tag, x.text) for x in t],
                         [("neutral", "Fine."), ("playful", "Oh, you think so?"), ("serious", "Listen up. Done.")])
        self.assertFalse(any("@" in x.text for x in t))

    def test_max_chars_splits_long_runs(self):
        t = takes.split(" ".join(["This is a sentence."] * 30), max_chars=100)
        self.assertGreater(len(t), 3)
        self.assertTrue(all(len(x.text) <= 100 for x in t))


class DirectorTest(unittest.TestCase):
    def test_check_rejects_changed_words_and_cleans_fields(self):
        import director
        line = "Imagine that method, step by step, done."
        self.assertIsNone(director.check({"text": "Imagine that method."}, line, director.TAGS))
        item, fixes = director.check({"text": line, "tag": "Dreamy", "pause": "huge", "style": "(Warm), slow",
                                      "stress": ["step by step", "nothing"]}, line, director.TAGS)
        self.assertEqual((item["tag"], item["pause"], item["style"]), ("neutral", "short", "warm, slow"))
        self.assertTrue(set(item["stress"]) <= {"step", "by"} or item["stress"] == ["step"])  # capped at 1 word in 7
        self.assertTrue(any("nothing" in f for f in fixes))

    def test_smoothing_merges_sentence_lines_and_blips(self):
        import director
        mk = lambda n, text, tag: {"n": n, "text": text, "tag": tag}  # noqa: E731
        items = [mk(1, "One thing.", "warm"), mk(2, "Another,", "serious"), mk(3, "and more.", "warm"),
                 mk(4, "Third.", "warm"), mk(5, "Fourth.", "warm"), mk(6, "Fifth?", "question"), mk(7, "Sixth.", "warm")]
        director.smooth_tags(items)
        self.assertEqual([i["tag"] for i in items], ["warm", "warm", "warm", "warm", "warm", "question", "warm"])

    def test_question_kind(self):
        import director
        k = director.question_kind
        self.assertEqual(k("Want to see the map of the trail?"), "yesno")
        self.assertEqual(k("Did it work?"), "yesno")
        self.assertEqual(k("So, didn't you hear me?"), "yesno")
        self.assertEqual(k("You finished it?"), "yesno")
        self.assertEqual(k("Why would you ask?"), "wh")
        self.assertEqual(k("And what if every volunteer could learn it in a day?"), "wh")
        self.assertEqual(k("How come nobody told me?"), "wh")
        self.assertIsNone(k("Oh, you think you're so clever, don't you?"))  # tag question: left to the model
        self.assertIsNone(k("This is a statement."))

    def test_enforce_questions_makes_the_tag_follow_the_contour(self):
        import director
        mk = lambda n, text, tag, contour="falls": {"n": n, "text": text, "tag": tag, "contour": contour}  # noqa: E731
        items = [mk(1, "It works.", "warm"), mk(2, "Want to see the map", "warm"), mk(3, "of the trail?", "warm"),
                 mk(4, "Why would you ask?", "question"), mk(5, "Fine.", "neutral")]
        log = director.enforce_questions(items)
        self.assertEqual([i["tag"] for i in items], ["warm", "question", "question", "neutral", "neutral"])
        self.assertEqual([i["contour"] for i in items], ["falls", "rises", "rises", "falls", "falls"])
        self.assertEqual(len(log), 3)
        self.assertEqual(director.enforce_questions([mk(1, "Hi?", "warm")], ["neutral", "warm"]), [])  # no question clip: untouched

    def test_check_derives_form_and_contour_when_missing(self):
        import director
        item, fixes = director.check({"text": "Want to see it?", "tag": "warm"}, "Want to see it?", director.TAGS)
        self.assertEqual((item["form"], item["contour"]), ("yes-no question", "rises"))
        item, _ = director.check({"text": "Why now?", "tag": "warm", "form": "wh-question", "contour": "falls"}, "Why now?", director.TAGS)
        self.assertEqual((item["form"], item["contour"]), ("wh-question", "falls"))

    def test_prompt_files_are_loaded_and_mention_the_rules(self):
        import director
        self.assertIn("yes-no question", director.SYSTEM)
        self.assertIn("contour", director.SYSTEM)
        self.assertIn("LINE MODE", director.LINE_MODE)

    def test_plain_emit_turns_long_pauses_into_paragraph_breaks(self):
        import director
        res = {"timed": False, "lines": [{"head": "", "text": "One.", "music": False, "tag": "warm", "stress": [], "pause": "beat"},
                                         {"head": "", "text": "Two.", "music": False, "tag": "warm", "stress": [], "pause": "short"}]}
        self.assertEqual(director.emit(res), "@warm One.\n\n@warm Two.\n")

    def test_emit_marks_tag_and_keeps_music_lines(self):
        import director
        res = {"lines": [{"head": "0:00.0 - 0:03.0   ", "text": "[Music only]", "music": True},
                         {"head": "0:03.0 - 0:07.5   ", "text": "A rare day.", "music": False, "tag": "warm", "stress": ["rare"]}]}
        self.assertEqual(director.emit(res), "0:00.0 - 0:03.0   [Music only]\n0:03.0 - 0:07.5   @warm A rare day.\n")
        self.assertIn("A RARE day.", director.emit(res, "caps"))

class VcrefTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.refs = self.tmp / "refs"
        self.refs.mkdir()
        make_refs(self.refs)
        (self.refs / "rules.json").write_text(json.dumps([{"pattern": "(?i)sorry", "tag": "sad"}]), encoding="utf-8")
        self.out = vcref.pack(self.refs, self.tmp / "v.vcref", name="t", config={"engine": "voxcpm"})

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_roundtrip(self):
        b = vcref.Bundle(self.out)
        self.assertTrue(b.verify())
        self.assertEqual(sorted(b.tags), ["neutral", "question", "sad"])
        self.assertEqual((b.default, b.config), ("neutral", {"engine": "voxcpm"}))
        x, sr = b.read("sad")
        orig, _ = sf.read(self.refs / "sad.wav", dtype="float32")
        self.assertEqual(sr, 48000)
        self.assertLess(float(np.abs(x - orig).max()), 1e-6)  # FLAC 24-bit is lossless for 24-bit sources
        self.assertEqual(b.rules()[0][1], "sad")

    def test_smaller_than_wav_and_unknown_tag_falls_back(self):
        b = vcref.Bundle(self.out)
        self.assertLess(self.out.stat().st_size, sum(p.stat().st_size for p in self.refs.glob("*.wav")))
        self.assertEqual(b.resolve("nope").tag, "neutral")
        self.assertTrue(b.resolve("sad").wav.exists())

    def test_tamper_and_foreign_files_are_rejected(self):
        bad = self.tmp / "bad.vcref"
        with zipfile.ZipFile(self.out) as src, zipfile.ZipFile(bad, "w") as dst:
            dst.comment = vcref.MAGIC
            for n in src.namelist():
                data = src.read(n)
                dst.writestr(n, data[:-3] + b"xyz" if n == "clips/sad.flac" else data)
        with self.assertRaises(vcref.VcrefError):
            vcref.Bundle(bad).verify()
        plain = self.tmp / "plain.vcref"
        with zipfile.ZipFile(plain, "w") as z:
            z.writestr("manifest.json", "{}")
        with self.assertRaises(vcref.VcrefError):
            vcref.Bundle(plain)
        with self.assertRaises(vcref.VcrefError):
            vcref.Bundle(self.tmp / "missing.vcref")


if __name__ == "__main__":
    unittest.main()
