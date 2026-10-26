"""Split a script into takes, each tagged with the reference clip (intonation) it should be spoken with.

Pure Python + regex, no dependencies. Tag choice, first match wins:
  1. an explicit marker anywhere in a line:   `@sad I really thought it would be different this time.`
     It holds until the next marker or the end of the line: `Fine. @playful Oh, you think so? @serious Listen.`
  2. the voice bundle's own rules (regex -> tag), then the built-in punctuation rules:
        sentence ends with `?` -> question,   ends with `!` -> excited
  3. the default tag (neutral).
A tag the voice does not have falls back to the default and is reported in `missing`.
Consecutive sentences with the same tag are merged into one take (up to max_chars).
"""
import re
from collections import namedtuple

Take = namedtuple("Take", "tag text gap")  # gap = seconds of silence after the take

SENT = re.compile(r"[^.!?]+[.!?]+[\"')\]]*|[^.!?]+$")
MARK = re.compile(r"(?:^|\s)@([a-z][a-z0-9_-]{0,23})(?=\s|$)")
BUILTIN = [(re.compile(r"\?[\"')\]]*\s*$"), "question"), (re.compile(r"![\"')\]]*\s*$"), "excited")]
GAP_SENT, GAP_PARA = 0.18, 0.55


def sentences(line):
    return [s.strip() for s in SENT.findall(line.strip()) if s.strip()]


def tag_for(sentence, rules=(), default="neutral"):
    for rx, tag in list(rules) + BUILTIN:
        if rx.search(sentence):
            return tag
    return default


def split(text, rules=(), available=None, default="neutral", max_chars=280, missing=None):
    """-> [Take]. `available` (set of tags the voice has) enables fallback to `default`; unknown tags go in `missing`."""
    out = []
    for para in re.split(r"\n\s*\n", text.replace("\r", "").strip()):
        first = len(out)
        for line in para.split("\n"):
            line = line.strip()
            if not line:
                continue
            pieces = MARK.split(line)  # [text, tag, text, tag, text ...]: an @tag holds until the next @tag or the line end
            runs = [(None, pieces[0])] + list(zip(pieces[1::2], pieces[2::2]))
            for forced, s in ((f, s) for f, run in runs for s in sentences(run)):
                tag = forced or tag_for(s, rules, default)
                if available is not None and tag not in available:
                    if missing is not None and tag not in missing:
                        missing.append(tag)
                    tag = default
                prev = out[-1] if len(out) > first else None  # merge only within a paragraph
                if prev is not None and prev.tag == tag and len(prev.text) + len(s) + 1 <= max_chars:
                    out[-1] = Take(tag, f"{prev.text} {s}", GAP_SENT)
                else:
                    out.append(Take(tag, s, GAP_SENT))
        if len(out) > first:
            out[-1] = out[-1]._replace(gap=GAP_PARA)
    if out:
        out[-1] = out[-1]._replace(gap=0.0)
    return out
