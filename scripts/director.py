"""Local-LLM voiceover director: per-line intonation tag, stress words and pause, with the whole script as context.

python director.py <script.txt> <direction_file> <out.json> [--tags neutral,warm,...] [--emit tagged.txt] [--stress MODE]

<script.txt> is either plain text (one segment per sentence) or a timed script (`m:ss.s - m:ss.s  text` lines; music cues
`[...]` pass through). In timed mode every script line stays one segment, so the slots never move.
Words are never changed: each segment must reproduce its input line exactly (checked), else the chunk is retried and finally
falls back to the punctuation rules (takes.py) with no stress.
`--emit` writes the script back with `@tag` markers (and stress markup per --stress) for scripts/timed_synth.py / vox_synth.py.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import takes  # noqa: E402

TAGS = ["neutral", "warm", "serious", "excited", "sad", "annoyed", "playful", "question"]
PAUSES = {"none": 0.1, "short": 0.2, "beat": 0.45, "long": 0.8}
TIMED = re.compile(r"^(\s*\d+:\d{1,2}(?:\.\d+)?\s*-\s*\d+:\d{1,2}(?:\.\d+)?\s+)(.+?)\s*$")
MAX_STRESS_RATIO = 1 / 7
CHUNK_LINES = 8

PROMPTS = Path(__file__).parent / "prompts"
SYSTEM = (PROMPTS / "director_system.txt").read_text(encoding="utf-8").strip()
LINE_MODE = (PROMPTS / "director_line_mode.txt").read_text(encoding="utf-8").strip()
FORMS = ["statement", "yes-no question", "wh-question", "tag question", "alternative question", "command", "fragment"]
CONTOURS = ["falls", "rises", "holds"]
WH = {"who", "whom", "whose", "what", "when", "where", "why", "how", "which"}
# first words that make a '?' sentence a yes-no question: auxiliaries/modals (incl. negated), plus the short spoken openers
# ("Want to see...?", "Ready to start?") and "you ..." declarative questions
YESNO_START = {"do", "does", "did", "is", "are", "was", "were", "am", "can", "could", "will", "would", "shall", "should", "may",
               "might", "must", "have", "has", "had", "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent", "cant",
               "couldnt", "wont", "wouldnt", "shouldnt", "havent", "hasnt", "hadnt", "aint", "want", "wanna", "ready", "anyone",
               "anybody", "anything", "any", "need", "mind", "care", "interested", "curious", "you", "ok", "okay", "right"}


def question_kind(sentence):
    """'wh' / 'yesno' / None for a sentence ending in '?'. Leading conjunctions and interjections are skipped
    ("And why...?", "So, did you...?")."""
    t = sentence.strip()
    if not re.search(r"\?[\"')\]]*$", t):
        return None
    toks = [re.sub(r"[^a-z']", "", w.lower()).replace("'", "") for w in t.split()]
    toks = [w for w in toks if w]
    while toks and toks[0] in {"and", "but", "so", "well", "oh", "then", "now", "hmm", "um", "uh"}:
        toks = toks[1:]
    if not toks:
        return None
    if re.search(r",\s*(?:do|does|did|is|are|was|were|will|would|can|could|should|have|has)(?:n't|nt)?\s+"
                 r"(?:you|i|he|she|it|we|they|there)\s*\?[\"')\]]*$", t, re.I):
        return None  # tag question ("..., don't you?"): rises when asked, falls when confirming; leave it to the model
    if toks[0] in WH or (toks[0] == "how" and len(toks) > 1 and toks[1] == "come"):
        return "wh"
    return "yesno" if toks[0] in YESNO_START else None


def words(s):
    return re.findall(r"[A-Za-z0-9']+", s)


def check(item, line, tags):
    """Normalise one model item against its input line; returns (item, fixes) or None when the words differ."""
    if not isinstance(item, dict) or re.sub(r"\s+", "", str(item.get("text", ""))) != re.sub(r"\s+", "", line):
        return None
    fixes = []
    tag = str(item.get("tag", "")).strip().lower()
    if tag not in tags:
        fixes.append(f"tag {tag!r} -> neutral")
        tag = "neutral"
    pause = str(item.get("pause", "short")).strip().lower()
    if pause not in PAUSES:
        fixes.append(f"pause {pause!r} -> short")
        pause = "short"
    form = str(item.get("form", "")).strip().lower()
    contour = str(item.get("contour", "")).strip().lower()
    kind = question_kind(line)
    if form not in FORMS:
        form = {"wh": "wh-question", "yesno": "yes-no question"}.get(kind, "statement")
        fixes.append("form derived from the text")
    if contour not in CONTOURS:
        contour = "rises" if form == "yes-no question" else "falls"
        fixes.append("contour derived from the form")
    have = {w.lower(): w for w in words(line)}
    stress = []
    for phrase in item.get("stress") or []:
        for w in words(str(phrase)):  # a stressed phrase ("step by step") is several words
            if w.lower() in have and have[w.lower()] not in stress:
                stress.append(have[w.lower()])
            elif w.lower() not in have:
                fixes.append(f"stress {w!r} not in line, dropped")
    cap = max(1, int(len(words(line)) * MAX_STRESS_RATIO))
    if len(stress) > cap:
        fixes.append(f"stress cut to {cap}")
        stress = stress[:cap]
    style = re.sub(r"[()\[\]{}\n]", " ", str(item.get("style", "")))
    return {"text": line, "function": str(item.get("function", ""))[:80], "form": form, "contour": contour,
            "stress": stress, "tag": tag,
            "style": re.sub(r"\s+,", ",", re.sub(r"\s+", " ", style)).strip().lower()[:70], "pause": pause}, fixes


def smooth_tags(items):
    """Deterministic clean-up of the model's tags (a tag is a whole voice colour, not a per-line accent).
    1. All lines of one sentence share one tag (the commonest; ties go to the first line).
    2. A single sentence whose tag differs from both equal neighbours takes theirs (genuine `question` is kept).
    Edits `items` in place; returns the list of changes."""
    from collections import Counter
    groups, cur = [], []
    for it in items:
        cur.append(it)
        if re.search(r"[.!?][\"')\]]*$", it["text"].strip()):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    log = []

    def setg(g, tag, why):
        for it in g:
            if it["tag"] != tag:
                log.append(f"line {it['n']}: tag {it['tag']} -> {tag} ({why})")
                it["tag"] = tag

    for g in groups:
        counts = Counter(it["tag"] for it in g)
        top = max(counts.values())
        setg(g, next(it["tag"] for it in g if counts[it["tag"]] == top), "one tag per sentence")
    tags = [g[0]["tag"] for g in groups]
    for k in range(1, len(groups) - 1):
        if tags[k - 1] == tags[k + 1] != tags[k] and tags[k] != "question":
            setg(groups[k], tags[k - 1], "single-sentence blip")
            tags[k] = tags[k - 1]
    return log


def _sentence_groups(items):
    groups, cur = [], []
    for it in items:
        cur.append(it)
        if re.search(r"[.?!][\"')\]]*$", it["text"].strip()):
            groups.append(cur)
            cur = []
    return groups + ([cur] if cur else [])


def enforce_questions(items, tags=TAGS):
    """Make the tag agree with the contour, whatever the model chose. Only the `question` clip ends with a rise.
      - a yes-no question (by its first words, or because the model said it rises) uses `question`
      - a sentence that does not rise never uses `question`, even if it ends in '?' (wh-questions fall)
    Works on whole sentences (a sentence split over several lines gets one tag). Edits `items`; returns the changes."""
    if "question" not in tags:
        return []
    log = []
    for g in _sentence_groups(items):
        text = " ".join(it["text"].strip() for it in g)
        kind = question_kind(text)
        rises = kind == "yesno" or (kind is None and any(it.get("contour") == "rises" for it in g))
        if kind == "wh":
            rises = False
        for it in g:
            want = "question" if rises else (it["tag"] if it["tag"] != "question" else "neutral")
            if it["tag"] != want:
                log.append(f"line {it['n']}: tag {it['tag']} -> {want} ({'rising yes-no question' if rises else 'does not rise'})")
                it["tag"] = want
            it["contour"] = "rises" if rises else ("falls" if it.get("contour") == "rises" else it.get("contour", "falls"))
    return log


def parse_script(path):
    """-> (rows, timed). rows: [{n, head (timestamp prefix or ''), text, music}]"""
    raw = Path(path).read_text(encoding="utf-8-sig").replace("﻿", "").replace("​", "")
    rows, timed = [], False
    for ln in raw.splitlines():
        m = TIMED.match(ln)
        if m:
            timed = True
            rows.append({"head": m.group(1), "text": m.group(2), "music": m.group(2).startswith("[")})
    if timed:
        return rows, True
    sents = [s for para in re.split(r"\n\s*\n", raw.strip()) for s in takes.sentences(" ".join(para.split("\n")))]
    return [{"head": "", "text": s, "music": False} for s in sents], False


def direct(script, direction, tags=TAGS, log=print):
    from tagger import Tagger  # heavy import (torch/transformers) only when actually running the model
    import torch
    rows, timed = parse_script(script)
    speak = [r for r in rows if not r["music"]]
    llm = Tagger()
    full = "\n".join(r["text"] for r in speak)
    system = SYSTEM + "\n\n" + LINE_MODE
    done, fixes, fallbacks = [], [], 0
    for c0 in range(0, len(speak), CHUNK_LINES):
        chunk = speak[c0:c0 + CHUNK_LINES]
        prev = "; ".join(f"{d['tag']}/{d['pause']}" for d in done[-4:]) or "(start of script)"
        user = (f"DIRECTION: {direction}\n\nAVAILABLE TAGS: {', '.join(tags)}\n\nFULL SCRIPT (for understanding the whole piece; tag only the lines listed below):\n{full}\n\n"
                f"TAGS ALREADY CHOSEN FOR THE PREVIOUS LINES (tag/pause, for continuity): {prev}\n\nLINES TO TAG:\n"
                + "\n".join(f"{i + 1}. {r['text']}" for i, r in enumerate(chunk)))
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        got = None
        for attempt in range(3):
            enc = llm.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(llm.model.device)
            with torch.no_grad():
                out = llm.model.generate(**enc, max_new_tokens=3000, do_sample=False)
            raw = llm.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            m = re.search(r"\[.*\]", raw, re.S)
            try:
                items = json.loads(m.group(0)) if m else None
            except json.JSONDecodeError:
                items = None
            if items and len(items) == len(chunk):
                res = [check(it, r["text"], tags) for it, r in zip(items, chunk)]
                if all(res):
                    got = res
                    break
            log(f"chunk {c0 // CHUNK_LINES + 1}: attempt {attempt + 1} rejected (count/words mismatch)")
            msgs = msgs[:2]
        if got is None:  # keep the words, let the punctuation rules choose
            fallbacks += 1
            got = [({"text": r["text"], "function": "", "stress": [], "tag": takes.tag_for(r["text"], default="neutral"),
                     "style": "", "pause": "short"}, ["fallback: punctuation rules"]) for r in chunk]
        for r, (d, fx) in zip(chunk, got):
            d["n"] = len(done) + 1
            done.append(d)
            fixes += [f"line {d['n']}: {f}" for f in fx]
        log(f"chunk {c0 // CHUNK_LINES + 1}/{-(-len(speak) // CHUNK_LINES)} done")
    fixes += smooth_tags(done)
    fixes += enforce_questions(done, tags)
    it = iter(done)
    lines =[{**r, **({} if r["music"] else next(it))} for r in rows]
    return {"direction": direction, "timed": timed, "lines": lines, "fixes": fixes, "fallback_chunks": fallbacks}


def stressed(text, stress, mode):
    """Stress markup for the synth. off: none. caps: stressed word in capitals."""
    if mode == "off" or not stress:
        return text
    for w in stress:
        text = re.sub(rf"(?<![A-Za-z0-9']){re.escape(w)}(?![A-Za-z0-9'])", w.upper() if mode == "caps" else w, text, count=1)
    return text


def emit(result, mode="off"):
    out = []
    for r in result["lines"]:
        if r["music"]:
            out.append(f"{r['head']}{r['text']}")
        else:
            out.append(f"{r['head']}@{r['tag']} {stressed(r['text'], r['stress'], mode)}")
            if not result.get("timed") and r.get("pause") in ("beat", "long"):
                out.append("")  # a blank line is a paragraph-sized pause for takes.split
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("direction_file")
    ap.add_argument("out_json")
    ap.add_argument("--tags", default=",".join(TAGS))
    ap.add_argument("--emit", help="also write the script with @tag markers here")
    ap.add_argument("--stress", default="off", choices=["off", "caps"])
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).parent))
    import rocm_shim  # noqa: F401
    res = direct(a.script, Path(a.direction_file).read_text(encoding="utf-8").strip(), a.tags.split(","))
    Path(a.out_json).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    if a.emit:
        Path(a.emit).write_text(emit(res, a.stress), encoding="utf-8")
    print(f"done: {len(res['lines'])} lines, {len(res['fixes'])} fixes, {res['fallback_chunks']} fallback chunks")
