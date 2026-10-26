"""Local-LLM director for VoxCPM2.

Input : plain script text + a natural-language direction ("quiet, emotional documentary narrator").
Output: tagged script, one segment per line:  (delivery description)verbatim text
VoxCPM2 reads the leading parenthetical as a delivery instruction, so the LLM is told exactly what that
instruction may contain. Words are never changed: the segment texts must reproduce the input.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/ (tagger)
from tagger import Tagger, split_chunks, _norm  # noqa: E402

SYSTEM = """You are a voice director for VoxCPM2, a voice-cloning text-to-speech model.
You split a script into short segments and write a delivery instruction for each one.

HOW VoxCPM2 USES THE INSTRUCTION
- The instruction is a short plain-English phrase placed in parentheses before the segment text. The model performs it; it is never spoken.
- The voice itself is cloned from a reference recording. So describe DELIVERY ONLY: emotion, energy, pace, volume, tone, breathiness, pauses.
- NEVER describe identity or timbre (gender, age, accent, "deep voice", "young woman"). That would fight the cloned voice.
- Good: "quiet, sad, slow", "warm and gentle, unhurried", "tense, whispering, faster", "restrained grief, voice almost breaking", "calm, matter-of-fact".
- Use 2-6 short descriptors, lowercase, comma-separated, max 70 characters. No parentheses or brackets inside. No full sentences. No words from the script.
- Do not over-act. Keep the requested DIRECTION as the baseline and shift only where the meaning of the text changes (hope, shock, loss, relief...).
- Neighbouring segments should change gradually, not jump between extremes.

OUTPUT
Return ONLY a JSON array. Each item: {"style": str, "text": str, "break": bool}
- "text": the segment, usually 1-2 sentences, split at sentence or clause boundaries.
- Concatenated in order, all "text" values must reproduce the input EXACTLY. Never add, drop, reword, reorder or re-punctuate words.
- "break": true ONLY where the input text itself has a paragraph break (blank line or line break) after this segment, otherwise false."""


def clean_style(s: str) -> str:
    s = re.sub(r"[()\[\]{}\n]", " ", str(s))
    return re.sub(r"\s+", " ", s).strip().lower()[:70]


class VoxTagger(Tagger):
    def _ask(self, style, context, chunk):
        user = (f"DIRECTION: {style}\n\nPREVIOUS CONTEXT (do not tag): {context or '(start of script)'}\n\n"
                f"TEXT TO TAG:\n{chunk}")
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True).to(self.model.device)
        import torch
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=2048, do_sample=False)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    @staticmethod
    def _parse(raw, chunk):
        m = re.search(r"\[.*\]", raw, re.S)
        if not m:
            return None
        try:
            segs = json.loads(m.group(0))
            if not segs or _norm("".join(s["text"] for s in segs)) != _norm(chunk):
                return None
            return [{"style": clean_style(s.get("style", "")), "text": s["text"].strip(), "break": bool(s.get("break"))}
                    for s in segs if s["text"].strip()]
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            return None

    def direct(self, text, direction):
        """Returns (tagged_text, fell_back_chunks)."""
        text = text.replace("\ufeff", "").replace("\u200b", "")
        lines, context, fallbacks = [], "", 0
        for chunk in split_chunks(text):
            segs = None
            for _ in range(3):
                segs = self._parse(self._ask(direction, context, chunk), chunk)
                if segs:
                    break
            if not segs:  # keep the words, apply the plain direction
                fallbacks += 1
                segs = [{"style": clean_style(direction), "text": s.strip(), "break": False}
                        for s in split_chunks(chunk, 200) if s.strip()]
            for s in segs:
                lines.append(f"({s['style']}){s['text']}" if s["style"] else s["text"])
                if s["break"]:
                    lines.append("")
            context = chunk[-300:]
        return "\n".join(lines).strip() + "\n", fallbacks
