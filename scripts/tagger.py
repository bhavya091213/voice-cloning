"""Local-LLM emotion/pace tagger.

Input : plain text + a natural-language style description.
Output: list of segments [{text, emotion, intensity, pace, pause_ms}] covering the text verbatim.
Long text is split into sentence-aligned chunks; each chunk is tagged separately with the
previous chunk's tail as context so the performance stays coherent across chunk borders.
"""
import json
import re
import sys
from pathlib import Path

# The tagger needs a newer transformers (guarded torch.distributed import) than IndexTTS's pinned one.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "libs_llm"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LLM_DIR = Path(__file__).resolve().parent.parent / "models" / "qwen3-4b-instruct"
EMOTIONS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]
MAX_CHUNK_CHARS = 700

SYSTEM = f"""You are a voice director. You split a script into short performance segments and tag each one.
Return ONLY a JSON array. Each item: {{"text": str, "emotion": one of {EMOTIONS}, "intensity": 0.0-1.0, "pace": 0.8-1.25, "pause_ms": 0-1200}}.
Rules:
- The segment texts, concatenated in order, must reproduce the input EXACTLY. Never add, drop, reword or reorder words or punctuation.
- Split at sentence or clause boundaries; segments are usually 1-2 sentences.
- pace: 1.0 normal, <1 slower, >1 faster. pause_ms: silence after the segment.
- "calm" with low intensity is the neutral default. Do not over-act; follow the requested style and the meaning of the text.
- Keep tagging consistent with the previous context, changing only where the meaning changes."""


def split_chunks(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    sentences = re.findall(r"[^.!?\n]+[.!?]*[\"')\]]*\s*|\n+", text)
    chunks, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) > limit:
            chunks.append(cur)
            cur = ""
        cur += s
    if cur.strip():
        chunks.append(cur)
    return chunks


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


class Tagger:
    def __init__(self, device: str = ""):
        if not device:
            import torch as _t
            device = "cuda" if _t.cuda.is_available() else ("mps" if _t.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(LLM_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(LLM_DIR, torch_dtype=torch.bfloat16).to(device)
        self.model.eval()

    def unload(self):
        del self.model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def _ask(self, style: str, context: str, chunk: str) -> str:
        user = (f"STYLE: {style}\n\nPREVIOUS CONTEXT (do not tag): {context or '(start of script)'}\n\n"
                f"TEXT TO TAG:\n{chunk}")
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=2048, do_sample=False)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    @staticmethod
    def _parse(raw: str, chunk: str):
        m = re.search(r"\[.*\]", raw, re.S)
        if not m:
            return None
        try:
            segs = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        if not segs or _norm("".join(s["text"] for s in segs)) != _norm(chunk):
            return None  # LLM altered the words
        clean = []
        for s in segs:
            emo = s.get("emotion", "calm")
            clean.append({
                "text": s["text"].strip(),
                "emotion": emo if emo in EMOTIONS else "calm",
                "intensity": min(1.0, max(0.0, float(s.get("intensity", 0.3)))),
                "pace": min(1.25, max(0.8, float(s.get("pace", 1.0)))),
                "pause_ms": int(min(1200, max(0, s.get("pause_ms", 250)))),
            })
        return [c for c in clean if c["text"]]

    def tag(self, text: str, style: str) -> list[dict]:
        out, context = [], ""
        text = text.replace("﻿", "").replace("​", "")
        for chunk in split_chunks(text):
            segs = None
            for _ in range(3):  # retry on malformed / altered output
                segs = self._parse(self._ask(style, context, chunk), chunk)
                if segs:
                    break
            if not segs:  # safe fallback: untagged sentences, text preserved
                segs = [{"text": s.strip(), "emotion": "calm", "intensity": 0.2, "pace": 1.0, "pause_ms": 300}
                        for s in split_chunks(chunk, 200) if s.strip()]
            out.extend(segs)
            context = chunk[-300:]
        return out


if __name__ == "__main__":
    import sys
    t = Tagger()
    print(json.dumps(t.tag(Path(sys.argv[1]).read_text(encoding="utf-8"), sys.argv[2]), indent=1))
