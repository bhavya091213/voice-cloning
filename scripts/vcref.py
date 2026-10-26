"""vcref: a one-file voice reference bundle (.vcref).

A bundle holds every reference clip of one voice (one per intonation), each with its transcript, plus the default
settings and the regex rules that choose a clip for a piece of text. Code reads it through `Bundle`; nothing else
needs to know the layout.

Container: a ZIP archive (stdlib only) whose archive comment is b"VCREF1".
    manifest.json        format/version, name, default tag, clips{tag: file, text, seconds, label, source, sha256},
                         rules[{pattern, tag}], config{engine, style, lufs, denoise, cfg}
    clips/<tag>.flac     24-bit lossless clip (about half the size of WAV)
    parts/<name>.flac    optional: the full enrollment recordings (pack --parts)

CLI:  python vcref.py pack <refs_dir> -o voice.vcref [--parts <parts_dir>] | info F | verify F | unpack F <dir>
"""
import argparse
import hashlib
import io
import json
import re
import sys
import time
import zipfile
from collections import namedtuple
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import paths  # noqa: E402

MAGIC = b"VCREF1"
EXT = ".vcref"
VERSION = 1
Clip = namedtuple("Clip", "tag wav txt text")


class VcrefError(Exception):
    pass


def _flac(x, sr):
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="FLAC", subtype="PCM_24")
    return buf.getvalue()


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def pack(refs_dir, out, name=None, config=None, rules=None, parts_dir=None):
    """refs_dir holds <tag>.wav, <tag>.txt and index.json ({"clips": {tag: {text, seconds, label, source}}})."""
    refs_dir, out = Path(refs_dir), Path(out)
    idx = json.loads((refs_dir / "index.json").read_text(encoding="utf-8"))
    if not idx.get("clips"):
        raise VcrefError("index.json has no clips")
    rules_file = refs_dir / "rules.json"
    if rules is None and rules_file.exists():
        rules = json.loads(rules_file.read_text(encoding="utf-8"))
    clips, sr_seen = {}, set()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        z.comment = MAGIC
        for tag, meta in idx["clips"].items():
            x, sr = sf.read(refs_dir / f"{tag}.wav", dtype="float32")
            data = _flac(x, sr)
            z.writestr(f"clips/{tag}.flac", data)
            sr_seen.add(sr)
            clips[tag] = {**meta, "file": f"clips/{tag}.flac", "sha256": _sha(data), "sample_rate": sr}
        parts = {}
        for p in sorted(Path(parts_dir).glob("*.wav")) if parts_dir else []:
            x, sr = sf.read(p, dtype="float32")
            data = _flac(x, sr)
            z.writestr(f"parts/{p.stem}.flac", data)
            parts[p.stem] = {"file": f"parts/{p.stem}.flac", "sha256": _sha(data), "seconds": round(len(x) / sr, 1)}
        manifest = {"format": "vcref", "version": VERSION, "name": name or out.stem, "created": time.strftime("%Y-%m-%d"),
                    "default": idx.get("default") or ("neutral" if "neutral" in clips else next(iter(clips))),
                    "clips": clips, "rules": rules or [], "config": config or {}, "parts": parts}
        z.writestr("manifest.json", json.dumps(manifest, indent=1, ensure_ascii=False))
    tmp.replace(out)
    return out


class Bundle:
    def __init__(self, path):
        self.path = Path(path)
        try:
            self._z = zipfile.ZipFile(self.path)
        except (zipfile.BadZipFile, FileNotFoundError) as e:
            raise VcrefError(f"{path}: not a readable vcref ({e})")
        if self._z.comment != MAGIC:
            raise VcrefError(f"{path}: not a vcref file (bad magic)")
        m = json.loads(self._z.read("manifest.json"))
        if m.get("format") != "vcref" or m.get("version", 0) > VERSION:
            raise VcrefError(f"{path}: unsupported vcref version {m.get('version')}")
        self.manifest = m

    name = property(lambda s: s.manifest["name"])
    default = property(lambda s: s.manifest["default"])
    config = property(lambda s: s.manifest.get("config", {}))
    tags = property(lambda s: list(s.manifest["clips"]))

    @property
    def id(self):
        return _sha("".join(sorted(c["sha256"] for c in self.manifest["clips"].values())).encode())[:12]

    def rules(self):
        """[(compiled regex, tag)] from the bundle, to be tried before the built-in punctuation rules."""
        return [(re.compile(r["pattern"]), r["tag"]) for r in self.manifest.get("rules", [])]

    def verify(self):
        for tag, c in self.manifest["clips"].items():
            if _sha(self._z.read(c["file"])) != c["sha256"]:
                raise VcrefError(f"{self.path.name}: clip '{tag}' is corrupt (hash mismatch)")
        for n, p in self.manifest.get("parts", {}).items():
            if _sha(self._z.read(p["file"])) != p["sha256"]:
                raise VcrefError(f"{self.path.name}: part '{n}' is corrupt (hash mismatch)")
        return True

    def read(self, tag):
        x, sr = sf.read(io.BytesIO(self._z.read(self.manifest["clips"][tag]["file"])), dtype="float32")
        return x, sr

    def resolve(self, tag, stack=False):
        """Clip for `tag` as files the TTS can read (decoded once into work/cache/vcref/<id>/); unknown tag -> default.
        stack=True: the default (neutral narration) clip first, then the tag's clip, as one prompt with both transcripts.
        The voice stays anchored on the long neutral recording while the last clip (what the model continues from)
        carries the delivery."""
        if tag not in self.manifest["clips"]:
            tag = self.default
        if stack and tag != self.default:
            d = paths.WORK / "cache" / "vcref" / self.id
            wav, txt = d / f"stack_{tag}.wav", d / f"stack_{tag}.txt"
            if not (wav.exists() and txt.exists()):
                a, b = self.resolve(self.default), self.resolve(tag)
                xa, sr = sf.read(a.wav, dtype="float32")
                xb, sr_b = sf.read(b.wav, dtype="float32")
                assert sr == sr_b
                sf.write(wav, np.concatenate([xa, np.zeros(int(0.4 * sr), np.float32), xb]), sr, subtype="PCM_24")
                txt.write_text(a.text + " " + b.text, encoding="utf-8")
            return Clip(tag, wav, txt, txt.read_text(encoding="utf-8"))
        d = paths.WORK / "cache" / "vcref" / self.id
        wav, txt = d / f"{tag}.wav", d / f"{tag}.txt"
        if not (wav.exists() and txt.exists()):
            d.mkdir(parents=True, exist_ok=True)
            x, sr = self.read(tag)
            sf.write(wav, x, sr, subtype="PCM_24")
            txt.write_text(self.manifest["clips"][tag]["text"], encoding="utf-8")
        return Clip(tag, wav, txt, self.manifest["clips"][tag]["text"])

    def unpack(self, out_dir):
        out_dir = Path(out_dir)
        for tag in self.tags:
            c = self.resolve(tag)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{tag}.wav").write_bytes(c.wav.read_bytes())
            (out_dir / f"{tag}.txt").write_text(c.text, encoding="utf-8")
        return out_dir


def _cli():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack"); p.add_argument("refs_dir"); p.add_argument("-o", "--out", required=True)
    p.add_argument("--parts"); p.add_argument("--name")
    for n in ("info", "verify"):
        sub.add_parser(n).add_argument("file")
    u = sub.add_parser("unpack"); u.add_argument("file"); u.add_argument("dir")
    a = ap.parse_args()
    if a.cmd == "pack":
        out = pack(a.refs_dir, a.out, name=a.name, parts_dir=a.parts)
        print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
        return
    b = Bundle(a.file)
    if a.cmd == "info":
        for t, c in b.manifest["clips"].items():
            print(f"{t:10} {c['seconds']:5.1f}s  {c.get('label', ''):28} {c['text'][:60]}")
        print(f"default={b.default} rules={len(b.manifest['rules'])} parts={list(b.manifest['parts'])} id={b.id}")
    elif a.cmd == "verify":
        print("ok" if b.verify() else "BAD")
    else:
        print(b.unpack(a.dir))


if __name__ == "__main__":
    try:
        _cli()
    except VcrefError as e:
        raise SystemExit(f"vcref: {e}")
