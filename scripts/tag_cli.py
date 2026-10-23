"""python tag_cli.py <text_file> <style> <out_json>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import tagger  # noqa: E402

text = Path(sys.argv[1]).read_text(encoding="utf-8-sig")
segs = tagger.Tagger().tag(text, sys.argv[2])
Path(sys.argv[3]).write_text(json.dumps(segs, indent=1), encoding="utf-8")
print(f"tagged {len(segs)} segments")
