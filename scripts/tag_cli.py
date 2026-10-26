"""python tag_cli.py <text_file> <direction_file> <out_json>   (VoxCPM2 tagged script)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rocm_shim  # noqa: F401,E402
import tagger_vox  # noqa: E402

text = Path(sys.argv[1]).read_text(encoding="utf-8-sig")
direction = Path(sys.argv[2]).read_text(encoding="utf-8").strip()
tagged, fb = tagger_vox.VoxTagger().direct(text, direction)
Path(sys.argv[3]).write_text(json.dumps({"tagged": tagged, "fallback_chunks": fb}), encoding="utf-8")
print("done")
