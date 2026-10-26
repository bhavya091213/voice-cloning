# VoiceClone

Fully local voice-cloning TTS with a mastering chain, a browser Studio UI, and a live pipeline dashboard.
Built and tested on Windows 11 with an AMD RX 6900 XT (ROCm PyTorch). Everything runs on your own machine; no audio leaves it.

## Pipeline

```
record / upload (Parts A, B, C) -> pick reference clip (Whisper) -> TTS clone -> master -> output/<job>.wav
```

1. **Capture**: record or upload up to three reference parts in the Studio (48 kHz mono float32).
2. **Reference pick** (`scripts/enroll.py`): Whisper small with word timestamps selects the best 9-20 s sentence window and writes `reference/<speaker>/ref.wav` + `ref.txt`.
3. **Synthesis**: pick an engine.
   - **VoxCPM2** (default, `scripts/vox_synth.py`): native 48 kHz, optional style prefix such as `(warm, slow)`.
   - **Qwen3-TTS 1.7B Base** (`scripts/qwen_synth.py`): 24 kHz, in-context cloning from the reference and its transcript.
4. **Mastering** (`scripts/master.py`): optional denoise, de-ess, EQ, 3-band compression, bus compression (~3 dB gain reduction), LUFS normalization, lookahead limiter and true-peak ceiling (-1 dBTP). Output is 24-bit 48 kHz.

`scripts/vc.py` is the skill CLI (profiles, doctor, phase tracking); `scripts/run.py` orchestrates one job; `scripts/status.py` feeds the dashboard.

## Web UI

`dashboard/app_server.py` serves, on `127.0.0.1:8787`:

| Route | Page |
|---|---|
| `/` | Studio: record, enroll, generate, play results |
| `/dashboard` | Live GPU/RAM/stage/log view |
| `/architecture` | Pipeline Mermaid diagram |
| `/models` | Internals of Qwen3-TTS and VoxCPM2 |

Access is protected by a key in `dashboard/token.txt` (auto-generated on first run, git-ignored). Open `http://127.0.0.1:8787/?key=<token>`.
To reach it remotely use a Cloudflare quick tunnel:

```
cloudflared tunnel --url http://127.0.0.1:8787 --no-autoupdate --protocol http2
```

## Setup

- Python 3.11 venv at `.venv`, PyTorch ROCm from `https://repo.amd.com/rocm/whl/gfx103X-all/` (torch 2.9.1, torchaudio 2.9.0).
- Isolated package dirs, git-ignored, to avoid version conflicts: `libs_qwen` (qwen-tts), `libs_vox` (voxcpm), `libs_llm` (transformers 4.57.1).
- Model weights go in `models/` (git-ignored): `qwen3-tts-1.7b-base`, `qwen3-tts-tokenizer`, `voxcpm2`, `whisper`. Hugging Face cache at `models/hf_cache`.
- `ffmpeg` must be installed.
- Paths are currently hard-coded to `D:\VoiceClone` in several scripts.
- Import `scripts/rocm_shim.py` first in any torch script. It works around ROCm-on-Windows gaps (stubbed `torch.distributed`, cuDNN/MIOpen disabled, soundfile-based `torchaudio.load/save`).

## Run

```
.venv\Scripts\python.exe dashboard\app_server.py
```

Or from the CLI:

```
.venv\Scripts\python.exe scripts\run.py --speaker me --text-file text.txt --job job_test --engine voxcpm --style "warm"
```

## Layout

```
scripts/     pipeline code (synth, enroll, master, run, status, rocm_shim)
dashboard/   web server and HTML pages
docs/        notes, recording script
reference/   per-speaker audio (git-ignored)
models/      weights (git-ignored)
work/        intermediates and job output (git-ignored)
output/      final masters (git-ignored)
```

## Legacy / unused

`scripts/tagger.py`, `tag_cli.py` (Qwen3-4B emotion tagger) and `scripts/synth.py` (IndexTTS-2.5) are kept but not part of the live pipeline.

## Known issues

- Mastering dulls 6-10 kHz by several dB (de-esser and high-band multiband); needs easing.
- Qwen3-TTS Base has no emotion/instruction control.

## Agent skills and Claude mod

- `profiles/<name>.json` + `configs/presets/*.json`: reusable voices and settings (reference audio stays in `reference/`).
- `.claude/skills/voiceclone` (Claude Code) and `.agents/skills/voiceclone` (Codex): the same phased workflow over `scripts/vc.py`.
- `python scripts/vc.py doctor [--fix]`: dependency check, writes `work/doctor.json`.
- The `voiceclone-progress` Claude mod (`mods/voiceclone-progress`, `/vc-progress`) shows phases, output playback and the doctor.
