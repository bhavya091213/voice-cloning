# VoiceClone

Fully local voice-cloning TTS with a mastering chain, a browser Studio UI, a live pipeline dashboard, and agent skills (Claude Code and Codex) that run the whole workflow in phases.
Built and tested on Windows 11 with an AMD RX 6900 XT (ROCm PyTorch); macOS Apple Silicon (MPS) is supported but untested. Everything runs on your own machine; no audio leaves it.

## Contents

| I want to… | Go to |
|---|---|
| Understand how it works | [Pipeline](#pipeline) · [Engines](#engines) · [Mastering](#mastering) |
| Install and set it up | [Setup](#setup) · [Windows](#windows-amd-rocm) · [macOS](#macos-apple-silicon) · [Check dependencies](#check-dependencies-doctor) |
| Generate speech now | [Quick start](#quick-start) · [CLI](#command-line) |
| Use the browser UI or phone | [Web UI](#web-ui) · [Remote access](#remote-access) |
| Reuse a voice and settings | [Profiles and presets](#profiles-and-presets) |
| Let an agent drive it | [Agent skills](#agent-skills) · [Claude mod](#claude-mod-progress-playback-doctor) |
| Find a file or script | [Repository layout](#repository-layout) |
| Fix a problem | [Known issues](#known-issues) · [Legacy / unused](#legacy--unused) |

---

## Pipeline

```
record / upload (Parts A, B, C) -> pick reference clip (Whisper) -> TTS clone -> master -> output/<job>.wav
```

1. **Capture**: record or upload up to three reference parts in the Studio (48 kHz mono float32).
2. **Reference pick** (`scripts/enroll.py`): Whisper small with word timestamps selects the best 9-20 s sentence window and writes `reference/<speaker>/ref.wav` + `ref.txt`.
3. **Synthesis**: one of the [engines](#engines).
4. **Mastering** (`scripts/master.py`): see [Mastering](#mastering).

`scripts/run.py` orchestrates one job; `scripts/status.py` feeds the dashboard; `scripts/vc.py` is the skill CLI.

### Engines

| Engine | Script | Output | Notes |
|---|---|---|---|
| **VoxCPM2** (default) | `scripts/vox_synth.py` | native 48 kHz | optional style prefix such as `(warm, slow)` |
| **Qwen3-TTS 1.7B Base** | `scripts/qwen_synth.py` | 24 kHz | in-context cloning from the reference and its transcript; no emotion or instruction control |

### Mastering

Optional denoise, de-ess, EQ, 3-band compression, bus compression (~3 dB gain reduction), LUFS normalization, lookahead limiter and true-peak ceiling (-1 dBTP). Output is 24-bit 48 kHz.

---

## Setup

Supported platforms:

| Platform | Backend | Status |
|---|---|---|
| Windows 11 + AMD GPU | ROCm PyTorch (CUDA also works) | developed and tested here |
| macOS, Apple Silicon (M1 or later) | PyTorch MPS | supported, **untested** (written without a Mac to run it on) |

All scripts find the repo from their own location (override with `VOICECLONE_ROOT`) and pick the device automatically (`cuda` > `mps` > `cpu`, override with `VOICECLONE_DEVICE`; `VOICECLONE_DTYPE` overrides the dtype, default bfloat16 on GPU).

Common to both platforms:

- Python 3.11 venv at `.venv`.
- Isolated package dirs, git-ignored, to avoid version conflicts: `libs_qwen` (qwen-tts + transformers), `libs_vox` (voxcpm), `libs_llm` (transformers 4.57.1).
- Model weights go in `models/` (git-ignored): `qwen3-tts-1.7b-base`, `qwen3-tts-tokenizer`, `voxcpm2`, `whisper`. Hugging Face cache at `models/hf_cache`.
- `ffmpeg` must be installed.
- Import `scripts/rocm_shim.py` first in any torch script. It works around Windows/ROCm gaps (stubbed `torch.distributed`, cuDNN/MIOpen disabled on Windows, soundfile-based `torchaudio.load/save`).

### Windows (AMD ROCm)

PyTorch ROCm from `https://repo.amd.com/rocm/whl/gfx103X-all/` (torch 2.9.1, torchaudio 2.9.0), then `pip install -r requirements.txt`, then the TTS packages and weights as under macOS below (use `.venv\Scripts\python.exe`).

### macOS (Apple Silicon)

```
git clone https://github.com/bhavya091213/voice-cloning.git && cd voice-cloning
bash scripts/setup_mac.sh
```

The script installs Python 3.11, ffmpeg and cloudflared with Homebrew, creates `.venv`, installs PyTorch (MPS) and `requirements.txt`, installs the two TTS packages into `libs_qwen` / `libs_vox`, downloads the weights from Hugging Face (`openbmb/VoxCPM2`, `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, `Qwen/Qwen3-TTS-Tokenizer-12Hz`, about 15 GB), then runs the doctor. Run commands with `.venv/bin/python` instead of `.venv\Scripts\python.exe`.

Mac notes: unsupported MPS ops fall back to CPU (`PYTORCH_ENABLE_MPS_FALLBACK=1`, set for you); `torch.compile` is turned off for VoxCPM2 on Mac; generation will be slower than a discrete GPU; the live dashboard shows RAM but no GPU stats.

### Check dependencies (doctor)

```
.venv\Scripts\python.exe scripts\vc.py doctor          # report (macOS: .venv/bin/python scripts/vc.py ...)
.venv\Scripts\python.exe scripts\vc.py doctor --fix    # also create folders, start server/tunnel
```

Checks the venv, packages, ROCm GPU, extra libs, ffmpeg, model weights, disk space, Studio server, tunnel, access key and each profile's enrollment. It never installs or deletes; failing checks print a fix hint. The report is written to `work/doctor.json` and shown in the [Claude mod](#claude-mod-progress-playback-doctor).

---

## Quick start

1. Start the Studio: `.venv\Scripts\python.exe dashboard\app_server.py`, then open `http://127.0.0.1:8787/?key=<token>` (token in `dashboard/token.txt`, generated on first run).
2. Record or upload your reference parts and enroll.
3. Paste text, choose an engine and style, generate, and play the result.

Or let an agent do it: ask Claude Code or Codex to use the [`voiceclone` skill](#agent-skills).

### Command line

```
.venv\Scripts\python.exe scripts\run.py --speaker me --text-file text.txt --job job_test --engine voxcpm --style "warm"
```

Options: `--engine qwen|voxcpm`, `--style`, `--denoise 0-1`, `--lufs -16`. Output lands in `output\<job>.wav`.

---

## Web UI

`dashboard/app_server.py` serves, on `127.0.0.1:8787`:

| Route | Page |
|---|---|
| `/` | Studio: record, enroll, generate, play results |
| `/dashboard` | Live GPU/RAM/stage/log view |
| `/architecture` | Pipeline Mermaid diagram |
| `/models` | Internals of Qwen3-TTS and VoxCPM2 |

Access is protected by a key in `dashboard/token.txt` (auto-generated on first run, git-ignored). After editing `app_server.py`, restart it.

### Remote access

Use a Cloudflare quick tunnel to reach the Studio from a phone:

```
# macOS: brew install cloudflared
cloudflared tunnel --url http://127.0.0.1:8787 --no-autoupdate --protocol http2
```

Open `https://<tunnel>.trycloudflare.com/?key=<token>`. The URL changes every time the tunnel restarts.

---

## Profiles and presets

Reference audio stays in `reference/<speaker>/` and is never copied or overwritten. Two small files make a voice and its settings reusable:

| What | Where | Holds |
|---|---|---|
| **Profile** | `profiles/<name>.json` (git-ignored; see `profiles/example.json`) | which speaker folder, plus default engine, style, LUFS, denoise |
| **Preset** | `configs/presets/*.json` | named setting bundles: `narration`, `conversational`, `ad-read`, `qwen-plain` |

```
scripts\vc.py profile list
scripts\vc.py profile create <name> --speaker <folder> --preset narration
scripts\vc.py profile create <name> --audio a.wav [b.wav c.wav]     # new voice, converted to 48 kHz mono
scripts\vc.py profile set <name> --engine voxcpm --style "(warm)" --lufs -18
scripts\vc.py enroll <name>                                          # skipped if already enrolled; --force to redo
scripts\vc.py generate <name> --text-file t.txt --job <id>           # uses the profile's defaults
```

Run these with `.venv\Scripts\python.exe`. A fresh clone needs your own audio: `reference/` is not in git. A script to read while recording is in [`docs/recording_script.md`](docs/recording_script.md).

---

## Agent skills

The same seven-phase workflow, for two agents. Both drive `scripts/vc.py` and share `work/skill_progress.json`, so a run started in one shows in the other.

| Agent | Skill location |
|---|---|
| Claude Code | [`.claude/skills/voiceclone`](.claude/skills/voiceclone/SKILL.md) |
| Codex | [`.agents/skills/voiceclone`](.agents/skills/voiceclone/SKILL.md) (plus [`AGENTS.md`](AGENTS.md)) |

| # | Phase | What happens |
|---|---|---|
| 1 | Preflight / doctor | dependency report, optional safe fixes |
| 2 | Profile | reuse a profile, register an existing speaker, or create a new voice from audio |
| 3 | Enroll | pick the reference clip; confirm its transcript |
| 4 | Config | choose a preset or engine, style, LUFS, denoise |
| 5 | Script | provide the text |
| 6 | Generate | synthesize and master |
| 7 | Review | check loudness and peak, adjust and regenerate |

Ask for it in plain words ("clone my voice reading this text"), or type `/voiceclone` in Claude Code. The skill asks only for what is missing and reuses existing profiles and enrollments.

### Claude mod: progress, playback, doctor

[`mods/voiceclone-progress`](mods/voiceclone-progress/README.md) adds a pane to Claude Code (`/vc-progress`) with three tabs:

- **Progress**: phase checklist, progress bar and live pipeline stages.
- **Outputs**: newest files from `output/` with LUFS and true peak; click to play in your default audio player; copy the Studio link.
- **Doctor**: run the dependency checks or run and fix, with a fix hint for each failure.

Install: `claude --plugin-dir mods/voiceclone-progress` from the repo root.

---

## Repository layout

```
scripts/     pipeline code (synth, enroll, master, run, status, vc, paths, rocm_shim, setup_mac.sh)
dashboard/   web server and HTML pages
configs/     reusable presets
profiles/    your voice profiles (git-ignored except example.json)
mods/        Claude Code mod (voiceclone-progress)
.claude/     Claude Code skill
.agents/     Codex skill
docs/        notes, recording script
requirements.txt   shared Python dependencies (torch is installed separately)
reference/   per-speaker audio (git-ignored)
models/      weights (git-ignored)
work/        intermediates, job output, progress and doctor reports (git-ignored)
output/      final masters (git-ignored)
```

---

## Known issues

- Mastering dulls 6-10 kHz by several dB (de-esser and high-band multiband); needs easing.
- Qwen3-TTS Base has no emotion/instruction control.
- macOS support is untested; expect rough edges (MPS op fallbacks, speed). Please report failures from `vc.py doctor` and the pipeline log.

## Legacy / unused

`scripts/tagger.py`, `tag_cli.py` (Qwen3-4B emotion tagger) and `scripts/synth.py` (IndexTTS-2.5) are kept but not part of the live pipeline.
