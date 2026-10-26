---
name: voiceclone
description: Run the local VoiceClone pipeline as phased steps (preflight, voice profile, enroll, config, script, generate, review). Reuses saved voice profiles, reference audio and configs. Use when the user wants to clone a voice, generate speech/narration in a saved voice, add a new speaker, or tweak engine/style/loudness settings.
---

# VoiceClone — phased workflow

Project: the VoiceClone repo root (`D:\VoiceClone` on the Windows machine; wherever it is cloned on macOS). Everything goes through `scripts/vc.py`, run with the venv's Python: `.venv\Scripts\python.exe` on Windows, `.venv/bin/python` on macOS (call it `VC` below). Paths below use Windows separators; swap `\` for `/` on macOS. `VC` records every phase in `work\skill_progress.json`, which the **voiceclone-progress** mod pane displays live.

**Reuse first.** Reference audio lives in `reference\<speaker>\` and is never copied or overwritten. A *profile* (`profiles\<name>.json`) points at a speaker and stores default engine/style/LUFS/denoise. *Presets* (`configs\presets\*.json`) are named config bundles. Never re-enroll or write into `reference\<speaker>\parts` over existing recordings.

Ask the user only for what is missing, one phase at a time. Start with `VC progress init --label "<short job label>"`. Mark each phase with `VC progress set <n> running|done|skipped|error "<detail>"` (commands below mark most automatically).

## Phase 1 — Preflight / doctor
`VC doctor` (full dependency report: venv, packages, ROCm GPU, libs, ffmpeg, weights, disk, server, tunnel, profiles; also shown in the pane's Doctor tab). `VC doctor --fix` applies safe repairs only (folders, starts server/tunnel); never installs or deletes, so relay the `fix` hint for anything else. `VC preflight` is the quick check. If the server is down, start `dashboard\app_server.py` in the background; if the tunnel is needed, follow the `always-attach-studio-link` memory.

## Phase 2 — Profile (input: which voice?)
Run `VC profile list`. Ask: reuse a profile, register an existing speaker folder, or create a new voice?
- Reuse: `VC profile show <name>`; go to Phase 3.
- Register existing speaker: `VC profile create <name> --speaker <folder> [--preset P]`.
- New voice: need a profile name and 1–3 audio files (clean, single speaker, ≥30 s total of natural speech; `docs\recording_script.md` has a script) or the user records in the Studio. `VC profile create <name> --audio a.wav [b.wav c.wav]` (converted to 48 kHz mono; refuses to overwrite).

## Phase 3 — Enroll
`VC enroll <name>`. Skips automatically if `ref.wav`/`ref.txt` exist (reuse). Use `--force` only if the user asks to re-pick the reference clip. Show the user `ref_text` (`profile show`) and ask them to confirm it matches what is said.

## Phase 4 — Config (input: engine, style, loudness)
Offer presets (`narration`, `conversational`, `ad-read`, `qwen-plain`) or custom values:
- `engine`: `voxcpm` (48 kHz, supports style prefix like `(warm, slow)`) or `qwen` (24 kHz, no style control).
- `style`, `lufs` (default -16; podcasts -16, audiobooks -18…-20), `denoise` (0–1, usually 0).
Persist with `VC profile set <name> --preset P --engine E --style S --lufs N --denoise D`. One-off overrides can go straight on `generate`.

## Phase 5 — Script (input: the text)
Get the text from the user (paste or file). Write it UTF-8 to `work\jobs\_input\<job>.txt` (create the folder). Split into paragraphs; with VoxCPM a leading `(style)` on a line overrides delivery for that line. Warn about digits/abbreviations: spell them out. Then `VC progress set 5 done "<n> words"`.

## Phase 6 — Generate
Pick a job id like `<name>_<topic>_<YYYYMMDD-HHMM>`. `VC generate <name> --text-file <file> --job <id> [--engine ..] [--style ..] [--lufs ..]`. It runs synthesis then mastering (long runs: use background execution and poll `VC progress show`; live stage detail is in `work\status.json` and `work\pipeline.log`). Output: `output\<id>.wav` (24-bit 48 kHz).

## Phase 7 — Review
Report output path, LUFS/true peak from `work\jobs\<id>\meta.json`, and the Studio link. Ask whether to adjust (→ Phase 4 with `profile set`, regenerate with a new job id) or finish. Then `VC progress set 7 done`.

On any failure: `VC progress set <n> error "<why>"`, read `work\pipeline.log`, fix, retry that phase only.
