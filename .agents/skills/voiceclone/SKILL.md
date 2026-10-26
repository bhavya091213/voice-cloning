---
name: voiceclone
description: Run the local VoiceClone pipeline as phased steps (preflight/doctor, voice profile, enroll, config, script, generate, review). Reuses saved voice profiles, reference audio and configs. Use when asked to clone a voice, generate speech or narration in a saved voice, add a speaker, or tune engine/style/loudness.
---

# VoiceClone — phased workflow (Codex)

Repo root: where this repo is cloned (`D:\VoiceClone` on the Windows machine). All steps go through `scripts/vc.py`, run with the venv's Python: `.venv\Scripts\python.exe` on Windows, `.venv/bin/python` on macOS (call it `VC`). Paths below use Windows separators; swap `\` for `/` on macOS. `VC` records phase state in `work\skill_progress.json` (shared with the Claude skill and its progress pane), so both agents see the same run.

**Reuse first.** Reference audio lives in `reference\<speaker>\` and is never copied or overwritten. A *profile* (`profiles\<name>.json`) points at a speaker and stores default engine/style/LUFS/denoise. *Presets* are in `configs\presets\*.json`. Never re-enroll over, or write into, existing `reference\<speaker>\parts`.

Ask the user only for what is missing, one phase at a time. Begin with `VC scripts\vc.py progress init --label "<job label>"`. All commands below are `.venv\Scripts\python.exe scripts\vc.py ...`.

1. **Preflight / doctor** — `doctor` (full dependency report: venv, packages, ROCm GPU, libs, ffmpeg, weights, disk, Studio server, tunnel, profiles; writes `work\doctor.json`). Use `doctor --fix` to apply safe repairs (create folders, start server/tunnel). It never installs or deletes; report the `fix` hint for anything else. `preflight` is the quick version.
2. **Profile** — `profile list`; ask: reuse a profile, register an existing speaker folder (`profile create <name> --speaker <folder> [--preset P]`), or create a new voice (`profile create <name> --audio a.wav [b.wav c.wav]`, clean single-speaker speech, ≥30 s total; script in `docs\recording_script.md`).
3. **Enroll** — `enroll <name>` (auto-skips if `ref.wav`/`ref.txt` exist; `--force` only if the user asks). Show `profile show <name>` `ref_text` and ask the user to confirm it.
4. **Config** — presets `narration`, `conversational`, `ad-read`, `qwen-plain`, or custom: `engine` (`voxcpm` 48 kHz with `(style)` prefix, or `qwen` 24 kHz, no style), `style`, `lufs` (default -16; audiobooks -18…-20), `denoise` (0–1). Save with `profile set <name> --preset P --engine E --style S --lufs N --denoise D`.
5. **Script** — get the text, write UTF-8 to `work\jobs\_input\<job>.txt`; spell out digits and abbreviations. Then `progress set 5 done "<n> words"`.
6. **Generate** — job id like `<name>_<topic>_<YYYYMMDD-HHMM>`; `generate <name> --text-file <file> --job <id> [--engine ..] [--style ..] [--lufs ..]`. Long runs: poll `progress show`, `work\status.json`, `work\pipeline.log`. Output: `output\<id>.wav` (24-bit 48 kHz).
7. **Review** — report the output path, LUFS/true peak from `work\jobs\<id>\meta.json`; offer adjustments (back to step 4, regenerate with a new job id). Then `progress set 7 done`.

On failure: `progress set <n> error "<why>"`, read `work\pipeline.log`, fix, retry only that phase.
