# VoiceClone — agent notes

Local voice-cloning TTS (Windows ROCm; macOS Apple Silicon supported but untested). See `README.md` for the pipeline.

- Use the `voiceclone` skill (`.agents/skills/voiceclone` for Codex, `.claude/skills/voiceclone` for Claude Code). Both drive `scripts/vc.py` and share `work/skill_progress.json`.
- Python: the venv (`.venv\Scripts\python.exe`, `.venv/bin/python` on macOS). Import `scripts/rocm_shim.py` first in torch scripts. Paths come from `scripts/paths.py`, never hard-coded.
- Never overwrite `reference/<speaker>/parts`. Profiles (`profiles/`), presets (`configs/presets/`) and packed voices (`voices/*.vcref`) are the reusable config. They hold personal voice data and are git-ignored (except `profiles/example.json`).
- Layout: `scripts/` live pipeline, `scripts/legacy/` unused (IndexTTS, old tagger CLI), `scripts/tools/` helpers, `dashboard/` Studio (`app_server.py` + `app.html`) and live stats (`stats.py`), `tests/` unit tests (`python -m unittest discover -s tests`).
- After editing `dashboard/app_server.py` or `stats.py`, restart the server. Run `vc.py doctor` to check dependencies.
- Do not move or delete anything in `output/` or `work/jobs/` unless the user asks; the Studio library only filters.
