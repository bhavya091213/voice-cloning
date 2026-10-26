# VoiceClone — agent notes

Local voice-cloning TTS (Windows, ROCm). See `README.md` for the pipeline.

- Use the `voiceclone` skill (`.agents/skills/voiceclone` for Codex, `.claude/skills/voiceclone` for Claude Code). Both drive `scripts/vc.py` and share `work/skill_progress.json`.
- Python: `.venv\Scripts\python.exe`. Import `scripts/rocm_shim.py` first in torch scripts.
- Never overwrite `reference/<speaker>/parts`. Profiles (`profiles/`) and presets (`configs/presets/`) are the reusable config.
- After editing `dashboard/app_server.py`, restart it. Run `vc.py doctor` to check dependencies.
