# voiceclone-progress

A Claude Code mod that opens a pane beside the chat tracking the VoiceClone skill: seven phases (preflight, profile, enroll, config, script, generate, review) with a progress bar, the inputs chosen, and live pipeline stages while a job runs. It polls D:/VoiceClone/work/skill_progress.json and status.json every 1.5 seconds. Use /vc-progress to reopen the pane.

## Install

From the repo root: `claude --plugin-dir mods/voiceclone-progress`, then `/vc-progress`. Paths assume the repo lives at `D:/VoiceClone`.
