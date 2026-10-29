# Caliber Learnings

Accumulated patterns and anti-patterns from development sessions.
Auto-managed by [caliber](https://github.com/caliber-ai-org/ai-setup) — do not edit manually.

- **[convention]** If a subagent returns a blank response, resume it with the same sessionID and send "continue" (up to 10 retries) before spawning a fresh agent or re-prompting
