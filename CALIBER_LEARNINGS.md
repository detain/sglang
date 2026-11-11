# Caliber Learnings

Accumulated patterns and anti-patterns from development sessions.
Auto-managed by [caliber](https://github.com/caliber-ai-org/ai-setup) — do not edit manually.

- **[preference]** When a spawned subagent session dies with an infrastructure/streaming error (not a completion), RESUME that same agent session with an audit-first recovery prompt (check committed work, tee'd logs, live processes) instead of spawning a fresh one
- **[preference]** Always spawn subagents with background=true
- **[anti-pattern]** Never land commits into /home/my/sglang/sglang's working tree while prod servers import from it via the editable venv pointer — integrate on branches with `git branch -f` ref-moves; the working tree's checked-out state is an operator-gated boot decision
