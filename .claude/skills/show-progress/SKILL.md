---
name: show-progress
description: Summarize completed tasks, the active task, evidence, blockers, and what is next. Same as saying "Show my progress."
---

# show-progress

Rules for `Show my progress.` (`PROJECT_INIT.md` Section 4):

1. Read `progress/state.json` and the run records in `progress/runs/`.
2. Summarize:
   - completed tasks (numbers and titles);
   - the active task and its status;
   - execution-evidence highlights from the run records;
   - current blockers, if any;
   - the next locked task's title and one-line goal.
3. Keep it short. Use a small table if it helps.
4. Do not generate new task content. Do not change state.
5. If `docs/progress.html` is out of sync with `progress/state.json`,
   fix the HTML page to match the JSON.
