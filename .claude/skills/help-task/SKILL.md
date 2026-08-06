---
name: help-task
description: Help the learner past the current blocker with one focused hint or the smallest fix. Same as saying "Help me with the current task."
---

# help-task

Rules for `Help me with the current task.` (`PROJECT_INIT.md` Section 4):

1. Read `progress/state.json` and the current task's files to find the
   one thing blocking the learner right now.
2. Diagnose only the current blocker. Do not audit the whole task.
3. Give one useful hint or the smallest code change first. Let the
   learner apply and run it.
4. Prefer a small fix or a focused explanation. Keep the lab runnable.
5. Do not unlock future tasks. Do not mark anything complete.
6. Point back to the relevant HTML lesson section when the blocker is a
   concept gap, not a code bug.

If the blocker is external (service outage, incompatible device, missing
access), record it clearly and suggest the documented fallback path.
