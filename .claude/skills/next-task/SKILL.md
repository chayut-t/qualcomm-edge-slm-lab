---
name: next-task
description: Generate the learner's next task pack, or point back to the active task. Same as saying "Give me my next task."
---

# next-task

Rules for `Give me my next task.` (`PROJECT_INIT.md` Section 4):

1. Read `progress/state.json` and the latest run evidence in
   `progress/runs/`.
2. If a task is already `active` or `awaiting_review`, point the learner
   back to it. Do not create another task.
3. If the prior task is `task_complete`, increment `current_task`, set
   `status` to `active`, and generate only the next task pack:
   the HTML lesson (Section 6.2 structure), the needed notebook or code,
   and `progress/runs/<NN>-run.json`.
4. Update `docs/index.html` and `docs/progress.html` to match state.
5. Tell the learner exactly what to read and run, and which outputs to
   inspect, using the Section 6.4 handoff order.
6. Stop before running the learner's local lab. The learner controls
   execution.

Never generate more than one task. Never pre-create future task files.
Use results and errors from the previous task to adjust the new lesson.
