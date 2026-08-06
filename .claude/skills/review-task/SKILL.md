---
name: review-task
description: Review the learner's current task against its completion gate. Same as saying "Review my current task."
---

# review-task

Rules for `Review my current task.` (`PROJECT_INIT.md` Section 4):

1. Set `status` to `awaiting_review` in `progress/state.json`.
2. Inspect the current task's executed notebook, code, tests, and
   generated outputs. Look at real execution evidence: executed cells,
   test results, output files, measurements.
3. Give focused hints for incomplete or incorrect work. Help fix problems.
4. Mark the task complete only when its gate (in the task's HTML lesson)
   is satisfied:
   - the learner ran the important cells or commands;
   - required automated checks pass, or a real external blocker is
     recorded;
   - expected artifacts or measurements exist;
   - any required parameter-change experiment was run.
5. On pass: update `progress/runs/<NN>-run.json` from evidence, add the
   task number to `completed_tasks`, set `status` to `task_complete`,
   keep `current_task` unchanged, refresh `docs/progress.html` and
   `docs/index.html`.
6. Do not generate the next task until the learner asks for it.

A file existing is not evidence. Do not require essays, predictions,
observations, or reflections. Never fake a passing check.
