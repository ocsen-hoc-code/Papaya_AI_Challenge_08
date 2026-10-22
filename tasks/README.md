# Tasks

Each task is a `Txx.md` file. Its status is the `status:` line in the frontmatter.

| Status | Meaning |
|---|---|
| `todo` | Not started |
| `pending` | In progress, or waiting (blocked, in review, or waiting for the user to confirm) |
| `done` | Finished: every acceptance criterion is met |

## Commands

```bash
python3 tasks/tasks.py list                  # all tasks
python3 tasks/tasks.py list --status pending # filter by status
python3 tasks/tasks.py next                  # todo tasks whose dependencies are done
python3 tasks/tasks.py set T03 pending       # start a task
python3 tasks/tasks.py set T03 done          # finish a task
python3 tasks/tasks.py board                 # regenerate BOARD.md
```

`BOARD.md` is generated. Do not edit it by hand; if you edit a `Txx.md` file directly, run `board` again.

## Workflow
1. Pick a task from `next`, then `set <ID> pending`.
2. When every item under **Acceptance criteria** in the task file is ticked, `set <ID> done`.
3. Record findings, decisions and blockers in the task's **Notes** section.
