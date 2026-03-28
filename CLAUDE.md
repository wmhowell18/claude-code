# CLAUDE.md

## Workflow rules

### transformer-backgammon

- Always commit and push changes **before** running tests. The test suite requires dependencies that are only available in the CI/remote environment, not locally.
- **End-of-session documentation**: At the end of every session, make a note of all completed tasks and update the relevant docs:
  - `TODO.md` — Mark items as `[x]` done with date, add new entries for any new work
  - `NEXT_STEPS.md` — Update if training pipeline or architecture changed
  - `BEATING_XG.md` — Update "Where We Are Today" table if capabilities changed
  - `README.md` — Update architecture description if model architecture changed
  - `PROJECT_STRUCTURE.md` — Update if new files were added
