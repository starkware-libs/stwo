# Run artifacts

Tracked documentation lives in this folder tree.

Runtime artifacts are written to `.state/` and are intentionally ignored:

- `.state/upstream/`
- `.state/worktrees/`
- `.state/runs/`

Each run gets:

- `results.tsv`
- `results.jsonl`
- benchmark JSON and logs
- agent response snapshots

