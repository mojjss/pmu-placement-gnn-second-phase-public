# Repository guidance

- Read README.md and docs/REPRODUCING.md before changing the project.
- Use the documented environment. Run lightweight tests with `python -m unittest discover -s tests -v`; optional dependency tests may skip.
- Inspect saved evidence with `pmu-gnn verify`.
- Preserve scientific behavior; document any explicitly requested change.
- Never overwrite saved paper data, checkpoints, or results.
- Write new experiment outputs to a fresh, separate directory.
- Keep mean coverage distinct from full-coverage frequency; K is a reference budget.
- Report discrepancies without modifying evidence to match a claim.
- Request authorization before full training or benchmark campaigns.
- Keep credentials, private paths, and internal audit history out of public files.
- Do not commit, push, or publish unless explicitly requested.
