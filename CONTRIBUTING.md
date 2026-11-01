# Contributing

Small, focused improvements to documentation, portability, tests, and supported
workflows are welcome. Open an issue before substantial method changes or long
experiment campaigns.

## Local checks

Create an environment and install the editable package as described in README.md.
For documentation, CLI, or core changes, run:

```bash
pmu-gnn --help
pmu-gnn --version
python -m unittest discover -s tests/software -v
```

Software CI checks installation, syntax, CLI behavior and synthetic unit tests.
It does not read final experiment results. For changes involving saved artifacts,
run `pmu-gnn verify` separately and report its actual outcome. The broader suite
remains available with `python -m unittest discover -s tests -v`.

Do not weaken the verifier or rewrite records to make a paper value pass.
Software tests and artifact verification have different scopes; a passing code
workflow does not establish the paper's numerical results.

## Preserve the method and evidence

- Keep the original bus-feature scales, stored connection directions, and
  multiplicities unless a scientific change is explicitly proposed and tested.
- Keep direct one-hop coverage distinct from electrical state-estimation rules.
- Keep Greedy reference budget `K` distinct from an independently learned count.
- Do not overwrite saved data, checkpoints, figures, or per-scenario results.
- Write new experiments to a fresh `runs/` or `example_runs/` directory.
- Label new results separately and record configuration, scenario IDs, software
  versions, checkpoint hashes, and timing boundaries.
- Preserve copyright, license, and third-party notices.

For a pull request, describe the problem, changed behavior, checks performed,
and any remaining limitation. Avoid unrelated formatting changes or large
generated outputs.

## Bug reports

Include the commit/version, operating system, Python/dependency versions,
command, and a minimal error or example. Remove credentials and private paths.
For vulnerabilities, follow [SECURITY.md](SECURITY.md) instead of a public report.
