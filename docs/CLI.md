# Command-line reference

Install the checkout with `python -m pip install -e .`, then run `pmu-gnn`.
Alternatively, use `python scripts/pmu.py` from the checkout without installing
the package. Scientific commands still need their optional dependencies.

```bash
pmu-gnn --help
pmu-gnn --version
pmu-gnn train --help
pmu-gnn --root /path/to/checkout saved --example IEEE14/line_0_0
```

`--root` is a global option and precedes the subcommand. Commands using
`--system` accept `14`, `57`, `118`, or `300`; saved example IDs instead use
the form `IEEE14/line_0_0`.

## Software checks versus result verification

`--help` and `--version` do not read datasets, checkpoints or saved results.
Software CI uses these interfaces and the synthetic-input suite:

```bash
python -m unittest discover -s tests/software -v
```

Commands such as `saved`, `verify`, `tables` and `figures` intentionally access
saved artifacts only when requested. Their checks and error statuses are
unchanged. `verify` is the separate command for checking packaged evidence;
Software CI does not run it or claim that its checks passed.

## Commands

| Command | Purpose | Dependency / execution note |
| --- | --- | --- |
| `saved` | Read a documented saved example; without an example, run verification | Standard library |
| `verify` | Check saved hashes, identities, memberships, and coverage arithmetic | Standard library; missing/changed evidence is an error |
| `tables` | Export Table II panels (a)–(d) from verified saved records | Standard library; verification must succeed first |
| `figures` | Copy Fig. 1/3 PNG previews or replot Fig. 2 history | Fig. 2 requires Matplotlib and Times New Roman |
| `split` | Recompute grouped memberships without training | Install graph/ML extras; writes fresh outputs |
| `build-dataset` | Generate new intact/N-1 graphs and Greedy labels | Install graph extras; explicit experiment flag |
| `greedy` | Compute a Greedy reference for an existing sample | Install graph extras; explicit experiment flag |
| `train` | Train and select a validation checkpoint | Install ML extras; explicit experiment flag |
| `predict` | Run a new prediction for one existing scenario | Install ML extras; explicit experiment flag |
| `evaluate` | Evaluate a checkpoint on fixed held-out scenarios | Install ML extras; explicit experiment flag |
| `transfer` | Evaluate a source-system checkpoint on target tests | Install ML extras; explicit experiment flag |
| `benchmark` | Measure original Greedy/GNN placement runtime | Install graph/ML extras; explicit checkpoint and experiment flag |
| `ablation` | Compare raw Top-K and coverage-aware selection | Install ML extras; explicit checkpoint and experiment flag |

All scientific commands except `split` require `--confirm-new-experiment`.
This flag authorizes that command's new computation; it does not authenticate
the output as an original paper result. Existing output directories are refused.

## Saved inspection and exports

```bash
pmu-gnn saved --example IEEE14/line_0_0
pmu-gnn verify --output runs/verify-01
pmu-gnn tables --output runs/tables-01
pmu-gnn figures --figure 3 --output runs/figure3-01
```

If verification fails, inspect the named path or discrepancy. Do not delete
manifest entries, change expected hashes, or edit coverage figures to hide it.
The current missing IEEE300 record and changed summary are described in
[RESULTS.md](RESULTS.md). Table export will not complete while verification fails.

## New experiment example

```bash
pmu-gnn greedy --system 14 --sample-id line_0_0 --output example_runs/greedy-01 --confirm-new-experiment
```

The command writes `greedy.json` in the fresh output directory. For new training,
evaluation, transfer, and dataset commands, follow [REPRODUCING.md](REPRODUCING.md)
so the correct configuration, memberships, and checkpoint are selected.

## Two runtime measurements

```bash
pmu-gnn benchmark --system 14 --checkpoint models/held_out/IEEE14/checkpoint.pt --budget stored --output runs/time-stored-01 --confirm-new-experiment
pmu-gnn benchmark --system 14 --checkpoint models/held_out/IEEE14/checkpoint.pt --budget computed --output runs/time-computed-01 --confirm-new-experiment
```

`stored` reads the Greedy reference budget before timing. `computed` includes
the Greedy calculation of that budget during timing. Both then select the given
number of PMUs with the GNN pipeline. Available corrected checkpoints differ
from those used for the historical runtime panel, so these commands produce
separately labeled new measurements.

Use `--limit N` only for a limited smoke run. Such a run is not the full
benchmark population. Full reproduction status and boundaries remain in
[REPRODUCING.md](REPRODUCING.md) and [RESULTS.md](RESULTS.md).
