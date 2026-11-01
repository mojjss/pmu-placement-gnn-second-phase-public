# Greedy-Supervised GNN for PMU Placement

[![Software CI](https://github.com/mojjss/pmu-placement-gnn-second-phase-public/actions/workflows/ci.yml/badge.svg)](https://github.com/mojjss/pmu-placement-gnn-second-phase-public/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](pyproject.toml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Research software for learning PMU placement from Greedy reference solutions.
A compact graph neural network predicts a PMU probability for each bus; a
coverage-aware decoder then selects locations using a Greedy reference PMU
budget. The project provides IEEE test-system graphs, saved results, available
checkpoints, and commands for inspection and new experiments.



## What is implemented

- Direct one-hop observability: a PMU observes its own bus and adjacent buses.
- Greedy reference placement for intact and individual line/transformer outages.
- Four systems: IEEE 14-bus, 57-bus, 118-bus, and 300-bus.
- Saved datasets containing 701 graphs: 697 outage cases and four intact cases.
- Grouped training, validation, and test splits with fixed seed 42.
- Two GCN layers with 128 hidden channels, ReLU, and a two-class classifier.
- Coverage-aware decoding with exactly `K` selections and a Top-`M` candidate pool.
- Saved-result inspection, table export, figure export, and new-experiment commands.

Greedy is a heuristic, not a certified minimum-PMU solver. `K` comes from
Greedy reference placements; the GNN does not independently predict this budget.
The study re-optimizes each topology separately and does not establish one fixed
PMU deployment that remains observable under every outage.

Bus features keep their original scales. The GCN retains stored connection
directions and repetitions without automatically adding reverse connections;
physical coverage is calculated on the undirected network. Stored branch
attributes are not directly used by the GCN. See [the method](docs/METHOD.md).

## Repository layout

| Path | Contents |
| --- | --- |
| `src/pmu_placement_gnn/` | Python package, CLI, and portable method definitions |
| `configs/` | Method and experiment settings |
| `data/` | Saved graphs, indices, and split memberships |
| `models/held_out/` | Four available system-specific checkpoints |
| `results/paper/` | Saved coverage, transfer, runtime, ablation, and figure records |
| `examples/` | Twelve saved intact/outage examples across the four systems |
| `scripts/` | Checkout CLI launcher and clean ZIP builder |
| `tests/` | Core, packaging, and optional scientific checks |
| `docs/` | Method, reproduction, results, and CLI guides |

The [first-phase repository](https://github.com/mojjss/pmu-placement-gnn-first-phase-public)
is a related release with different implementation defaults. Its normalization,
edge handling, commands, and DOI must not be substituted for this release.

## Installation

Python 3.11+ is required. Clone the repository and create an environment:

```bash
git clone https://github.com/mojjss/pmu-placement-gnn-second-phase-public.git
cd pmu-placement-gnn-second-phase-public
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Install the base package:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

The base package needs no scientific dependencies and can inspect saved examples.
For graph/ML experiments and plotting, install the optional dependencies:

```bash
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[graph,ml,plots]"
```

`torchvision` and `torchaudio` are not required. GPU users should select a
compatible PyTorch build rather than the CPU command above.
[requirements-reproduce.txt](requirements-reproduce.txt) records historical
versions; it is not a complete environment lock.

## Quickstart: inspect existing results

```bash
pmu-gnn --help
pmu-gnn --version
pmu-gnn saved --example IEEE14/line_0_0
pmu-gnn verify
```

These commands do not train or execute a GNN. The verifier checks artifact
hashes, scenario identities, split membership, and saved coverage arithmetic.
It intentionally fails on missing or changed evidence; see the current
[verification limitation](#reproduction-status-and-evidence-limits).

Once the saved evidence passes verification, export the four table panels:

```bash
pmu-gnn tables --output runs/tables-01
```

Use `python scripts/pmu.py` instead of `pmu-gnn` when running directly from the
checkout. From another directory, pass `--root /path/to/checkout` before the
subcommand. Output directories must be fresh; existing evidence is protected.

## New experiments

After installing the relevant optional dependencies:

```bash
pmu-gnn greedy --system 14 --sample-id line_0_0 --output example_runs/greedy-01 --confirm-new-experiment
pmu-gnn train --system 14 --config configs/paper.json --output runs/train-14-01 --confirm-new-experiment
```

New scientific runs require `--confirm-new-experiment` and write separate
outputs. They do not replace saved paper results. See the
[CLI reference](docs/CLI.md) and [reproduction guide](docs/REPRODUCING.md)
for dataset generation, prediction, held-out evaluation, transfer, runtime,
ablation, and figure commands.

## Reproduction status and evidence limits

The paper reports 99.65%–100% mean bus coverage and a 95.12% IEEE300
full-coverage rate. **Mean coverage and full-coverage frequency are different
metrics.** Some historical runtime/ablation checkpoints and native figure inputs
are not distributed, so this release does not claim complete historical
reproduction.


## Software tests and continuous integration

After installing the package, run the software-only suite:

```bash
python -m unittest discover -s tests/software -v
```

The [Software CI workflow](.github/workflows/ci.yml) checks package installation,
Python syntax, CLI behavior, and unit tests with synthetic inputs on Linux and
Windows. It does not load datasets, model checkpoints, saved experiment outputs,
or final paper tables. A passing Software CI run concerns the code interface
and tested behavior, not reproduction of published numerical results.

Saved-artifact checks remain available separately:

```bash
pmu-gnn verify
```

The existing full test suite is also retained:
`python -m unittest discover -s tests -v`. Some of its checks read saved
examples and previews; graph/ML checks skip without optional dependencies.
See [the evidence status](docs/RESULTS.md) before interpreting artifact checks.

## Citation

If you use this software or method in research, please cite the software release
and associated paper. [CITATION.cff](CITATION.cff) supplies GitHub's
**Cite this repository** information. No archival DOI is assigned here; the
first-phase DOI identifies a different release.



## Contributing, support, and license

See [CONTRIBUTING.md](CONTRIBUTING.md) for changes and ordinary bug reports.
Report security-sensitive issues according to [SECURITY.md](SECURITY.md),
without posting exploit details publicly.

Original project software: **Apache License 2.0**; see [LICENSE](LICENSE).
Copyright 2026 Mojtaba Sadafi. Third-party libraries and test-system data retain
their applicable terms; the software license does not relicense the paper or
third-party material.

Author: [Mojtaba Sadafi](https://mojsadafi.ir).
