# Reproduction guide

Commands below use `pmu-gnn` after editable installation, or `python scripts/pmu.py` from the checkout. `--root <checkout>` works from other directories. Outputs must be new and outside saved data, models and paper results. Use `runs/NEW_NAME` or `example_runs/NEW_NAME`.

## Saved records: no model execution

```powershell
pmu-gnn saved --example IEEE57/line_0_0
pmu-gnn verify --output runs/verify-01
pmu-gnn tables --output runs/tables-01
pmu-gnn figures --figure 1 --output runs/figure1-01
pmu-gnn figures --figure 2 --output runs/figure2-01
pmu-gnn figures --figure 3 --output runs/figure3-01
```

The verifier checks hashes, sample/index identities, memberships, checkpoint selection hashes, coverage and aggregate arithmetic. Table export creates four CSV panels and provenance from saved records, retaining the IEEE300 disagreement. It is not the historical final Word-table authoring script.

Fig.2 is a new replot of saved IEEE300 history, requiring Matplotlib and Times New Roman. Fig.1/3 commands copy retained authenticated PNG previews; they do not regenerate native artwork or run illustration inference. See the per-item mapping in RESULTS.md.

## Graphs, Greedy labels and grouped splits

```powershell
pmu-gnn build-dataset --system 14 --intact-only --output runs/intact-14-01 --confirm-new-experiment
pmu-gnn build-dataset --system 14 --output runs/n1-14-01 --confirm-new-experiment
pmu-gnn greedy --system 14 --sample-id line_0_0 --output example_runs/greedy-01 --confirm-new-experiment
pmu-gnn split --system 14 --config configs/paper.json --output runs/split-01
```

The full dataset command enumerates new physical outages and exports samples/, index.csv and generation.json. It does not replace the saved inputs. Split recomputes group hashes, assignments and training weights without training. Its split.json/memberships.csv can be compared with data/splits/. For a separately generated dataset pass `--index runs/n1-14-01/index.csv` to split/train. New construction is not claimed as an authenticated replay of the original generator environment.

## Training and evaluation: expensive new executions

```powershell
pmu-gnn train --system 14 --config configs/paper.json --output runs/train-14-01 --confirm-new-experiment
pmu-gnn predict --system 14 --sample-id line_1_1 --output runs/predict-14-01 --confirm-new-experiment
pmu-gnn evaluate --system 14 --output runs/evaluate-14-01 --confirm-new-experiment
pmu-gnn transfer --source-system 57 --system 118 --output runs/transfer-01 --confirm-new-experiment
```

Train writes memberships, split, history, checkpoint.pt, selection.json and run.json. Predict/evaluate default to packaged corrected system checkpoints. Transfer defaults to corrected source57 weights and target118 fixed tests, using target stored Greedy K without target tuning.

To evaluate new weights use `--checkpoint runs/train-14-01/checkpoint.pt --memberships runs/train-14-01/memberships.csv`. Noncatalog weights need a matching selection hash or explicit trusted `--checkpoint-sha256`. Weights are loaded with weights_only=True; do not pass untrusted files. New outputs are not declared original paper evidence.

## Runtime and A/B ablation: expensive new executions

```powershell
pmu-gnn benchmark --system 14 --checkpoint models/held_out/IEEE14/checkpoint.pt --budget stored --output runs/time-stored-01 --confirm-new-experiment
pmu-gnn benchmark --system 14 --checkpoint models/held_out/IEEE14/checkpoint.pt --budget computed --output runs/time-computed-01 --confirm-new-experiment
pmu-gnn ablation --system 14 --checkpoint models/held_out/IEEE14/checkpoint.pt --output runs/ablation-01 --confirm-new-experiment
```

These explicitly selected available models differ from the January models underlying the historical runtime/ablation panels. The commands are supported new workflows, not reproductions of those panel values. Exact historical reruns remain blocked until separately authorized checkpoint distribution. No download placeholder or substitute is provided.

Runtime includes resident-grid outage, graph/features/tensors, prediction, ranking and complete decoder. Startup/loading, disk reads, correctness checks and serialization are excluded. One CPU/interop thread, one warmup and three repeats are used. Stored K is read outside timing; computed K includes incremental Greedy inside timing. Driver scheduling/overhead differs from the original benchmark. `--limit N` identifies a limited smoke population, never the full paper population. Ablation runs only A raw Top-K/B coverage-aware; saved additional controls remain in raw evidence.

## Lightweight checks

`python -m unittest discover -s tests -v` runs core checks. Optional graph/ML checks skip when dependencies are absent. Current executed commands and precise limitations are in RESULTS.md. The recorded historical requirements and the newer test environment must not be conflated.

Optional EMF/PDF figure exports are intentionally archived outside this release, not missing required inputs. The manifest retains their hashes and archive status, while all four required PNG previews remain subject to normal hash verification. Figure generation inputs and numerical records are preserved.
