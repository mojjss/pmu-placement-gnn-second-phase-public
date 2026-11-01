"""Independent Python saved-record verification; no Node, NumPy or Torch required."""
import math
from pathlib import Path
from .artifacts import Catalog, RUN, RUNTIME, ABLATION, TRANSFER, load_json, read_csv, sha
from .npz import numeric_sample, sample_edges
from .coverage import coverage, observed_nodes


def require(ok, text):
    if not ok:
        raise ValueError(text)


def verify(root):
    cat = Catalog(root)
    expected_previews = {
        "results/paper/figures/fig1/fig-1-corrected.png",
        "results/paper/figures/fig2/fig2_accuracy.png",
        "results/paper/figures/fig3/fig3_gnn.png",
        "results/paper/figures/fig3/fig3_greedy.png",
    }
    require(set(cat.manifest.get("required_previews", [])) == expected_previews, "Required preview policy changed")
    require(expected_previews <= {r["path"] for r in cat.records}, "Required preview is absent from catalog")
    for archived in cat.manifest.get("archived_optional_exports", []):
        path = Path(archived["former_path"])
        require(path.as_posix().startswith("results/paper/figures/") and path.suffix.lower() in {".pdf", ".emf"}, "Only optional figure formats may be marked archived")
        require(archived["status"] == "HASH_VERIFIED_ARCHIVED_OUTSIDE_RELEASE", "Unverified archive status")
    for r in cat.records:
        cat.path(r, verify=True)
    report = dict(status="PARTIAL_PAPER_EVIDENCE", artifact_hashes=len(cat.records), samples=0, quality={}, transfer={}, runtime={}, ablation={})
    datasets = {}
    for system, expected in [("14", 21), ("57", 81), ("118", 187), ("300", 412)]:
        index = cat.ending("/IEEE" + system + "/index.csv")
        rows = read_csv(index); datasets[system] = (index, rows)
        require(len(rows) == expected, "Dataset row count")
        require(len({r["sample_id"] for r in rows}) == expected, "Duplicate scenario ID")
        for row in rows:
            require(cat.sample(system, row["file"]).is_file(), "Referenced sample absent")
            report["samples"] += 1
        prefix = RUN + "/IEEE" + system + "/"
        mem = read_csv(cat.source(prefix + "memberships.csv"))
        require(len(mem) == expected and len({int(r["index"]) for r in mem}) == expected, "Membership coverage")
        owners = {}
        for m in mem:
            require(rows[int(m["index"])]["sample_id"] == m["sample_id"], "Membership scenario mismatch")
            require(m["partition"] in {"train", "validation", "test"}, "Unknown partition")
            group = m["input_sha256"]
            require(owners.get(group, m["partition"]) == m["partition"], "Cross-partition group overlap")
            owners[group] = m["partition"]
            if m["scenario_type"] == "intact":
                require(m["partition"] == "train", "Intact case outside train")
        saved = load_json(cat.source(prefix + "test_results.json"))
        tests = {int(m["index"]) for m in mem if m["partition"] == "test"}
        require(len(saved) == len(tests) and {r["index"] for r in saved} == tests, "Test membership mismatch")
        observed = full = ksum = 0
        for r in saved:
            require(rows[r["index"]]["sample_id"] == r["sample_id"], "Test identity mismatch")
            z = numeric_sample(cat.sample(system, rows[r["index"]]["file"]))
            n = z["x"][0][0]; k = sum(v == 1 for v in z["y"][1]); selected = r["placement"]
            require(len(selected) == k == r["reference_K"], "Reference K mismatch")
            count = len(observed_nodes(sample_edges(z), n, selected))
            require(math.isclose(count / n, r["coverage"], abs_tol=1e-12), "Saved coverage mismatch")
            observed += count; full += count == n; ksum += k
        history = read_csv(cat.source(prefix + "history.csv"))
        require([int(h["epoch"]) for h in history] == list(range(1, 151)), "History epochs")
        maximum = max(float(h["validation_node_accuracy"]) for h in history)
        epoch = next(int(h["epoch"]) for h in history if float(h["validation_node_accuracy"]) == maximum)
        selection = load_json(cat.source(prefix + "selection.json"))
        require(selection["selected_epoch"] == epoch, "First maximum checkpoint mismatch")
        require(sha(cat.source(prefix + "checkpoint.pt")) == selection["checkpoint_sha256"], "Selected checkpoint hash")
        report["quality"][system] = dict(test_n=len(saved), mean_coverage_percent=100 * observed / (len(saved) * int(system)),
                                        observed=observed, bus_instances=len(saved)*int(system), full_count=full,
                                        full_percent=100*full/len(saved), mean_K=ksum/len(saved), selected_epoch=epoch)
    runtime = read_csv(cat.source(RUNTIME + "/raw-paired.csv"))
    require(len(runtime) == 25092, "Runtime total count")
    ablation = read_csv(cat.source(ABLATION + "/raw-ablation.csv"))
    require(len(ablation) == 6273, "Ablation total count")
    for system, count in [("14", 20), ("57", 80), ("118", 186), ("300", 411)]:
        report["runtime"][system] = {}
        for method in ["original_Greedy", "optimized_Greedy", "optimized_supplied_K", "optimized_acquired_K"]:
            cell = [r for r in runtime if r["system"] == system and r["method"] == method and r["boundary"] == "online_preparation_inclusive"]
            require(len(cell) == count*3, "Runtime method population")
            pairs = {}
            for r in cell:
                pairs.setdefault(r["sample_id"], []).append(int(r["repeat"]))
                require(math.isfinite(float(r["ms"])) and float(r["ms"]) > 0, "Invalid timing")
            require(len(pairs) == count and all(sorted(v) == [0, 1, 2] for v in pairs.values()), "Runtime pairing/repetitions")
            report["runtime"][system][method] = sum(float(r["ms"]) for r in cell)/len(cell)
        a = [r for r in ablation if r["system"] == system and r["method"] == "A"]
        b = [r for r in ablation if r["system"] == system and r["method"] == "B"]
        require(len(a) == len(b) == count, "Ablation population")
        by_b = {r["sample_id"]: r for r in b}
        for r in a:
            other = by_b[r["sample_id"]]
            require(all(r[k] == other[k] for k in ["K", "input_sha256", "checkpoint_sha256"]), "A/B pairing mismatch")
        index, rows = datasets[system]
        for r in a + b:
            z = numeric_sample(cat.sample(system, rows[int(r["index"])]["file"]))
            require(math.isclose(coverage(sample_edges(z), int(system), load_json_value(r["placement"])), float(r["coverage"]), abs_tol=1e-12), "Ablation saved coverage")
        report["ablation"][system] = dict(n=count, improvement_pp=100*(sum(float(r["coverage"]) for r in b)-sum(float(r["coverage"]) for r in a))/count)
    transfer = load_json(cat.source(TRANSFER + "/records.json"))
    require(len(transfer) == 111, "Transfer total count")
    for method in sorted({r["method"] for r in transfer}):
        cell = [r for r in transfer if r["method"] == method]
        require(len(cell) == 37, "Transfer population")
        observed = full = 0
        index, rows = datasets["118"]
        for r in cell:
            z = numeric_sample(cat.sample("118", rows[r["index"]]["file"]))
            obs = observed_nodes(sample_edges(z), 118, r["selected_indices"])
            require(math.isclose(len(obs)/118, r["coverage"], abs_tol=1e-12) and r["full_coverage"] == (len(obs) == 118), "Transfer coverage/flag")
            observed += len(obs); full += len(obs) == 118
        report["transfer"][method] = dict(n=37, full_count=full, mean_coverage_percent=100*observed/(37*118))
    require(report["quality"]["300"]["full_count"] == 9, "Audited IEEE300 full count changed")
    report["discrepancy"] = dict(paper_IEEE300_full_percent=95.12, saved_full_count=9, denominator=82,
                                 status="CONFLICT; not silently repaired")
    report["limits"] = ["No training/inference/timing/solver/pickle execution", "Final table authorship remains candidate-unconfirmed",
                        "Saved group identifiers checked; independent group reconstruction is a separate split command"]
    return report


def load_json_value(value):
    import json
    return json.loads(value)
