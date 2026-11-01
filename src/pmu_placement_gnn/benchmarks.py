"""New runtime/ablation execution with explicit checkpoint provenance."""
import csv
import time
from pathlib import Path
from .artifacts import save_json, sha
from .coverage import coverage
from .dataset import dataset, settings
from .experiments import checkpoint
from .method import graph_definitions, optimized_definitions, sparse_definitions


def runtime(root, system, output, checkpoint_path, checkpoint_sha=None, budget="stored", limit=None):
    import numpy as np
    import torch
    import pandapower.networks as networks
    from torch_geometric.data import Data
    cfg = settings(root)
    torch.set_num_threads(cfg["runtime_threads"]); torch.set_num_interop_threads(cfg["runtime_interop_threads"])
    env, ckpath, ck = checkpoint(root, system, checkpoint_path, checkpoint_sha)
    opt = optimized_definitions(); sparse = sparse_definitions(opt); graph_defs = graph_definitions()
    samples, rows = dataset(root, system)
    net = getattr(networks, "case" + str(system))()

    def resident_graph(row):
        table = getattr(net, row["kind"]); idx = int(row["idx"]); before = table.at[idx, "in_service"]
        table.at[idx, "in_service"] = False
        try:
            return graph_defs["build_graph"](net)
        finally:
            table.at[idx, "in_service"] = before

    def call(method, row, stored_k):
        graph = resident_graph(row)
        if method == "original_Greedy":
            return env["greedy_pmu_placement_simple"](graph)
        # Original conversion computes every stored feature/attribute, even unused edge attributes.
        x, edges, attrs, y, ids = graph_defs["graph_to_gnn_arrays"](graph, net, [])
        data = Data(x=torch.tensor(x, dtype=torch.float32), edge_index=torch.tensor(edges, dtype=torch.long),
                    y=torch.zeros(len(x), dtype=torch.long))
        if budget == "stored":
            return sparse["supplied"](env["model"], data, stored_k, "sparse", False)
        return sparse["acquired"](env["model"], data, "sparse", False)

    output_rows = []; indices = [i for i, r in enumerate(rows) if r["scenario_type"] == "faulted"]
    if limit is not None:
        indices = indices[:limit]
    with torch.no_grad():
        for i in indices:
            row = rows[i]; z = samples.arrays(i); k = int((z["y"] == 1).sum())
            for method in ("original_Greedy", "GNN"):
                expected = call(method, row, k)  # one excluded warmup
                for repeat in range(cfg["runtime_repetitions"]):
                    start = time.perf_counter_ns(); selected = call(method, row, k)
                    ms = (time.perf_counter_ns() - start) / 1e6
                    if selected != expected or len(selected) != k:
                        raise ValueError("Placement/repetition/reference-K mismatch")
                    output_rows.append(dict(system=str(system), sample_id=row["sample_id"], repeat=repeat, method=method,
                        budget=budget, ms=ms, K=k, placement=selected,
                        coverage=coverage(z["edge_index"].T.tolist(), len(z["x"]), selected)))
    save_json(Path(output) / "timings.json", output_rows)
    save_json(Path(output) / "run.json", dict(status="NEW_BENCHMARK_NOT_HISTORICAL_TIMING", checkpoint_sha256=sha(ckpath),
              boundary="resident graph outage+features+tensors+original Greedy or sparse GCN+ranking+K-step decoder",
              excluded="imports, network/model load, saved dataset reads, validation and serialization",
              warmup=1, repetitions=3, threads=1, interop_threads=1, K=budget,
              population="all faulted training-inclusive" if limit is None else "limited first-fault example population",
              caveat="Portable driver has different Python overhead/scheduling; not byte-identical benchmark runner or paper timing reproduction"))
    return dict(output=str(output), timing_records=len(output_rows))


def ablation(root, system, output, checkpoint_path, checkpoint_sha=None, limit=None):
    import torch
    env, ckpath, ck = checkpoint(root, system, checkpoint_path, checkpoint_sha)
    samples, rows = dataset(root, system)
    opt = optimized_definitions(); sparse = sparse_definitions(opt)
    indices = [i for i, r in enumerate(rows) if r["scenario_type"] == "faulted"]
    if limit is not None:
        indices = indices[:limit]
    records = []
    with torch.no_grad():
        for i in indices:
            data = samples[i]; k = int((data.y == 1).sum())
            probs = sparse["probabilities"](env["model"], data, "sparse", False)
            selections = {"A_raw_TopK": torch.topk(torch.from_numpy(probs), k).indices.tolist(),
                          "B_coverage_aware": opt["incremental"](probs, data.edge_index.numpy(), k)}
            for method, selected in selections.items():
                records.append(dict(index=i, sample_id=rows[i]["sample_id"], K=k, method=method, placement=selected,
                    coverage=coverage(data.edge_index.T.tolist(), data.num_nodes, selected)))
    save_json(Path(output) / "ablation.json", records)
    save_json(Path(output) / "run.json", dict(status="NEW_ABLATION_NOT_PAPER_REPLACEMENT", checkpoint_sha256=sha(ckpath),
              controls="A/B only; original C/D/E additional controls not reimplemented", population="faulted training-inclusive"))
    return dict(output=str(output), records=len(records))
