"""Path-portable conference graph conversion, label export and grouped splits."""
import csv
import json
from pathlib import Path
from .artifacts import Catalog, load_json, new_output, read_csv, save_json, sha
from .method import graph_definitions, split_definitions


def settings(root, config=None):
    config_path = Path(config) if config else Path(root) / "configs/paper.json"
    if not config_path.is_absolute():
        config_path = Path(root) / config_path
    value = load_json(config_path)
    if value["method"] != load_json(Path(root) / "configs/reference_method.json"):
        raise ValueError("Configuration changes the audited method; this interface is for the frozen conference settings")
    return value


class Samples:
    def __init__(self, directory, rows, expected_hashes=None):
        self.directory, self.rows = Path(directory).resolve(), rows
        self.expected_hashes = expected_hashes or {}

    def path(self, i):
        f = (self.directory / self.rows[i]["file"].replace("\\", "/")).resolve()
        if not f.is_relative_to(self.directory):
            raise ValueError("Sample escapes dataset directory")
        if self.expected_hashes and sha(f) != self.expected_hashes.get(f):
            raise ValueError("Pinned dataset sample changed: " + self.rows[i]["sample_id"])
        return f

    def arrays(self, i):
        import numpy as np
        with np.load(self.path(i), allow_pickle=False) as z:
            return {k: z[k].copy() for k in ["x", "edge_index", "edge_attr", "y", "bus_ids"]}

    def labels(self, i):
        return self.arrays(i)["y"]

    def __getitem__(self, i):
        import torch
        from torch_geometric.data import Data
        z = self.arrays(i)
        return Data(x=torch.tensor(z["x"], dtype=torch.float32),
                    edge_index=torch.tensor(z["edge_index"], dtype=torch.long),
                    edge_attr=torch.tensor(z["edge_attr"], dtype=torch.float32),
                    y=torch.tensor(z["y"], dtype=torch.long))


def dataset(root, system, index=None):
    path = Path(index) if index else Path(root) / "data" / ("IEEE" + str(system)) / "index.csv"
    if not path.is_absolute():
        path = Path(root) / path
    rows = read_csv(path)
    expected = None
    if index is None:
        cat = Catalog(root)
        cat.ending("/IEEE" + str(system) + "/index.csv", verify=True)
        expected = {cat.path(r): r["sha256"] for r in cat.records if r["path"].startswith("data/IEEE" + str(system) + "/samples/")}
        if len(expected) != len(rows):
            raise ValueError("Pinned dataset sample inventory differs from index")
    return Samples(path.parent, rows, expected), rows


def prepare(root, system, output, config=None, index=None):
    cfg = settings(root, config)
    samples, rows = dataset(root, system, index)
    env = split_definitions()
    hashes = []
    for i in range(len(rows)):
        z = samples.arrays(i)
        hashes.append(env["input_hash"](z["x"], z["edge_index"], z["bus_ids"]))
    parts = env["grouped_split"](rows, hashes)
    counts, weights = env["training_weights"](samples.labels, parts["train"])
    report = dict(system=str(system), configuration=cfg["method"], partitions=parts,
                  training_label_counts=counts, class_weights=weights,
                  dataset_index_sha256=sha(samples.directory / "index.csv"),
                  provenance="PORTABLE_DERIVATIVE_NEW_SPLIT; compare with saved memberships before any experiment")
    save_json(Path(output) / "split.json", report)
    with (Path(output) / "memberships.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["index", "sample_id", "scenario_type", "file", "partition", "input_sha256", "group_id"])
        w.writeheader()
        for part, indices in parts.items():
            for i in indices:
                w.writerow(dict(index=i, sample_id=rows[i]["sample_id"], scenario_type=rows[i]["scenario_type"],
                                file=rows[i]["file"], partition=part, input_sha256=hashes[i], group_id=hashes[i]))
    return samples, rows, report


def build(root, system, output, intact_only=False):
    import numpy as np
    import pandapower.networks as networks
    env = graph_definitions()
    net = getattr(networks, "case" + str(system))()
    base = env["build_graph"](net)
    base_pmus = env["greedy_pmu_placement"](base)
    scenarios = [("intact", None, None)]
    if not intact_only:
        for kind in ("line", "trafo"):
            scenarios.extend(("faulted", kind, int(i)) for i in getattr(net, kind).index)
    rows = []
    Path(output, "samples").mkdir()
    for position, (scenario_type, kind, idx) in enumerate(scenarios):
        current = net if kind is None else env["deactivate"](net, kind, idx)
        graph = env["build_graph"](current)
        selected = base_pmus if kind is None else env["greedy_pmu_placement"](graph)
        x, edges, attr, y, buses = env["graph_to_gnn_arrays"](graph, current, selected)
        sample_id = "intact" if kind is None else f"{kind}_{idx}_{position-1}"
        filename = "samples/" + sample_id + ".npz"
        fixed = 100 * len(env["observed_nodes"](graph, base_pmus)) / len(graph)
        reopt = 100 * len(env["observed_nodes"](graph, selected)) / len(graph)
        components = env["nx"].number_connected_components(graph)
        graph_y = np.array([reopt, len(selected)] if kind is None else
                           [fixed, reopt, len(selected)-len(base_pmus), components], dtype=np.float32)
        np.savez(Path(output) / filename, x=x, edge_index=edges, edge_attr=attr, y=y, bus_ids=buses,
                 graph_y=graph_y, scenario_type=np.array([scenario_type]))
        failed = (None, None) if kind is None else env["failed_edge_buses"](net, kind, idx)
        rows.append(dict(sample_id=sample_id, file=filename, scenario_type=scenario_type, kind=kind or "", idx=idx if idx is not None else "",
                         failed_u=failed[0] if kind else "", failed_v=failed[1] if kind else "", coverage_fixed_percent=fixed,
                         coverage_reopt_percent=reopt, components=components))
    with Path(output, "index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    save_json(Path(output) / "generation.json", dict(status="NEW_DATASET_NOT_HISTORICAL_REPLACEMENT", system=str(system),
              intact_only=intact_only, source="audited Greedy definitions; original generator execution binding remains incomplete"))
    return dict(samples=len(rows), output=str(output))
