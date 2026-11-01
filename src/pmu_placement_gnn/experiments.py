"""Explicit new-experiment interfaces; never called by saved-result inspection."""
import csv
from pathlib import Path
from .artifacts import Catalog, RUN, load_json, save_json, sha
from .coverage import coverage
from .dataset import dataset, prepare, settings
from .method import model_definitions, training_definition, evaluation_definition


def checkpoint(root, system, value=None, expected_sha=None):
    import torch
    cat = Catalog(root)
    path = Path(value) if value else cat.source(RUN + "/IEEE" + str(system) + "/checkpoint.pt", verify=True)
    if not path.is_absolute():
        path = Path(root) / path
    path = path.resolve()
    known = next((r for r in cat.records if cat.path(r) == path), None)
    selection_path = path.parent / "selection.json"
    expected = expected_sha or (known["sha256"] if known else
               load_json(selection_path).get("checkpoint_sha256") if selection_path.is_file() else None)
    if not expected or sha(path) != expected:
        raise ValueError("Checkpoint requires an audited catalog hash or a matching explicit/selection SHA256")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    env = model_definitions(system)
    model = env["PMUGCN"](ck["in_channels"], ck["hidden_channels"], ck["out_channels"])
    model.load_state_dict(ck["state_dict"], strict=True); model.eval(); env["model"] = model
    return env, path, ck


def train(root, system, output, config=None, index=None):
    import torch
    cfg = settings(root, config)
    torch.set_num_threads(cfg["training_threads"])
    torch.set_num_interop_threads(cfg["training_interop_threads"])
    samples, rows, report = prepare(root, system, output, config, index)
    env = model_definitions(system)
    training_definition(cfg["method"])(env, samples, report, Path(output))
    save_json(Path(output) / "run.json", dict(status="NEW_TRAINING_NOT_PAPER_REPLACEMENT", system=str(system),
              configuration=cfg, test_evaluation=False, method="audited train_and_select definition unchanged"))
    return dict(output=str(output), selected=load_json(Path(output) / "selection.json"))


def evaluate(root, system, output, checkpoint_path=None, checkpoint_sha=None, memberships=None, source_system=None, index=None):
    import torch
    source = str(source_system or system)
    env, ckpath, ck = checkpoint(root, source, checkpoint_path, checkpoint_sha)
    samples, rows = dataset(root, system, index)
    membership = Path(memberships) if memberships else Catalog(root).source(RUN + "/IEEE" + str(system) + "/memberships.csv")
    if not membership.is_absolute():
        membership = Path(root) / membership
    from .artifacts import read_csv
    assigned = read_csv(membership)
    if len(assigned) != len(rows) or len({int(r["index"]) for r in assigned}) != len(rows) or any(rows[int(r["index"])]["sample_id"] != r["sample_id"] for r in assigned):
        raise ValueError("Membership/dataset identity mismatch")
    parts = {part: sorted(int(r["index"]) for r in assigned if r["partition"] == part)
             for part in ("train", "validation", "test")}
    report = dict(partitions=parts)
    save_json(Path(output) / "selection.json", dict(checkpoint_sha256=sha(ckpath), source_system=source,
              target_system=str(system), selection_basis="existing checkpoint; no target model selection"))
    # Preserve the corrected evaluation's historical forward/probability and decoder calls.
    fn = evaluation_definition()
    fn(env, samples, rows, report, Path(output))
    save_json(Path(output) / "run.json", dict(status="NEW_EVALUATION_NOT_PAPER_REPLACEMENT", source_system=source,
              target_system=str(system), checkpoint_sha256=sha(ckpath), memberships_sha256=sha(membership), reference_K="target stored Greedy labels"))
    return dict(output=str(output), records=len(parts["test"]))


def predict(root, system, sample_id, output, checkpoint_path=None, checkpoint_sha=None):
    import torch
    env, ckpath, ck = checkpoint(root, system, checkpoint_path, checkpoint_sha)
    samples, rows = dataset(root, system)
    found = [i for i, r in enumerate(rows) if r["sample_id"] == sample_id]
    if len(found) != 1:
        raise ValueError("Unknown or ambiguous sample ID")
    i = found[0]; data = samples[i]; k = int((data.y == 1).sum())
    with torch.no_grad():
        probs = torch.softmax(env["model"](data.x, data.edge_index), -1)[:, 1].tolist()
        selected = env["gnn_select_pmus_coverage_aware"](data, k, topM=max(2*k, 80))
    result = dict(status="NEW_PREDICTION", system=str(system), sample_id=sample_id, reference_K=k,
                  checkpoint_sha256=sha(ckpath), probabilities=probs, placement=selected,
                  coverage=coverage(data.edge_index.T.tolist(), data.num_nodes, selected))
    save_json(Path(output) / "prediction.json", result)
    return result


def greedy_example(root, system, sample_id, output):
    from .dataset import dataset
    from .method import graph_definitions
    import networkx as nx
    samples, rows = dataset(root, system)
    i = next(i for i, r in enumerate(rows) if r["sample_id"] == sample_id)
    z = samples.arrays(i); graph = nx.Graph(); graph.add_nodes_from(range(len(z["x"])))
    graph.add_edges_from(z["edge_index"].T.tolist())
    env = graph_definitions(); selected = env["greedy_pmu_placement"](graph)
    result = dict(status="NEW_GREEDY_EXECUTION", system=str(system), sample_id=sample_id,
                  placement=selected, K=len(selected), coverage=coverage(z["edge_index"].T.tolist(), len(z["x"]), selected))
    save_json(Path(output) / "greedy.json", result)
    return result
