"""Inspect and export existing records; never compute new model predictions."""
import csv
import json
import shutil
from pathlib import Path
from .artifacts import Catalog, RUN, load_json, read_csv, save_json
from .verification import verify


def example(root, identifier):
    parts = identifier.split("/")
    if len(parts) != 2 or parts[0] not in {"IEEE14", "IEEE57", "IEEE118", "IEEE300"}:
        raise ValueError("Example must be IEEE14/line_0_0 or another sample_manifest.csv entry")
    rows = read_csv(Path(root) / "examples/sample_manifest.csv")
    matches = [r for r in rows if "IEEE" + r["system"] == parts[0] and r["sample_id"] == parts[1]]
    if len(matches) != 1:
        raise ValueError("Example is not in the documented manifest")
    return load_json(Path(root) / matches[0]["output_path"])


def tables(root, output):
    report = verify(root)
    rows = [dict(system="IEEE"+s, **v, paper_reported_full_percent={"14":100,"57":93.75,"118":81.08,"300":95.12}[s],
                 evidence_status="CONFLICT" if s == "300" else "SAVED_CHECK_PASSED;FINAL_USE_UNCONFIRMED")
            for s, v in report["quality"].items()]
    collections = {"table_ii_a_quality.csv": rows,
                   "table_ii_b_transfer.csv": [dict(method=k, **v) for k,v in report["transfer"].items()],
                   "table_ii_c_runtime.csv": [dict(system=s, **v) for s,v in report["runtime"].items()],
                   "table_ii_d_ablation.csv": [dict(system=s, **v) for s,v in report["ablation"].items()]}
    for name, values in collections.items():
        with (Path(output) / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(values[0])); w.writeheader(); w.writerows(values)
    save_json(Path(output) / "provenance.json", dict(kind="NEW_EXPORT_FROM_EXISTING_RECORDS", report=report,
              note="Not the historical final Word table authoring script; IEEE300 disagreement retained"))
    return dict(output=str(output), files=list(collections))


def figures(root, output, figure):
    cat = Catalog(root)
    if figure in {"1", "3"}:
        # Authenticated exports can be copied without rendering or loading an illustration checkpoint.
        prefix = "results/paper/figures/fig" + figure + "/"
        records = [r for r in cat.records if r["path"].startswith(prefix) and Path(r["path"]).suffix == ".png"]
        if not records:
            raise ValueError("No verified saved PNG previews available")
        for r in records:
            shutil.copyfile(cat.path(r, verify=True), Path(output) / Path(r["path"]).name)
        save_json(Path(output) / "provenance.json", dict(kind="BYTE_IDENTICAL_SAVED_EXPORT_COPY", artifact_ids=[r["artifact_id"] for r in records],
                  note="Native Visio rebuild/illustration inference is blocked by private dependencies; this is export inspection"))
        return dict(files=len(records), output=str(output))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    history = read_csv(cat.source(RUN + "/IEEE300/history.csv", verify=True))
    # Match the authenticated figure's font; fail instead of silently substituting.
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    with plt.rc_context({"font.family":"Times New Roman"}):
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot([int(r["epoch"]) for r in history], [float(r["train_node_accuracy"]) for r in history], linewidth=1.5, label="Train acc")
        ax.plot([int(r["epoch"]) for r in history], [float(r["validation_node_accuracy"]) for r in history], linewidth=1.5, label="Val acc")
        ax.set(title="GNN training/validation accuracy (IEEE300)", xlabel="Epoch", ylabel="Accuracy")
        ax.grid(True); ax.legend(loc="upper left"); fig.tight_layout()
        for extension in ("png",):
            fig.savefig(Path(output) / ("fig2_accuracy."+extension), dpi=600, bbox_inches="tight")
        plt.close(fig)
    save_json(Path(output) / "provenance.json", dict(kind="PORTABLE_REPLOT_SAVED_HISTORY", records=150,
              note="Same saved curves; renderer/export bytes may differ; original final exports remain immutable"))
    return dict(output=str(output), records=150)
