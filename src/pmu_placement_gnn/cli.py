"""CLI help and saved inspection work without importing scientific libraries."""
import argparse
import json
from . import __version__
from .artifacts import project_root, new_output, save_json


def parser():
    p = argparse.ArgumentParser(
        prog="pmu-gnn",
        description="Greedy-supervised PMU placement: inspect saved records or run new experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Explore workflow commands without executing experiments:\n"
                "  pmu-gnn train --help\n"
                "  pmu-gnn benchmark --help\n"
                "Saved-result inspection and verification are explicit commands.\n"
                "Guide: docs/CLI.md; fresh output directories are required."),
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--root", help="Research project checkout (default: discover from current/source directory)")
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="Read-only Python hash/index/membership/coverage/arithmetic checks").add_argument("--output", help="Fresh report directory (optional; otherwise stdout only)")
    saved = commands.add_parser("saved", help="Inspect saved evidence or a documented example; no inference")
    saved.add_argument("--example", help="Example ID, e.g. IEEE14/line_0_0")
    tab = commands.add_parser("tables", help="Export tables from saved records; preserves discrepancies")
    tab.add_argument("--output", required=True)
    fig = commands.add_parser("figures", help="Fig2 saved-history replot; Fig1/3 copy authenticated exports")
    fig.add_argument("--figure", choices=["1", "2", "3"], required=True); fig.add_argument("--output", required=True)
    for name, description in [("build-dataset","Generate NEW intact/N-1 graphs and Greedy labels"),
                              ("split","Recompute grouped memberships from existing raw inputs"),
                              ("train","NEW 150-epoch training and validation checkpoint selection"),
                              ("predict","NEW reference-K single-scenario prediction"),
                              ("evaluate","NEW held-out evaluation using fixed membership"),
                              ("transfer","NEW frozen source checkpoint on target test scenarios"),
                              ("benchmark","NEW resident-grid runtime with stored/computed K"),
                              ("ablation","NEW paired raw Top-K and coverage-aware evaluation"),
                              ("greedy","NEW Greedy reference on an existing scenario")]:
        c = commands.add_parser(name, help=description)
        c.add_argument("--system", choices=["14","57","118","300"], required=True)
        c.add_argument("--output", required=True, help="Fresh isolated output directory; use runs/ or example_runs/")
        if name not in {"split"}:
            c.add_argument("--confirm-new-experiment", action="store_true", help="Explicit execution guard; absent means no scientific work")
        if name in {"split", "train"}:
            c.add_argument("--config", help="Defaults to configs/paper.json"); c.add_argument("--index", help="Optional new dataset index; default packaged IEEE index")
        if name == "build-dataset":
            c.add_argument("--intact-only", action="store_true")
        if name in {"predict", "greedy"}:
            c.add_argument("--sample-id", required=True)
        if name in {"predict", "evaluate", "transfer", "benchmark", "ablation"}:
            c.add_argument("--checkpoint", required=name in {"benchmark", "ablation"}, help="Defaults to corrected audited checkpoint except runtime/ablation require explicit selection")
            c.add_argument("--checkpoint-sha256", help="Expected hash for an explicitly supplied trusted checkpoint")
        if name in {"evaluate", "transfer"}:
            c.add_argument("--memberships", help="Defaults to target corrected memberships"); c.add_argument("--index")
        if name == "transfer":
            c.add_argument("--source-system", choices=["14","57","118","300"], required=True)
        if name == "benchmark":
            c.add_argument("--budget", choices=["stored", "computed"], required=True)
        if name in {"benchmark", "ablation"}:
            c.add_argument("--limit", type=int, help="First N fault cases for a later explicitly authorized smoke run; not full paper population")
    return p


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    try:
        root = project_root(args.root)
        if args.command not in {"verify","saved","tables","figures","split"} and not args.confirm_new_experiment:
            p.error("No experiment executed: add --confirm-new-experiment only when you intend a NEW scientific run")
        if hasattr(args, "limit") and args.limit is not None and args.limit < 1:
            p.error("--limit must be positive")
        output = new_output(root, args.output) if getattr(args, "output", None) else None
        if args.command == "verify":
            from .verification import verify
            result = verify(root)
            if output:
                save_json(output / "verification.json", result)
        elif args.command == "saved":
            from .outputs import example
            if args.example:
                result = example(root, args.example)
            else:
                from .verification import verify
                result = verify(root)
        elif args.command in {"tables", "figures"}:
            from .outputs import tables, figures
            result = tables(root, output) if args.command == "tables" else figures(root, output, args.figure)
        elif args.command == "build-dataset":
            from .dataset import build
            result = build(root, args.system, output, args.intact_only)
        elif args.command == "split":
            from .dataset import prepare
            _, _, report = prepare(root, args.system, output, args.config, args.index)
            result = dict(output=str(output), partitions={k:len(v) for k,v in report["partitions"].items()})
        elif args.command in {"train", "predict", "evaluate", "transfer", "greedy"}:
            from . import experiments
            if args.command == "train":
                result = experiments.train(root, args.system, output, args.config, args.index)
            elif args.command == "predict":
                result = experiments.predict(root, args.system, args.sample_id, output, args.checkpoint, args.checkpoint_sha256)
            elif args.command == "greedy":
                result = experiments.greedy_example(root, args.system, args.sample_id, output)
            else:
                result = experiments.evaluate(root, args.system, output, args.checkpoint, args.checkpoint_sha256,
                    args.memberships, getattr(args,"source_system",None), args.index)
        else:
            from . import benchmarks
            if args.command == "benchmark":
                result = benchmarks.runtime(root,args.system,output,args.checkpoint,args.checkpoint_sha256,args.budget,args.limit)
            else:
                result = benchmarks.ablation(root,args.system,output,args.checkpoint,args.checkpoint_sha256,args.limit)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, ImportError) as exc:
        p.exit(2, "Error: " + str(exc) + "\n")
