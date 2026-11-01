"""Project-relative artifact catalog and write isolation; standard library only."""
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def project_root(value=None):
    if value is not None:
        root = Path(value).resolve()
        if not (root / "results/paper/manifest.json").is_file():
            raise ValueError("--root must identify the research project with its provenance catalog")
        return root
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "results/paper/manifest.json").is_file():
        return candidate
    raise ValueError("Pass --root when using a wheel installed outside a research checkout")



class Catalog:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = load_json(self.root / "results/paper/manifest.json")
        self.records = self.manifest["records"]
        self.by_source = {r["key"]: r for r in self.records}

    def path(self, record, verify=False):
        target = (self.root / record["path"]).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError("Catalog traversal outside project")
        if not target.is_file():
            raise FileNotFoundError(record["path"])
        if verify and sha(target) != record["sha256"]:
            raise ValueError("Artifact hash mismatch: " + record["artifact_id"])
        return target

    def source(self, relative, verify=False):
        if relative not in self.by_source:
            raise ValueError("Historical dependency is not publicly packaged: " + relative)
        return self.path(self.by_source[relative], verify)

    def ending(self, suffix, verify=False):
        rows = [r for r in self.records if r["key"].endswith(suffix)]
        if len(rows) != 1:
            raise ValueError("Expected exactly one artifact ending in " + suffix)
        return self.path(rows[0], verify)

    def sample(self, system, filename):
        filename = filename.replace("\\", "/")
        index = self.ending("/IEEE" + str(system) + "/index.csv")
        target = (index.parent / filename).resolve()
        if not target.is_relative_to(index.parent):
            raise ValueError("Sample path traversal")
        return target


def new_output(root, value):
    root = Path(root).resolve()
    target = Path(value)
    if not target.is_absolute():
        target = Path(root) / target
    target = target.resolve()
    for name in ["data", "results", "figures", "tables", "models", "src", "configs", "examples", "provenance", "docs", "tests", ".venv"]:
        protected = Path(root) / name
        if target == protected or target.is_relative_to(protected):
            raise ValueError("Output is inside protected project evidence/source; use runs/ or example_runs/")
    if target == Path(root):
        raise ValueError("Output must be a fresh child directory")
    target.mkdir(parents=True, exist_ok=False)
    return target


RUN = "held_out"
RUNTIME = "runtime"
ABLATION = "ablation"
TRANSFER = "transfer"
