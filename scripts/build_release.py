"""Build local source/results ZIPs with one root and no nested bundles or environments."""
import argparse
import fnmatch
import os
from pathlib import Path
import zipfile

ROOT_FILES = {"README.md", "CITATION.cff", "pyproject.toml", "requirements-reproduce.txt",
              ".gitignore", ".gitattributes", "AGENTS.md", "LICENSE", "NOTICE", ".env.example"}
ROOT_DIRS = {"src", "configs", "data", "models", "results", "examples", "tests", "scripts", "docs"}
IGNORED_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "build", "dist",
                "runs", "example_runs", "verification_runs", "node_modules"}


def excluded(relative):
    path = Path(relative)
    if any(part in IGNORED_DIRS or part.endswith(".egg-info") for part in path.parts):
        return True
    if fnmatch.fnmatch(path.name, "*.py[cod]") or path.suffix.lower() in {".zip", ".key", ".pem"}:
        return True
    if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
        return True
    if len(path.parts) == 1:
        return path.name not in ROOT_FILES
    return path.parts[0] not in ROOT_DIRS


def selected_files(root, results_only=False):
    root = Path(root).resolve()
    selected = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        kept = []
        for name in dirs:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError("Symlink directories are not release inputs")
            rel = path.relative_to(root)
            if name not in IGNORED_DIRS and not name.endswith(".egg-info") and rel.parts[0] in ROOT_DIRS:
                kept.append(name)
        dirs[:] = kept
        for name in files:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError("Symlink files are not release inputs")
            rel = path.relative_to(root)
            if excluded(rel):
                continue
            if results_only and rel.parts[0] not in {"results", "examples"}:
                continue
            selected.append((path, rel.as_posix()))
    return sorted(selected, key=lambda item: item[1])


def build_zip(root, destination, results_only=False):
    root = Path(root).resolve()
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Choose a fresh archive output path")
    if destination.suffix.lower() != ".zip":
        raise ValueError("Archive output must end in .zip")
    inputs = selected_files(root, results_only)
    names = [root.name + "/" + rel for _, rel in inputs]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate archive member")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for (source, _), name in zip(inputs, names):
            archive.write(source, name)
    with zipfile.ZipFile(destination) as archive:
        if archive.namelist() != names or archive.testzip() is not None:
            raise ValueError("Archive validation failed")
    return len(inputs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh main ZIP; excludes all other ZIPs")
    parser.add_argument("--results-output", type=Path, help="Optional separate results/examples ZIP, never nested in main")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    print("Main archive files:", build_zip(root, args.output))
    if args.results_output:
        print("Optional results archive files:", build_zip(root, args.results_output, results_only=True))


if __name__ == "__main__":
    main()
