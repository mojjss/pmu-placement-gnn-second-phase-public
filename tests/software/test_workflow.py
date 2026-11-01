"""Test software behavior using synthetic inputs, not saved paper results."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest

import pmu_placement_gnn
from pmu_placement_gnn.artifacts import Catalog, new_output
from pmu_placement_gnn.cli import main, parser
from pmu_placement_gnn.coverage import observed_nodes

ROOT = Path(__file__).resolve().parents[2]


class SoftwareWorkflowTests(unittest.TestCase):
    def test_coverage_is_direct_and_undirected(self):
        edges = [(0, 1), (1, 2)]
        self.assertEqual(observed_nodes(edges, 3, [0]), {0, 1})
        self.assertEqual(observed_nodes(edges, 3, [2]), {1, 2})
        self.assertEqual(observed_nodes([(0, 1), (0, 1)], 3, [1]), {0, 1})

    def test_cardinality_does_not_imply_complete_coverage(self):
        self.assertEqual(observed_nodes([], 4, [0, 2]), {0, 2})
        with self.assertRaises(ValueError):
            observed_nodes([], 4, [0, 0])

    def test_output_directories_are_fresh_and_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for destination in ["data/new", "models/new", "src/new"]:
                with self.assertRaises(ValueError):
                    new_output(root, destination)
            output = new_output(root, "runs/new")
            self.assertTrue(output.is_dir())
            with self.assertRaises(FileExistsError):
                new_output(root, "runs/new")

    def test_catalog_rejects_traversal_with_synthetic_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "results/paper/manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"records": []}), encoding="utf-8")
            with self.assertRaises(ValueError):
                Catalog(root).path({"path": "../outside"})

    def test_runtime_budget_interface_and_execution_guard(self):
        for budget in ["stored", "computed"]:
            args = parser().parse_args([
                "benchmark", "--system", "14", "--checkpoint", "example.pt",
                "--budget", budget, "--output", "runs/example",
            ])
            self.assertEqual(args.budget, budget)
            self.assertFalse(args.confirm_new_experiment)

    def test_new_experiment_requires_explicit_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "results/paper/manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text('{"records": []}', encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(["--root", str(root), "train", "--system", "14",
                          "--output", "runs/not-authorized"])
            self.assertEqual(error.exception.code, 2)
            self.assertFalse((root / "runs/not-authorized").exists())

    def test_version_matches_package_metadata(self):
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(metadata["project"]["version"], pmu_placement_gnn.__version__)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as done:
                main(["--version"])
        self.assertEqual(done.exception.code, 0)
        self.assertEqual(output.getvalue().strip(),
                         "pmu-gnn " + pmu_placement_gnn.__version__)

    def test_cli_help_works_without_research_artifacts(self):
        # A code-only checkout: no data, models, results, or experiment configs.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = Path(pmu_placement_gnn.__file__).resolve().parent
            shutil.copytree(source, root / "src/pmu_placement_gnn",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            environment = dict(os.environ, PYTHONPATH=str(root / "src"),
                               PYTHONDONTWRITEBYTECODE="1")
            commands = next(action.choices for action in parser()._actions
                            if isinstance(getattr(action, "choices", None), dict))
            interfaces = [["--help"], ["--version"]]
            interfaces.extend([command, "--help"] for command in commands)
            for arguments in interfaces:
                with self.subTest(arguments=arguments):
                    result = subprocess.run(
                        [sys.executable, "-B", "-m", "pmu_placement_gnn", *arguments],
                        cwd=root, env=environment, capture_output=True, text=True,
                        timeout=20,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
            for name in ["data", "models", "results", "runs"]:
                self.assertFalse((root / name).exists())

    def test_invalid_cli_arguments_remain_errors(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(["--not-a-valid-option"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
