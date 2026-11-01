"""Packaging and figure policy checks; no scientific experiments."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile
from pmu_placement_gnn.artifacts import Catalog, project_root
from pmu_placement_gnn.outputs import figures

ROOT = project_root()
spec = importlib.util.spec_from_file_location("local_release_builder", ROOT / "scripts/build_release.py")
packer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packer)


class ReleasePolicy(unittest.TestCase):
    def test_cache_exclusions_keep_scientific_records(self):
        for path in [".venv/Lib/site.py", "venv/bin/python", "src/__pycache__/x.pyc", "x.egg-info/PKG-INFO", "dist/a.zip", ".env", "x.zip"]:
            self.assertTrue(packer.excluded(path), path)
        for path in ["data/IEEE14/samples/intact.npz", "models/held_out/IEEE14/checkpoint.pt", "results/paper/runtime/raw-paired.csv", "results/paper/raw.log", ".env.example", "README.md"]:
            self.assertFalse(packer.excluded(path), path)

    def test_single_root_without_nested_results_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            (root / "results").mkdir(parents=True)
            (root / "README.md").write_text("test")
            (root / "results/records.csv").write_text("a\n1\n")
            (root / "results.zip").write_bytes(b"excluded bundle")
            (root / ".venv").mkdir()
            (root / ".venv/private.txt").write_text("excluded environment")
            output = Path(directory) / "main.zip"
            self.assertEqual(packer.build_zip(root, output), 2)
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(set(archive.namelist()), {"project/README.md", "project/results/records.csv"})
            with self.assertRaises(FileExistsError):
                packer.build_zip(root, output)

    def test_required_pngs_and_optional_archive_formats(self):
        catalog = Catalog(ROOT)
        expected = set(catalog.manifest["required_previews"])
        self.assertEqual(len(expected), 4)
        self.assertEqual({str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "results/paper/figures").rglob("*.png")}, expected)
        self.assertFalse(list((ROOT / "results/paper/figures").rglob("*.pdf")))
        self.assertFalse(list((ROOT / "results/paper/figures").rglob("*.emf")))
        self.assertEqual(len(catalog.manifest["archived_optional_exports"]), 8)
        for record in catalog.manifest["archived_optional_exports"]:
            self.assertIn(Path(record["former_path"]).suffix.lower(), {".pdf", ".emf"})

    def test_copy_figure_panels_as_png_only(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = figures(ROOT, output, "3")
            self.assertEqual(result["files"], 2)
            self.assertEqual(sorted(p.suffix for p in output.iterdir()), [".json", ".png", ".png"])


if __name__ == "__main__":
    unittest.main()
