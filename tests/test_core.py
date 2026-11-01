"""Small behavior checks; no experiments or scientific dependencies required."""
import json
import tempfile
import unittest
from pathlib import Path
from pmu_placement_gnn.coverage import observed_nodes
from pmu_placement_gnn.artifacts import new_output, Catalog
from pmu_placement_gnn.cli import parser, main


class CoreTests(unittest.TestCase):
    def test_physical_coverage_does_not_propagate(self):
        self.assertEqual(observed_nodes([(0,1),(1,2)],3,[0]),{0,1})
        self.assertEqual(observed_nodes([(0,1),(0,1),(1,0)],3,[1]),{0,1})

    def test_exact_cardinality_is_not_full_coverage(self):
        self.assertEqual(observed_nodes([],4,[0,2]),{0,2})
        with self.assertRaises(ValueError):
            observed_nodes([],4,[0,0])

    def test_output_isolation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaises(ValueError):
                new_output(root,"data/new")
            result=new_output(root,"runs/new")
            self.assertTrue(result.is_dir())
            with self.assertRaises(FileExistsError):
                new_output(root,"runs/new")

    def test_catalog_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/"results/paper").mkdir(parents=True)
            (root/"results/paper/manifest.json").write_text('{"records": []}')
            with self.assertRaises(ValueError):
                Catalog(root).path({"path":"../outside"})

    def test_cli_has_explicit_runtime_budget(self):
        args=parser().parse_args(["benchmark","--system","14","--checkpoint","model.pt","--budget","computed","--output","runs/test"])
        self.assertEqual(args.budget,"computed")
        self.assertFalse(args.confirm_new_experiment)

    def test_example_origin_and_saved_result(self):
        from pmu_placement_gnn.artifacts import project_root
        from pmu_placement_gnn.outputs import example
        result=example(project_root(),"IEEE14/line_0_0")
        self.assertTrue(result["paper_dataset"])
        self.assertFalse(result["unseen_claim"])
        self.assertGreater(result["stored_greedy"]["K"],0)

if __name__ == "__main__":
    unittest.main()
