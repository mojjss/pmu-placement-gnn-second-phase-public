"""Optional small synthetic tests; skip if graph/ML dependencies are unavailable."""
import importlib.util
import unittest

@unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("networkx"), "graph extras unavailable")
class GraphContract(unittest.TestCase):
    def test_greedy_reference_and_parallel_adjacency(self):
        import networkx as nx
        from pmu_placement_gnn.method import graph_definitions
        env=graph_definitions(); g=nx.Graph(); g.add_nodes_from(range(4));g.add_edges_from([(0,1),(1,2)])
        selected=env["greedy_pmu_placement"](g)
        self.assertEqual(env["observed_nodes"](g,selected),set(range(4)))
        self.assertIn(3,selected)

    def test_input_groups_preserve_orientation_and_multiplicity(self):
        import numpy as np
        from pmu_placement_gnn.method import split_definitions
        f=split_definitions()["input_hash"]
        x=np.zeros((3,5),np.float32);b=np.arange(3);e=np.array([[0,1],[1,2]])
        self.assertEqual(f(x,e,b),f(x,e[:,::-1],b))
        self.assertNotEqual(f(x,e,b),f(x,e[::-1],b))
        self.assertNotEqual(f(x,e,b),f(x,np.concatenate([e,e[:,:1]],axis=1),b))

    def test_decoder_stops_at_K_and_gain_precedes_probability(self):
        import numpy as np
        from pmu_placement_gnn.method import optimized_definitions
        select=optimized_definitions()["incremental"]
        # Node1 covers all three;node0 has higher probability but smaller gain.
        self.assertEqual(select(np.array([.9,.5,.1]),np.array([[0,1],[1,2]]),1),[1])
        self.assertEqual(len(select(np.array([.9,.5,.1]),np.array([[0,1],[1,2]]),3)),3)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torch_geometric"), "ML extras unavailable")
class LossContract(unittest.TestCase):
    def test_repeated_neighbors_preserved_in_coverage_loss(self):
        import torch
        from torch_geometric.data import Data, Batch
        from pmu_placement_gnn.method import model_definitions
        env=model_definitions("14")
        data=Batch.from_data_list([Data(x=torch.zeros((2,5)), edge_index=torch.tensor([[0,0],[1,1]]), y=torch.tensor([0,1]))])
        logits=torch.zeros((2,2))
        # With p=.5, self and two repeated incident entries give(.5)^3 per node.
        self.assertAlmostEqual(float(env["coverage_penalty"](data,logits,lambda_cov=1.0)),.125)
        self.assertAlmostEqual(float(env["diversity_penalty"](data,logits,lambda_div=1.0)),.25)
        self.assertAlmostEqual(float(env["pmu_rate_penalty"](data,logits,lambda_rate=1.0)),0.0)
