"""Portable selected method definitions; no historical module top levels."""
import copy
import numpy as np
import networkx as nx
from typing import List, Set




def build_graph(net, include_transformers=True, include_three_winding=True):
    G = nx.Graph()

    # Nodes = buses, with voltage feature
    for b in net.bus.index:
        G.add_node(int(b), voltage_level=float(net.bus.at[b, 'vn_kv']))

    # Lines as edges (with features)
    for idx, row in net.line.iterrows():
        if "in_service" in row and not bool(row["in_service"]):
            continue
        G.add_edge(
            int(row["from_bus"]),
            int(row["to_bus"]),
            kind="line",
            idx=int(idx),
            length=float(row.get("length_km", 0.0)),
            resistance=float(row.get("r_ohm_per_km", 0.0)),
        )

    # 2-winding transformers as edges (with features)
    if include_transformers and len(net.trafo) > 0:
        for idx, row in net.trafo.iterrows():
            if "in_service" in row and not bool(row["in_service"]):
                continue
            G.add_edge(
                int(row["hv_bus"]),
                int(row["lv_bus"]),
                kind="trafo",
                idx=int(idx),
                sn_mva=float(row.get("sn_mva", 0.0)),
                impedance=float(row.get("vk_percent", 0.0)),
            )

    return G

def observed_nodes(G: nx.Graph, pmu_nodes: List[int]) -> Set[int]:
    observed = set(pmu_nodes)
    for n in pmu_nodes:
        observed |= set(G.neighbors(n))
    return observed

def greedy_pmu_placement(G: nx.Graph) -> List[int]:
    all_nodes = set(G.nodes())
    observed = set()
    pmus = []
    while observed != all_nodes:
        best_bus, best_gain = None, -1
        for bus in all_nodes - set(pmus):
            gain = len(observed_nodes(G, [bus]) - observed)
            if gain > best_gain:
                best_gain, best_bus = gain, bus
        pmus.append(best_bus)
        observed |= observed_nodes(G, [best_bus])
    return pmus

def deactivate(net, kind: str, idx: int):
    """
    Deactivates a specific component (line or transformer) in the network.
    Returns a *new* net.
    """
    net2 = copy.deepcopy(net)
    if kind == "line":
        net2.line.at[idx, "in_service"] = False
    elif kind == "trafo" and len(net2.trafo) > 0:
        net2.trafo.at[idx, "in_service"] = False
    else:
        raise ValueError(f"Invalid kind {kind}, expected 'line' or 'trafo'")
    return net2

def failed_edge_buses(net, kind: str, idx: int):
    """Return (u, v) original bus IDs for the failed element."""
    if kind == "line":
        u = int(net.line.at[idx, "from_bus"])
        v = int(net.line.at[idx, "to_bus"])
        return (u, v)
    elif kind == "trafo" and len(net.trafo) > 0:
        u = int(net.trafo.at[idx, "hv_bus"])
        v = int(net.trafo.at[idx, "lv_bus"])
        return (u, v)
    else:
        return None

def graph_to_gnn_arrays(G: nx.Graph, net, pmu_list):
    """
    Convert a NetworkX graph + pandapower net + PMU list
    into numpy arrays suitable for GNN training.
    """
    # --- Node indexing: map original bus IDs -> [0..N-1]
    buses = sorted(G.nodes())  # original bus IDs
    N = len(buses)
    bus_to_idx = {b: i for i, b in enumerate(buses)}

    # --- Some pandapower-based info for node features
    load_buses = set(net.load["bus"].astype(int).tolist()) if len(net.load) > 0 else set()
    gen_buses  = set(net.gen["bus"].astype(int).tolist())  if len(net.gen)  > 0 else set()
    ext_buses  = set(net.ext_grid["bus"].astype(int).tolist()) if len(net.ext_grid) > 0 else set()

    # Node features: [voltage_level, degree, has_load, has_gen, has_ext_grid]
    F_node = 5
    x = np.zeros((N, F_node), dtype=np.float32)
    for b in buses:
        i = bus_to_idx[b]
        voltage = G.nodes[b].get("voltage_level", 0.0)
        degree  = float(G.degree[b])
        has_load = 1.0 if b in load_buses else 0.0
        has_gen  = 1.0 if b in gen_buses else 0.0
        has_ext  = 1.0 if b in ext_buses else 0.0

        x[i, 0] = voltage
        x[i, 1] = degree
        x[i, 2] = has_load
        x[i, 3] = has_gen
        x[i, 4] = has_ext

    # --- Edge index + edge features
    E = G.number_of_edges()
    edge_index = np.zeros((2, E), dtype=np.int64)

    # Edge features: [length_km, r_ohm_per_km, sn_mva, impedance, is_line, is_trafo]
    F_edge = 6
    edge_attr = np.zeros((E, F_edge), dtype=np.float32)

    for e_idx, (u, v, data) in enumerate(G.edges(data=True)):
        ui = bus_to_idx[u]
        vi = bus_to_idx[v]
        edge_index[0, e_idx] = ui
        edge_index[1, e_idx] = vi

        kind = data.get("kind", "line")
        if kind == "line":
            edge_attr[e_idx, 0] = float(data.get("length", 0.0))
            edge_attr[e_idx, 1] = float(data.get("resistance", 0.0))
            edge_attr[e_idx, 4] = 1.0  # is_line
        elif kind == "trafo":
            edge_attr[e_idx, 2] = float(data.get("sn_mva", 0.0))
            edge_attr[e_idx, 3] = float(data.get("impedance", 0.0))
            edge_attr[e_idx, 5] = 1.0  # is_trafo

    # --- Node labels y (PMU placement)
    y = np.zeros((N,), dtype=np.int64)
    pmu_set = set(int(p) for p in pmu_list)
    for b in buses:
        if b in pmu_set:
            y[bus_to_idx[b]] = 1

    bus_ids = np.array(buses, dtype=np.int64)

    return x, edge_index, edge_attr, y, bus_ids
