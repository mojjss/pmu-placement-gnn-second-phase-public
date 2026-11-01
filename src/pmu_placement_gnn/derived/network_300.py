"""Portable selected method definitions; no historical module top levels."""
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
device = torch.device("cpu")




def _neighbors_list(edge_index, N):
    neighbors = [[] for _ in range(N)]
    src = edge_index[0].detach().cpu().tolist()
    dst = edge_index[1].detach().cpu().tolist()
    for u, v in zip(src, dst):
        if u == v:
            continue
        neighbors[u].append(v)
        neighbors[v].append(u)
    return neighbors

def coverage_penalty(batch, logits, lambda_cov=1.0):
    """
    Penalize probability of being unobserved:
      P(i unobserved) ≈ prod_{j in {i ∪ N(i)}} (1 - p_j)

    IMPORTANT: compute per-graph then mean, so big graphs don't dominate batches.
    """
    probs = F.softmax(logits, dim=-1)[:, 1]   # [N_total]
    one_minus = 1.0 - probs

    N = probs.size(0)
    neighbors = _neighbors_list(batch.edge_index, N)

    per_node_unobs = []
    for i in range(N):
        idxs = [i] + neighbors[i]
        idxs_t = torch.tensor(idxs, dtype=torch.long, device=probs.device)
        per_node_unobs.append(one_minus[idxs_t].prod())
    per_node_unobs = torch.stack(per_node_unobs)  # [N_total]

    # per-graph mean (batch.ptr marks graph boundaries)
    losses = []
    for g in range(batch.num_graphs):
        s = int(batch.ptr[g].item())
        e = int(batch.ptr[g + 1].item())
        if e > s:
            losses.append(per_node_unobs[s:e].mean())
    if not losses:
        return torch.tensor(0.0, device=probs.device)

    return lambda_cov * torch.stack(losses).mean()

def diversity_penalty(batch, logits, lambda_div=0.2):
    """
    Penalize adjacent PMUs: mean_{(u,v) in edges} p_u * p_v
    This reduces PMU clustering (very important for IEEE300 coverage).
    """
    probs = F.softmax(logits, dim=-1)[:, 1]
    ei = batch.edge_index

    # remove self loops
    mask = ei[0] != ei[1]
    u = ei[0][mask]
    v = ei[1][mask]

    # count each undirected edge once
    keep = u < v
    u = u[keep]
    v = v[keep]

    if u.numel() == 0:
        return torch.tensor(0.0, device=probs.device)

    return lambda_div * (probs[u] * probs[v]).mean()

def pmu_rate_penalty(batch, logits, lambda_rate=0.05):
    """
    Match average predicted PMU probability per-graph to label density.
    Improves calibration and helps ranking selection.
    """
    probs = F.softmax(logits, dim=-1)[:, 1]

    losses = []
    for g in range(batch.num_graphs):
        s = int(batch.ptr[g].item())
        e = int(batch.ptr[g + 1].item())
        if e <= s:
            continue
        target = batch.y[s:e].float().mean()
        pred   = probs[s:e].mean()
        losses.append((pred - target).pow(2))

    if not losses:
        return torch.tensor(0.0, device=probs.device)

    return lambda_rate * torch.stack(losses).mean()

class PMUGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin   = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.lin(x)  # logits per node
        return x

def train_one_epoch(
    model, loader, optimizer, criterion, device,
    lambda_cov=1.0,
    lambda_div=0.2,
    lambda_rate=0.05
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index)

        ce_loss   = criterion(out, batch.y)
        cov_loss  = coverage_penalty(batch, out, lambda_cov=lambda_cov)
        div_loss  = diversity_penalty(batch, out, lambda_div=lambda_div)
        rate_loss = pmu_rate_penalty(batch, out, lambda_rate=lambda_rate)

        loss = ce_loss + cov_loss + div_loss + rate_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_nodes
        preds = out.argmax(dim=-1)
        total_correct += (preds == batch.y).sum().item()
        total_nodes += batch.num_nodes

    avg_loss = total_loss / max(total_nodes, 1)
    acc = total_correct / max(total_nodes, 1)
    return avg_loss, acc

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)

        # keep validation as CE only (stable comparison)
        loss = criterion(out, batch.y)

        total_loss += loss.item() * batch.num_nodes
        preds = out.argmax(dim=-1)
        total_correct += (preds == batch.y).sum().item()
        total_nodes += batch.num_nodes

    if total_nodes == 0:
        return 0.0, 0.0

    return total_loss / total_nodes, total_correct / total_nodes

def build_neighbors(edge_index, n):
    ei = edge_index.cpu().numpy()
    neigh = [set() for _ in range(n)]
    for u, v in ei.T:
        u = int(u); v = int(v)
        if u == v:
            continue
        neigh[u].add(v)
        neigh[v].add(u)
    return neigh

@torch.no_grad()
def gnn_select_pmus_coverage_aware(data, K, topM=None):
    """
    Select K nodes guided by GNN probabilities but maximizing coverage gain.
    - Compute probs
    - Restrict candidates to topM by prob (optional)
    - Greedy select by marginal observed-node gain; tie-break by prob
    """
    data = data.to(device)
    logits = model(data.x, data.edge_index)
    probs  = F.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()

    N = data.num_nodes
    neigh = build_neighbors(data.edge_index, N)

    if topM is None:
        candidates = list(range(N))
    else:
        topM = int(min(N, topM))
        candidates = np.argsort(-probs)[:topM].tolist()

    selected = []
    observed = set()

    for _ in range(int(K)):
        best = None
        best_gain = -1
        best_prob = -1.0

        for n in candidates:
            if n in selected:
                continue
            new_obs = {n} | neigh[n]
            gain = len(new_obs - observed)

            if (gain > best_gain) or (gain == best_gain and probs[n] > best_prob):
                best = n
                best_gain = gain
                best_prob = probs[n]

        if best is None:
            break

        selected.append(best)
        observed |= ({best} | neigh[best])

    return selected

def pyg_to_nx_graph(data):
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    ei = data.edge_index.cpu().numpy()
    for u, v in ei.T:
        u = int(u); v = int(v)
        if u != v:
            G.add_edge(u, v)
    return G

def observed_nodes_simple(G, pmus):
    obs = set(pmus)
    for n in pmus:
        obs.update(G.neighbors(n))
    return obs

def greedy_pmu_placement_simple(G):
    all_nodes = set(G.nodes())
    pmus = []
    observed = set()
    while observed != all_nodes:
        best_node, best_gain = None, -1
        for n in all_nodes - set(pmus):
            gain = len(observed_nodes_simple(G, [n]) - observed)
            if gain > best_gain:
                best_gain, best_node = gain, n
        pmus.append(best_node)
        observed.update(observed_nodes_simple(G, [best_node]))
    return pmus
