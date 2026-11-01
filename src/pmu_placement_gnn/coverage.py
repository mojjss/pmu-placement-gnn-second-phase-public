"""Direct physical one-hop coverage; no two-hop propagation or solver."""

def observed_nodes(edges, n, placement):
    selected = set(placement)
    if len(selected) != len(placement) or any(i < 0 or i >= n for i in selected):
        raise ValueError("Invalid or duplicate PMU indices")
    observed = set(selected)
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError("Invalid branch endpoint")
        if u in selected:
            observed.add(v)
        if v in selected:
            observed.add(u)
    return observed


def coverage(edges, n, placement):
    return len(observed_nodes(edges, n, placement)) / n
