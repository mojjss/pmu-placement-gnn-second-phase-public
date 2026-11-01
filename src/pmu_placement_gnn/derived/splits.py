"""Grouped split and class-weight definitions."""
import hashlib,json,random
import numpy as np

def input_hash(x, edges, buses):
    """Canonical physical-bus order; directed edge multiset, never labels/edge_attr."""
    x = np.asarray(x, dtype='<f4').copy()
    edges = np.asarray(edges, dtype='<i8')
    buses = np.asarray(buses, dtype='<i8')
    if x.ndim != 2 or buses.shape != (len(x),) or len(set(buses.tolist())) != len(x):
        raise ValueError('Invalid physical bus mapping')
    if edges.ndim != 2 or edges.shape[0] != 2 or not np.isfinite(x).all():
        raise ValueError('Invalid inputs')
    if edges.size and (edges.min() < 0 or edges.max() >= len(x)):
        raise ValueError('Invalid edge endpoint')
    order = np.argsort(buses)
    physical = buses[edges].T
    physical = physical[np.lexsort((physical[:, 1], physical[:, 0]))]
    x[x == 0] = 0  # canonical signed zero
    h = hashlib.sha256()
    for a in (buses[order], x[order], physical):
        h.update(json.dumps(list(a.shape)).encode() + b'\0')
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()

def grouped_split(rows, hashes):
    if len(rows) != len(hashes) or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Invalid index')
    groups = {}
    for i, h in enumerate(hashes):
        if rows[i]['scenario_type'] not in ('intact', 'faulted'):
            raise ValueError('Unknown scenario type')
        groups.setdefault(h, []).append(i)
    intact = {hashes[i] for i, r in enumerate(rows) if r['scenario_type'] == 'intact'}
    if len(intact) != 1:
        raise ValueError('Expected one intact input group')
    eligible = sorted(set(groups) - intact)
    random.Random(42).shuffle(eligible)
    if len(eligible) < 3:
        raise ValueError('Insufficient independent groups for three partitions')
    target = round(0.2 * sum(r['scenario_type'] == 'faulted' for r in rows))
    # Nearest prefix to 20% of ALL faults, reserving one group for each later set.
    # Group identity and sizes only; no labels or performance enter membership.
    parts = {}
    for name, reserve in [('validation', 2), ('test', 1)]:
        cumulative = np.cumsum([len(groups[h]) for h in eligible])
        cut = min(range(1, len(eligible) - reserve + 1),
                  key=lambda k: (abs(int(cumulative[k - 1]) - target), k))
        parts[name] = sorted(i for h in eligible[:cut] for i in groups[h])
        eligible = eligible[cut:]
    parts['train'] = sorted(i for h in list(intact) + eligible for i in groups[h])
    verify_split(rows, hashes, parts)
    return parts

def verify_split(rows, hashes, parts):
    flat = [i for indices in parts.values() for i in indices]
    if sorted(flat) != list(range(len(rows))) or any(not v for v in parts.values()):
        raise ValueError('Incomplete, repeated, or empty assignment')
    owner = {}
    for part, indices in parts.items():
        for i in indices:
            if hashes[i] in owner and owner[hashes[i]] != part:
                raise ValueError('Cross-partition input overlap')
            owner[hashes[i]] = part
    for i, row in enumerate(rows):
        if row['scenario_type'] == 'intact' and owner[hashes[i]] != 'train':
            raise ValueError('Intact group outside training')

def training_weights(read_labels, indices):
    counts = np.zeros(2, dtype=np.int64)
    for i in indices:
        y = np.asarray(read_labels(i))
        if y.ndim != 1 or not np.isin(y, [0, 1]).all():
            raise ValueError('Invalid labels')
        counts += np.bincount(y.astype(np.int64), minlength=2)
    if np.any(counts == 0):
        raise ValueError('Training needs both classes')
    return counts.tolist(), [1.0, float(counts[0] / counts[1])]
