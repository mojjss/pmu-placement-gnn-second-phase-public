"""Portable selected method definitions; no historical module top levels."""
import random, copy, csv, time
import hashlib, json
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from ..artifacts import sha, save_json as save
CONFIG = {}




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

def train_and_select(env, samples, report, output):
    """Only training/validation graphs can be loaded before checkpoint finalization."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    parts = report['partitions']
    model = env['PMUGCN'](samples[parts['train'][0]].x.shape[1], 128, 2)
    env['model'] = model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(report['class_weights'], dtype=torch.float32))
    train_loader = DataLoader([samples[i] for i in parts['train']], batch_size=1, shuffle=True)
    val_loader = DataLoader([samples[i] for i in parts['validation']], batch_size=1, shuffle=False)
    best, state, best_epoch = 0.0, None, None
    start = time.perf_counter()
    with (output / 'history.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_node_accuracy', 'validation_loss',
                         'validation_node_accuracy', 'elapsed_seconds'])
        for epoch in range(1, 151):
            loss, acc = env['train_one_epoch'](model, train_loader, optimizer, criterion,
                                             torch.device('cpu'), lambda_cov=0.3)
            vl, va = env['evaluate'](model, val_loader, criterion, torch.device('cpu'))
            writer.writerow([epoch, loss, acc, vl, va, time.perf_counter() - start])
            f.flush()
            if va > best:
                best, state, best_epoch = va, copy.deepcopy(model.state_dict()), epoch
            print(f'IEEE{report["system"]} epoch={epoch} validation_accuracy={va:.6f}', flush=True)
    if state is None:
        raise RuntimeError('No strictly improving validation checkpoint; test remains sealed')
    model.load_state_dict(state)
    torch.save(dict(state_dict=state, in_channels=model.conv1.in_channels, hidden_channels=128,
                    out_channels=2, configuration=CONFIG, epoch=best_epoch,
                    validation_node_accuracy=best, split_sha256=sha(output / 'split.json')),
               output / 'checkpoint.pt')
    save(output / 'selection.json', dict(epochs_finished=150, selected_epoch=best_epoch,
         validation_node_accuracy=best, checkpoint_sha256=sha(output / 'checkpoint.pt')))
    model.eval()

def test_selected(env, samples, rows, report, output):
    if not (output / 'selection.json').is_file():
        raise RuntimeError('Test requires completed training and selection')
    records = []
    with torch.no_grad():
        for i in report['partitions']['test']:
            data = samples[i]
            logits = env['model'](data.x, data.edge_index)
            probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
            k = int((data.y == 1).sum())
            selected = env['gnn_select_pmus_coverage_aware'](data, k, topM=max(2*k, 80))
            if len(selected) != k or len(set(selected)) != k:
                raise ValueError('Decoder count mismatch')
            observed = set(selected)
            for u, v in data.edge_index.T.tolist():
                if u in selected:
                    observed.add(v)
                if v in selected:
                    observed.add(u)
            records.append(dict(index=i, sample_id=rows[i]['sample_id'], reference_K=k,
                                probabilities=probs.tolist(), placement=selected,
                                node_accuracy=float((logits.argmax(dim=-1) == data.y).float().mean()),
                                coverage=len(observed)/data.num_nodes))
    save(output / 'test_results.json', records)
