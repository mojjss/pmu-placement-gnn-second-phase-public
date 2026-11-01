# Method

The graph represents buses and active physical lines/transformers. Parallel physical branches are enumerated as outages separately; the historical simple graph collapses adjacency, and a remaining parallel branch preserves the connection. Coverage is the union of selected buses and their immediate neighbors, treated physically as undirected.

Raw node features are [nominal voltage, degree, has_load, has_generator, has_external_grid]. They are not scaled. Stored message-passing directions and repetitions are retained; reverse edges are not automatically inserted. Stored electrical edge attributes are not consumed by the plain GCN. GCN adjacency normalization and self-loops do not imply node-feature scaling.

The model has two GCN layers with 128 hidden channels and ReLU, followed by a two-class linear classifier. Softmax provides PMU probabilities. Training uses Adam, learning rate 1e-3, weight decay 1e-4, batch size 1, 150 epochs and one initialization. Cross-entropy weights use training-only counts [1,n0/n1]. The auxiliary coefficients are 0.3 coverage, 0.2 diversity and 0.05 density. Repeated incident entries remain in the unobserved-probability product; diversity keeps the original stored u<v arcs and multiplicity.

Seed-42 grouped splits assign approximately 60/20/20 of fault cases; the intact group is in training. Identical physical input groups cannot cross partitions. The first strictly improving maximum validation node accuracy selects the checkpoint. Validation loss is cross-entropy, as in the selected definitions.

K comes from stored Greedy labels or an explicit incremental-Greedy computation. The pool size is min(N,max(2K,80)). Selection compares gain first, then probability, then existing candidate order. It makes exactly K selections, without repair until full coverage. The teacher is heuristic, not a certified optimum.

Selected source definitions were extracted into ordinary importable modules under derived/. Scientific ASTs, decorators and signatures were retained, with explicit imports/global dependencies. Fresh module namespaces preserve per-model decoder globals. This transformation and the independent execution checks are recorded privately; static matching alone is not an execution-equivalence claim.

Portable drivers provide new path/output orchestration. Their executions produce new outputs, not retroactively authenticated historical results. Runtime driver overhead/order and figure export rendering can differ from historical execution. No first-phase normalization, symmetric connections, new architecture or feasibility repair is introduced.
