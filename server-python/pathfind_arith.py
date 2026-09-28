import random
from collections import deque

import torch


OPS = [
    ('A+=B', lambda a, b, c: (a + b, b, c)),
    ('A-=B', lambda a, b, c: (a - b, b, c)),
    ('A+=C', lambda a, b, c: (a + c, b, c)),
    ('A-=C', lambda a, b, c: (a - c, b, c)),
    ('B+=A', lambda a, b, c: (a, b + a, c)),
    ('B-=A', lambda a, b, c: (a, b - a, c)),
    ('B+=C', lambda a, b, c: (a, b + c, c)),
    ('B-=C', lambda a, b, c: (a, b - c, c)),
    ('C+=A', lambda a, b, c: (a, b, c + a)),
    ('C-=A', lambda a, b, c: (a, b, c - a)),
    ('C+=B', lambda a, b, c: (a, b, c + b)),
    ('C-=B', lambda a, b, c: (a, b, c - b)),
    ('A=0', lambda a, b, c: (0, b, c)),
    ('B=0', lambda a, b, c: (a, 0, c)),
    ('C=0', lambda a, b, c: (a, b, 0)),
]


class PathfindArithModel:
    """IAC network over the arithmetic state lattice (a, b, c).

    Nodes are every integer triple in [0,a_max]x[0,b_max]x[0,c_max].
    Directed edges are the legal single-step arithmetic moves (X+=Y, X-=Y
    for every ordered variable pair, plus X=0). Moves that would leave the
    domain simply don't exist. Activation spreads symmetrically (recall
    works both ways: "what leads here" and "where can I go"), but the plan
    readout only walks *forward* along directed edges.

    Hebbian learning: after each settle, every connected pair potentiates
    in proportion to the product of its rectified activations, with a soft
    cap at w_max. Learned weight is kept separate from the base weight so
    it can be exported/imported as a sparse list — the bridge stays
    stateless while the page carries the learned state between calls.
    """

    def __init__(self, a_max=10, b_max=10, c_max=5, excitatory=0.05,
                 inhibitory=-0.001, decay=0.1, w_max=0.3, eta=0.02,
                 norm=1.0, rectify=True, inh_mode='global'):
        self.a_max, self.b_max, self.c_max = a_max, b_max, c_max
        self.excitatory = excitatory
        self.inhibitory = inhibitory
        self.decay_w = decay
        self.w_max = w_max
        self.eta = eta
        self.norm = norm
        self.rectify = rectify
        self.max_a = 1.0
        self.min_a = -0.2

        self.states = [(a, b, c)
                       for a in range(a_max + 1)
                       for b in range(b_max + 1)
                       for c in range(c_max + 1)]
        self.idx = {s: i for i, s in enumerate(self.states)}
        self.n = len(self.states)

        # Directed edges: edge_ops[(i, j)] = list of op labels producing j from i.
        self.edge_ops = {}
        self.out_edges = [[] for _ in range(self.n)]
        for i, (a, b, c) in enumerate(self.states):
            for label, fn in OPS:
                nxt = fn(a, b, c)
                if nxt == (a, b, c) or nxt not in self.idx:
                    continue
                j = self.idx[nxt]
                if (i, j) not in self.edge_ops:
                    self.edge_ops[(i, j)] = []
                    self.out_edges[i].append(j)
                self.edge_ops[(i, j)].append(label)

        # Undirected pair list for spreading + Hebbian.
        pairs = set()
        for (i, j) in self.edge_ops:
            pairs.add((min(i, j), max(i, j)))
        self.pairs = sorted(pairs)
        self.pair_i = torch.tensor([p[0] for p in self.pairs])
        self.pair_j = torch.tensor([p[1] for p in self.pairs])

        self.w_base = torch.zeros((self.n, self.n))
        self.w_base[self.pair_i, self.pair_j] = excitatory
        self.w_base[self.pair_j, self.pair_i] = excitatory
        self.dw = torch.zeros(len(self.pairs))
        self.activations = torch.zeros(self.n)

        # Per-node incident-weight budget for homeostatic synaptic scaling.
        deg = torch.zeros(self.n)
        deg.index_add_(0, self.pair_i, torch.ones(len(self.pairs)))
        deg.index_add_(0, self.pair_j, torch.ones(len(self.pairs)))
        self.budget = deg * excitatory

        # Sibling competition: the out-neighbors of each node are rival
        # "next steps" and mutually inhibit. Collapses the diffuse
        # proximity cloud into a ridge along the winning route, which also
        # makes co-activation Hebbian path-selective.
        self.inh_mode = inh_mode
        if inh_mode == 'sibling':
            sib = torch.zeros((self.n, self.n))
            for outs in self.out_edges:
                for j in outs:
                    for k in outs:
                        if j != k:
                            sib[j, k] = 1.0
            self.sib = sib

    # ---- learned-weight import/export (sparse, page-carried state) ----

    def load_learned(self, sparse):
        lookup = {p: k for k, p in enumerate(self.pairs)}
        for i, j, w in sparse or []:
            key = (min(int(i), int(j)), max(int(i), int(j)))
            if key in lookup:
                self.dw[lookup[key]] = float(w)

    def export_learned(self, eps=1e-6):
        mask = self.dw > eps
        idxs = mask.nonzero(as_tuple=True)[0].tolist()
        return [[self.pairs[k][0], self.pairs[k][1], round(float(self.dw[k]), 6)]
                for k in idxs]

    def _weight_matrix(self):
        w = self.w_base.clone()
        w[self.pair_i, self.pair_j] += self.dw
        w[self.pair_j, self.pair_i] += self.dw
        return w

    # ---- dynamics ----

    def settle(self, start, goal, steps, history_every=0):
        w = self._weight_matrix()
        ext = torch.zeros(self.n)
        ext[self.idx[start]] = 1.0
        ext[self.idx[goal]] = 1.0
        clamped = [self.idx[start], self.idx[goal]]
        self.activations = torch.zeros(self.n)
        history = []
        for t in range(steps):
            out = torch.clamp(self.activations, min=0.0) if self.rectify else self.activations
            if self.inh_mode == 'none':
                inh = 0.0
            elif self.inh_mode == 'sibling':
                # Clamped endpoints are externally given, not competing
                # hypotheses — they must not inhibit their pool rivals.
                # Otherwise the goal suppresses exactly the states that
                # feed into it and kills its own backward gradient.
                out_inh = out.clone()
                out_inh[clamped] = 0.0
                inh = self.inhibitory * torch.mv(self.sib, out_inh)
            else:
                # Global inhibition as a scalar: inh * (sum of everyone else's output).
                inh = self.inhibitory * (out.sum() - out)
            net = torch.mv(w, out) + inh + ext
            pos = (net > 0).float()
            delta = net * (self.max_a - self.activations) * pos \
                + net * (self.activations - self.min_a) * (1 - pos)
            self.activations = torch.clamp(
                self.activations + delta - self.decay_w * self.activations,
                self.min_a, self.max_a)
            if history_every and (t % history_every == 0 or t == steps - 1):
                history.append([round(v, 4) for v in self.activations.tolist()])
        return history

    def hebbian_update(self):
        out = torch.clamp(self.activations, min=0.0)
        cur = self.excitatory + self.dw
        gain = self.eta * out[self.pair_i] * out[self.pair_j] * \
            torch.clamp(self.w_max - cur, min=0.0)
        w = cur + gain
        if self.norm > 0:
            # Homeostatic synaptic scaling: hold each node's total incident
            # weight at its budget, so potentiating one edge implicitly
            # depresses the node's others. Without this, edges around the
            # always-shared goal region (b=0/c=0 plane) grow without bound
            # and flood the activation gradient. Geometric mean of the two
            # endpoints' scale factors approximates joint normalization.
            sums = torch.zeros(self.n)
            sums.index_add_(0, self.pair_i, w)
            sums.index_add_(0, self.pair_j, w)
            f = self.budget / torch.clamp(sums, min=1e-9)
            scale = torch.sqrt(f[self.pair_i] * f[self.pair_j])
            w = w * (1.0 - self.norm + self.norm * scale)
        self.dw = torch.clamp(w - self.excitatory,
                              -0.9 * self.excitatory, self.w_max - self.excitatory)

    # ---- readout + analysis ----

    def greedy_path(self, start, goal, max_steps=60):
        acts = self.activations
        si, gi = self.idx[start], self.idx[goal]
        path = [si]
        visited = {si}
        cur = si
        for _ in range(max_steps):
            if cur == gi:
                return path, 'success'
            # Never step into an absorbing state (e.g. (0,0,0)) unless it's
            # the goal — zeroing edges make it a black hole for the walker.
            options = [j for j in self.out_edges[cur] if j not in visited
                       and (j == gi or self.out_edges[j])]
            if not options:
                options = [j for j in self.out_edges[cur] if j not in visited]
            if not options:
                return path, 'dead_end'
            cur = max(options, key=lambda j: float(acts[j]))
            visited.add(cur)
            path.append(cur)
        return path, 'success' if cur == gi else 'timeout'

    def bfs_path(self, start, goal):
        si, gi = self.idx[start], self.idx[goal]
        prev = {si: None}
        q = deque([si])
        while q:
            cur = q.popleft()
            if cur == gi:
                path = []
                while cur is not None:
                    path.append(cur)
                    cur = prev[cur]
                return path[::-1]
            for j in self.out_edges[cur]:
                if j not in prev:
                    prev[j] = cur
                    q.append(j)
        return None

    def path_ops(self, path):
        ops = []
        for i, j in zip(path, path[1:]):
            ops.append(self.edge_ops.get((i, j), ['?'])[0])
        return ops

    def analyze(self, start, goal, found_path, outcome):
        optimal = self.bfs_path(start, goal)
        if optimal is None:
            return {"outcome": "unreachable", "optimal_len": None,
                    "found_len": None, "efficiency": None,
                    "found_path": found_path, "found_ops": [],
                    "optimal_path": None, "optimal_ops": []}
        found_len = len(found_path) - 1
        optimal_len = len(optimal) - 1
        eff = round(optimal_len / found_len, 4) if outcome == 'success' and found_len > 0 else 0.0
        if outcome == 'success' and found_len == 0:
            eff = 1.0
        return {"outcome": outcome,
                "optimal_len": optimal_len,
                "found_len": found_len if outcome == 'success' else None,
                "efficiency": eff,
                "found_path": found_path,
                "found_ops": self.path_ops(found_path),
                "optimal_path": optimal,
                "optimal_ops": self.path_ops(optimal)}

    def run_episode(self, start, goal, steps, learn=True, gate='always',
                    history_every=0, max_steps=60):
        history = self.settle(start, goal, steps, history_every)
        path, outcome = self.greedy_path(start, goal, max_steps)
        analysis = self.analyze(start, goal, path, outcome)
        if learn and (gate == 'always' or outcome == 'success'):
            self.hebbian_update()
        return history, analysis

    # ---- modulo curriculum ----

    def sample_task(self, rng, start_min=None, start_max=None):
        lo = start_min or (0, 0, 0)
        hi = start_max or (self.a_max, self.b_max, self.c_max)
        a_lo, a_hi = max(0, lo[0]), min(self.a_max, hi[0])
        b_lo, b_hi = max(0, lo[1]), min(self.b_max, hi[1])
        c_lo, c_hi = max(1, lo[2]), min(self.c_max, hi[2])  # C >= 1: mod 0 undefined
        while True:
            a = rng.randint(a_lo, a_hi)
            b = rng.randint(b_lo, b_hi)
            c = rng.randint(c_lo, c_hi)
            goal = ((a + b) % c, 0, 0)
            if goal in self.idx and goal != (a, b, c):
                return (a, b, c), goal

    def network_info(self):
        directed = len(self.edge_ops)
        degs = [len(o) for o in self.out_edges]
        return {
            "states": self.states,
            "aMax": self.a_max, "bMax": self.b_max, "cMax": self.c_max,
            "numNodes": self.n,
            "numDirectedEdges": directed,
            "numPairs": len(self.pairs),
            "maxOutDegree": max(degs),
            "minOutDegree": min(degs),
            "edges": [[i, j, self.edge_ops[(i, j)][0]] for (i, j) in sorted(self.edge_ops)],
        }
