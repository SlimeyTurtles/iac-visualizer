import random
import torch


class PathfindMLPModel:
    """Layered MLP-style IAC planner.

    Layers stack like an MLP: layer 0 is a single root node (the current
    state), layers 1..depth-1 have `width` nodes each. Every node is a world
    state; every adjacent-layer edge is an action/affordance whose weight is
    the ease of taking it. Edges exist with probability conn_p and get an
    affordance ~ Normal(w_mean, w_sd) clipped to [w_lo, w_hi]. Weights are
    symmetric, so clamping the root and a goal node in the top layer lets
    plain IAC dynamics meet in the middle.

    The plan readout is a greedy connected ascent: from the root, at each
    layer pick the highest-activation node among those connected to the
    current one. Ground truth (reachability, easiest path = max mean
    affordance) comes from exact DP over the layered DAG.

    Deterministic per seed; lesions do not consume rng draws, so the same
    seed + lesions is the same network minus the lesioned nodes.
    """

    def __init__(self, width, depth, conn_p, seed=42,
                 w_mean=0.5, w_sd=0.25, w_lo=0.0, w_hi=1.0,
                 excitatory=0.1, inhibitory=-0.05, decay=0.05,
                 rectify=True, lesions=None):
        self.max_a = 1.0
        self.min_a = -0.2
        self.decay_w = decay
        self.rectify = rectify
        self.depth = depth
        self.width = width

        self.layers = []
        self.names = []
        for l in range(depth):
            count = 1 if l == 0 else width
            layer = [f"L{l}N{i + 1}" for i in range(count)]
            self.layers.append(layer)
            self.names.extend(layer)
        self.idx = {name: i for i, name in enumerate(self.names)}
        self.n = len(self.names)
        self.root = self.layers[0][0]
        self.lesions = set(lesions or []) & set(self.names)
        self.lesions.discard(self.root)

        rng = random.Random(seed)
        self.afford = torch.zeros((self.n, self.n))
        self.adj = torch.zeros((self.n, self.n), dtype=torch.bool)
        self.connections = []
        for l in range(depth - 1):
            for a in self.layers[l]:
                for b in self.layers[l + 1]:
                    if rng.random() < conn_p:
                        w = min(w_hi, max(w_lo, rng.gauss(w_mean, w_sd)))
                        i, j = self.idx[a], self.idx[b]
                        self.afford[i, j] = w
                        self.afford[j, i] = w
                        self.adj[i, j] = True
                        self.adj[j, i] = True
                        self.connections.append({
                            "source": a, "target": b,
                            "weight": round(w, 4), "type": "excitatory",
                        })

        self.weights = self.afford * excitatory
        for layer in self.layers:
            idxs = [self.idx[x] for x in layer]
            for i in idxs:
                for j in idxs:
                    if i != j:
                        self.weights[i, j] = inhibitory
        for name in self.lesions:
            i = self.idx[name]
            self.weights[i, :] = 0.0
            self.weights[:, i] = 0.0

        self.activations = torch.zeros(self.n)

    def _alive(self, name):
        return name not in self.lesions

    def _connected(self, a, b):
        return bool(self.adj[self.idx[a], self.idx[b]])

    def step(self, external_input_vector):
        output = torch.clamp(self.activations, min=0.0) if self.rectify else self.activations
        net_input = torch.mv(self.weights, output) + external_input_vector

        pos_mask = (net_input > 0).float()
        neg_mask = (net_input <= 0).float()

        delta_pos = net_input * (self.max_a - self.activations) * pos_mask
        delta_neg = net_input * (self.activations - self.min_a) * neg_mask
        decay_term = self.decay_w * self.activations

        self.activations = self.activations + delta_pos + delta_neg - decay_term
        self.activations = torch.clamp(self.activations, self.min_a, self.max_a)
        return self.activations

    def run(self, goal, steps):
        ext = torch.zeros(self.n)
        ext[self.idx[self.root]] = 1.0
        if goal in self.idx and self._alive(goal):
            ext[self.idx[goal]] = 1.0
        history = []
        for _ in range(steps):
            state = self.step(ext)
            history.append([round(v, 5) for v in state.tolist()])
        return history

    # ---- ground truth over the layered DAG (activation-independent) ----

    def static_analysis(self, goal):
        """Reachability, easiest path (max mean affordance) and path count
        from root to goal, by exact DP. Lesioned nodes are excluded."""
        best = {self.root: 0.0} if self._alive(self.root) else {}
        prev = {}
        counts = {self.root: 1} if self._alive(self.root) else {}
        for l in range(1, self.depth):
            nbest, ncounts = {}, {}
            for b in self.layers[l]:
                if not self._alive(b):
                    continue
                cands = []
                total = 0
                for a in self.layers[l - 1]:
                    if a in best and self._connected(a, b):
                        w = float(self.afford[self.idx[a], self.idx[b]])
                        cands.append((best[a] + w, a))
                        total += counts[a]
                if cands:
                    s, a = max(cands)
                    nbest[b] = s
                    prev[b] = a
                    ncounts[b] = total
            best, counts = nbest, ncounts

        reachable = goal in best
        optimal_path, optimal_mean, path_count = None, None, 0
        if reachable:
            path = [goal]
            while path[-1] != self.root:
                path.append(prev[path[-1]])
            optimal_path = path[::-1]
            optimal_mean = best[goal] / (self.depth - 1)
            path_count = counts[goal]
        return {
            "reachable": reachable,
            "optimal_path": optimal_path,
            "optimal_mean": round(optimal_mean, 5) if optimal_mean is not None else None,
            "_optimal_mean_raw": optimal_mean,
            "path_count": path_count,
        }

    # ---- activation readout ----

    def greedy_path(self, acts):
        """Connected ascent: from the root, pick the highest-activation node
        in the next layer among those connected to the current node."""
        path = [self.root]
        for l in range(1, self.depth):
            cur = path[-1]
            cands = [b for b in self.layers[l]
                     if self._alive(b) and self._connected(cur, b)]
            if not cands:
                return path, True  # dead end
            path.append(max(cands, key=lambda b: acts[self.idx[b]]))
        return path, False

    def analyze(self, goal, acts):
        static = self.static_analysis(goal)
        path, dead_end = self.greedy_path(acts)

        found_mean = None
        if len(path) > 1:
            ws = [float(self.afford[self.idx[path[i]], self.idx[path[i + 1]]])
                  for i in range(len(path) - 1)]
            found_mean = sum(ws) / len(ws)

        # Unconstrained per-layer argmax — does the greedy path coincide with
        # the globally most-active node in every layer?
        chain = []
        for l in range(self.depth):
            cands = [b for b in self.layers[l] if self._alive(b)]
            chain.append(max(cands, key=lambda b: acts[self.idx[b]]) if cands else None)
        argmax_match = (not dead_end) and chain == path
        links = [1 if chain[l] and chain[l + 1] and self._connected(chain[l], chain[l + 1]) else 0
                 for l in range(self.depth - 1)]
        argmax_coherence = sum(links) / len(links)

        if not static["reachable"]:
            outcome = "unreachable"
        elif dead_end:
            outcome = "dead_end"
        elif path[-1] != goal:
            outcome = "wrong_top"
        else:
            outcome = "success"

        efficiency = None
        if outcome == "success":
            opt = static.pop("_optimal_mean_raw")
            efficiency = round(found_mean / opt, 5) if opt and opt > 0 else 1.0
        else:
            static.pop("_optimal_mean_raw", None)

        return {
            **static,
            "outcome": outcome,
            "found_path": path,
            "dead_end": dead_end,
            "found_mean": round(found_mean, 5) if found_mean is not None else None,
            "efficiency": efficiency,
            "argmax_chain": chain,
            "argmax_match": argmax_match,
            "argmax_coherence": round(argmax_coherence, 4),
        }

    def network_info(self):
        return {
            "nodes": self.names,
            "layers": self.layers,
            "root": self.root,
            "top_layer": self.layers[-1],
            "node_layer": {name: l for l, layer in enumerate(self.layers) for name in layer},
            "connections": self.connections,
            "lesions": sorted(self.lesions),
            "idx_lookup": self.idx,
        }
