import random
import math
import torch


class PathfindPEModel:
    """Geometry-first IAC pathfinding with positional encoding.

    Every scenario and action gets a position vector in [0,1]^dim. The graph is
    generated FROM the geometry: each scenario connects to its k nearest
    actions, so vector similarity is ground truth for connectivity (a geometric
    random graph, Kleinberg-style). During simulation an optional guidance term
    biases each node's net input by its similarity to the clamped endpoints,
    turning blind IAC flooding into goal-directed (A*-like) spread.

    Deterministic per seed, so the bridge stays stateless.
    """

    def __init__(self, num_scenarios, num_actions, dim=2, k=3, seed=42,
                 weight_mode='uniform', excitatory=0.1, inhibitory=-0.05,
                 decay=0.05, rectify=True, inhibition_mode='sibling'):
        self.max_a = 1.0
        self.min_a = -0.2
        self.decay_w = decay
        self.rectify = rectify
        self.dim = dim

        rng = random.Random(seed)
        self.scenarios = [f"S{i + 1}" for i in range(num_scenarios)]
        self.actions = [f"A{j + 1}" for j in range(num_actions)]
        self.all_nodes = self.scenarios + self.actions
        self.idx_lookup = {name: i for i, name in enumerate(self.all_nodes)}
        self.n = len(self.all_nodes)

        self.positions = torch.tensor(
            [[rng.random() for _ in range(dim)] for _ in range(self.n)])
        # Max possible distance in the unit hypercube, for [0,1] similarity.
        self.max_dist = math.sqrt(dim)

        s_pos = self.positions[:num_scenarios]
        a_pos = self.positions[num_scenarios:]
        dists = torch.cdist(s_pos, a_pos)

        deg_a = [0] * num_actions
        edges = set()
        for s in range(num_scenarios):
            nearest = torch.argsort(dists[s])[:min(k, num_actions)]
            for a in nearest.tolist():
                edges.add((s, a))
                deg_a[a] += 1
        # Geometry can leave remote actions unpicked; attach them to their
        # nearest scenario so no node is isolated.
        for a in range(num_actions):
            if deg_a[a] == 0:
                edges.add((int(torch.argmin(dists[:, a])), a))

        self.weights = torch.zeros((self.n, self.n))
        self.connections = []
        for s, a in sorted(edges):
            sim = 1.0 - float(dists[s, a]) / self.max_dist
            if weight_mode == 'random':
                w = rng.uniform(0.2, 1.8) * excitatory
            elif weight_mode == 'geometric':
                w = excitatory * (0.5 + 1.5 * sim)  # closer pair -> stronger edge
            else:
                w = excitatory
            i = self.idx_lookup[self.scenarios[s]]
            j = self.idx_lookup[self.actions[a]]
            self.weights[i, j] = w
            self.weights[j, i] = w
            self.connections.append({
                "source": self.scenarios[s],
                "target": self.actions[a],
                "weight": round(w, 4),
                "similarity": round(sim, 4),
                "type": "excitatory",
            })

        # Sibling pools with their anchor (the shared neighbor defining the
        # pool). Used for sibling inhibition and for goal-directed inhibition.
        self.inhibitory_w = inhibitory
        self.inhibition_mode = inhibition_mode
        pools = {}
        for s, a in edges:
            a_idx = self.idx_lookup[self.actions[a]]
            s_idx = self.idx_lookup[self.scenarios[s]]
            pools.setdefault(('a', a_idx), []).append(s_idx)
            pools.setdefault(('s', s_idx), []).append(a_idx)
        self.pools = [(anchor, members) for (_, anchor), members in pools.items()]

        if inhibition_mode == 'sibling':
            for _, members in self.pools:
                for i in members:
                    for j in members:
                        if i != j:
                            self.weights[i, j] = inhibitory
        elif inhibition_mode == 'global':
            for pool in (self.scenarios, self.actions):
                idxs = [self.idx_lookup[name] for name in pool]
                for i in idxs:
                    for j in idxs:
                        if i != j:
                            self.weights[i, j] = inhibitory
        # 'directed' builds its inhibitory weights at run time, once the
        # goal endpoints are known.

        self.activations = torch.zeros(self.n)

    def _apply_directed_inhibition(self, goal_idxs, strength):
        """Directional inhibition: within each sibling pool, softmax over the
        members' goal proximity (x strength) decides who is EXEMPT from
        inhibition. Each member's incoming weight is scaled by
        (1 - alpha_m) * N/(N-1): at strength 0 this is exactly the uniform
        sibling baseline; at high strength the pool's best-placed member takes
        ~0x while the rest keep ~1x. Relative within the pool — the locally
        closest sibling wins even if it is globally far from the goals — and
        the pool's total inhibition budget is preserved at every strength."""
        field = torch.zeros(self.n)
        for gi in goal_idxs:
            d = torch.norm(self.positions - self.positions[gi], dim=1)
            field += 1.0 - d / self.max_dist
        field /= len(goal_idxs)
        # Within-pool proximity gaps are small (~0.1), so amplify before
        # softmax: strength 5 => e^5 odds ratio per 0.1 of proximity.
        temp = strength * 10.0
        for _, members in self.pools:
            n = len(members)
            if n < 2:
                continue
            scores = torch.tensor([float(field[m]) for m in members])
            alphas = torch.softmax(scores * temp, dim=0)
            for m, alpha in zip(members, alphas.tolist()):
                scale = (1.0 - alpha) * n / (n - 1)
                for j in members:
                    if j != m:
                        self.weights[m, j] = self.inhibitory_w * scale

    def similarity_field(self, clamped_nodes):
        """Mean similarity of every node to the clamped endpoints, in [0,1].

        Summing over both endpoints makes a ridge between them — the corridor
        prior that guidance pushes activation along.
        """
        idxs = [self.idx_lookup[n] for n in clamped_nodes if n in self.idx_lookup]
        if not idxs:
            return torch.zeros(self.n)
        field = torch.zeros(self.n)
        for i in idxs:
            d = torch.norm(self.positions - self.positions[i], dim=1)
            field += 1.0 - d / self.max_dist
        return field / len(idxs)

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

    def run(self, clamped_nodes, steps, dir_inhibition=1.0):
        goal_idxs = [self.idx_lookup[n] for n in clamped_nodes if n in self.idx_lookup]
        if self.inhibition_mode == 'directed' and goal_idxs:
            self._apply_directed_inhibition(goal_idxs, dir_inhibition)

        ext = torch.zeros(self.n)
        for name in clamped_nodes:
            if name in self.idx_lookup:
                ext[self.idx_lookup[name]] = 1.0

        history = []
        for _ in range(steps):
            state = self.step(ext)
            history.append(state.tolist())
        return history

    def network_info(self):
        return {
            "nodes": self.all_nodes,
            "scenarios": self.scenarios,
            "actions": self.actions,
            "node_types": {name: ("scenario" if name in self.scenarios else "action")
                           for name in self.all_nodes},
            "positions": {name: [round(v, 4) for v in self.positions[i].tolist()]
                          for name, i in self.idx_lookup.items()},
            "dim": self.dim,
            "connections": self.connections,
            "idx_lookup": self.idx_lookup,
        }
