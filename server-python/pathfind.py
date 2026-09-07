import random
import torch


class PathfindIACModel:
    """IAC network over a random bipartite scenario/action graph.

    Scenarios only connect to actions and vice versa (excitatory).
    Nodes of the same type compete via within-pool inhibition.
    The whole network is regenerated deterministically from `seed`,
    so the bridge can stay stateless between requests.
    """

    def __init__(self, num_scenarios, num_actions, density=0.25, seed=42,
                 weight_mode='uniform', excitatory=0.1, inhibitory=-0.05,
                 decay=0.05, max_connections=0):
        self.max_a = 1.0
        self.min_a = -0.2
        self.decay_w = decay

        rng = random.Random(seed)
        self.scenarios = [f"S{i + 1}" for i in range(num_scenarios)]
        self.actions = [f"A{j + 1}" for j in range(num_actions)]
        self.all_nodes = self.scenarios + self.actions
        self.idx_lookup = {name: i for i, name in enumerate(self.all_nodes)}
        self.n = len(self.all_nodes)

        cap = max_connections if max_connections > 0 else None
        deg_s = [0] * num_scenarios
        deg_a = [0] * num_actions
        edges = set()
        # Random pair order so the degree cap doesn't systematically starve
        # high-numbered nodes.
        pairs = [(s, a) for s in range(num_scenarios) for a in range(num_actions)]
        rng.shuffle(pairs)
        for s, a in pairs:
            if cap and (deg_s[s] >= cap or deg_a[a] >= cap):
                continue
            if rng.random() < density:
                edges.add((s, a))
                deg_s[s] += 1
                deg_a[a] += 1

        # No isolated nodes: give every node at least one partner, preferring
        # under-cap partners (falls back to any partner if all are at cap).
        for s in range(num_scenarios):
            if deg_s[s] == 0:
                options = [a for a in range(num_actions)
                           if cap is None or deg_a[a] < cap] or list(range(num_actions))
                a = rng.choice(options)
                edges.add((s, a))
                deg_s[s] += 1
                deg_a[a] += 1
        for a in range(num_actions):
            if deg_a[a] == 0:
                options = [s for s in range(num_scenarios)
                           if cap is None or deg_s[s] < cap] or list(range(num_scenarios))
                s = rng.choice(options)
                edges.add((s, a))
                deg_s[s] += 1
                deg_a[a] += 1

        self.weights = torch.zeros((self.n, self.n))
        self.connections = []
        for s, a in sorted(edges):
            if weight_mode == 'random':
                w = rng.uniform(0.2, 1.8) * excitatory
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
                "type": "excitatory",
            })

        for pool in (self.scenarios, self.actions):
            idxs = [self.idx_lookup[name] for name in pool]
            for i in idxs:
                for j in idxs:
                    if i != j:
                        self.weights[i, j] = inhibitory

        self.activations = torch.zeros(self.n)

    def step(self, external_input_vector):
        net_input = torch.mv(self.weights, self.activations) + external_input_vector

        pos_mask = (net_input > 0).float()
        neg_mask = (net_input <= 0).float()

        delta_pos = net_input * (self.max_a - self.activations) * pos_mask
        delta_neg = net_input * (self.activations - self.min_a) * neg_mask
        decay_term = self.decay_w * self.activations

        self.activations = self.activations + delta_pos + delta_neg - decay_term
        self.activations = torch.clamp(self.activations, self.min_a, self.max_a)
        return self.activations

    def run(self, clamped_nodes, steps):
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
            "connections": self.connections,
            "idx_lookup": self.idx_lookup,
        }
