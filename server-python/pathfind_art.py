import random
import torch


def _cos(a, b):
    na, nb = float(torch.norm(a)), float(torch.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(torch.dot(a, b)) / (na * nb)


class ARTPathfinder:
    """ESCF-style recall-based planner.

    World: states are vectors in [0,1]^dim; transitions link each state to its
    k nearest neighbors (geometric, so vector similarity predicts reachability).

    Learning: states are presented as shuffled noisy samples to a flat ART
    layer. Cosine resonance >= vigilance updates the winning hub's template
    (EWMA); otherwise a new hub is recruited. Low vigilance merges nearby
    states into coarse hubs, so planning happens over abstractions.
    Similarity is cosine on CENTERED vectors (x - 0.5): dense positive vectors
    all crowd the positive orthant, so raw cosine can't separate states.

    Transition spokes: each hub links to the hubs of its member states'
    graph successors.

    Planning: best-first recursive search with backtracking. From the current
    hub, successors are scored by cosine(successor template, goal vector) and
    tried best-first; a global visited set prevents cycles. The full trace of
    visits, scorings, dead ends, and backtracks is recorded for the UI.
    """

    def __init__(self, num_states, dim=4, k=3, seed=42, vigilance=0.9,
                 lr=0.3, noise=0.03, samples=5):
        self.dim = dim
        rng = random.Random(seed)
        self.states = [f"S{i + 1}" for i in range(num_states)]
        self.positions = torch.tensor(
            [[rng.random() for _ in range(dim)] for _ in range(num_states)])

        dists = torch.cdist(self.positions, self.positions)
        dists.fill_diagonal_(float('inf'))
        edges = set()
        for i in range(num_states):
            for j in torch.argsort(dists[i])[:min(k, num_states - 1)].tolist():
                edges.add((min(i, j), max(i, j)))
        self.edges = sorted(edges)
        self.adjacency = {i: set() for i in range(num_states)}
        for i, j in self.edges:
            self.adjacency[i].add(j)
            self.adjacency[j].add(i)

        # --- ART training on shuffled noisy episodes ---
        self.templates = []
        presentations = [(s, si) for s in range(num_states) for si in range(samples)]
        rng.shuffle(presentations)
        for s, _ in presentations:
            x = torch.tensor([min(1.0, max(0.0, v + rng.gauss(0, noise)))
                              for v in self.positions[s].tolist()])
            best, best_sim = -1, -2.0
            for h, t in enumerate(self.templates):
                sim = _cos(x - 0.5, t - 0.5)
                if sim > best_sim:
                    best, best_sim = h, sim
            if best >= 0 and best_sim >= vigilance:
                self.templates[best] = (1 - lr) * self.templates[best] + lr * x
            else:
                self.templates.append(x.clone())

        # Final assignment: each state's clean vector -> most resonant hub.
        self.assign = []
        for s in range(num_states):
            sims = [_cos(self.positions[s] - 0.5, t - 0.5) for t in self.templates]
            self.assign.append(max(range(len(self.templates)), key=lambda h: sims[h]))

        # Hub-level transition spokes from state-level edges.
        self.hub_adj = {h: set() for h in range(len(self.templates))}
        for i, j in self.edges:
            hi, hj = self.assign[i], self.assign[j]
            if hi != hj:
                self.hub_adj[hi].add(hj)
                self.hub_adj[hj].add(hi)

    def search(self, start_state, goal_state, max_depth=20):
        s_idx = self.states.index(start_state)
        g_idx = self.states.index(goal_state)
        start_hub, goal_hub = self.assign[s_idx], self.assign[g_idx]
        goal_vec = self.positions[g_idx] - 0.5

        trace = []
        visited = set()
        path = []

        def hub_score(h):
            return round(_cos(self.templates[h] - 0.5, goal_vec), 4)

        def dfs(h, depth):
            visited.add(h)
            path.append(h)
            trace.append({"type": "visit", "hub": h, "depth": depth})
            if h == goal_hub:
                trace.append({"type": "goal", "hub": h})
                return True
            if depth >= max_depth:
                trace.append({"type": "depth_limit", "hub": h})
                path.pop()
                return False
            succs = [x for x in self.hub_adj[h] if x not in visited]
            scored = sorted(((x, hub_score(x)) for x in succs),
                            key=lambda p: -p[1])
            trace.append({"type": "expand", "hub": h,
                          "scores": [[x, sc] for x, sc in scored]})
            for x, sc in scored:
                if x in visited:
                    continue
                trace.append({"type": "move", "from": h, "to": x, "score": sc})
                if dfs(x, depth + 1):
                    return True
            trace.append({"type": "backtrack", "hub": h})
            path.pop()
            return False

        found = dfs(start_hub, 0)
        return {
            "found": found,
            "start_hub": start_hub,
            "goal_hub": goal_hub,
            "hub_path": path if found else [],
            "trace": trace,
        }

    def network_info(self):
        members = {h: [] for h in range(len(self.templates))}
        for s, h in enumerate(self.assign):
            members[h].append(self.states[s])
        return {
            "states": self.states,
            "positions": {self.states[i]: [round(v, 4) for v in self.positions[i].tolist()]
                          for i in range(len(self.states))},
            "dim": self.dim,
            "edges": [[self.states[i], self.states[j]] for i, j in self.edges],
            "num_hubs": len(self.templates),
            "assign": {self.states[i]: h for i, h in enumerate(self.assign)},
            "hub_members": members,
            "hub_templates": {h: [round(v, 4) for v in t.tolist()]
                              for h, t in enumerate(self.templates)},
            "hub_adj": {h: sorted(v) for h, v in self.hub_adj.items()},
        }
