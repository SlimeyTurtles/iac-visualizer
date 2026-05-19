import torch
import torch.nn.functional as F


class UnsupervisedConceptModel:
    def __init__(
        self,
        pool_sizes,
        vigilance,
        learning_rate=0.1,
        gain=1.0,
        inhibition_strength=0.2,
        device="cpu",
    ):
        """
        pool_sizes: list like [20, 20, 20]
        vigilance: either float or list of per-pool thresholds
        """
        self.pool_sizes = pool_sizes
        self.total_dim = sum(pool_sizes)
        self.learning_rate = learning_rate
        self.gain = gain
        self.inhibition_strength = inhibition_strength
        self.device = device

        if isinstance(vigilance, (float, int)):
            self.vigilance = torch.tensor(
                [float(vigilance)] * len(pool_sizes),
                dtype=torch.float32,
                device=device,
            )
        else:
            self.vigilance = torch.tensor(
                vigilance,
                dtype=torch.float32,
                device=device,
            )

        # Hub templates: shape [num_hubs, total_dim]
        self.templates = torch.empty((0, self.total_dim), dtype=torch.float32, device=device)

        # Precompute slice boundaries for pools
        self.pool_slices = []
        start = 0
        for size in pool_sizes:
            end = start + size
            self.pool_slices.append((start, end))
            start = end

    def _normalize(self, x):
        return F.normalize(x, p=2, dim=-1, eps=1e-8)

    def _ensure_1d(self, x):
        if x.dim() != 1 or x.shape[0] != self.total_dim:
            raise ValueError(f"Expected input of shape [{self.total_dim}], got {tuple(x.shape)}")
        return x.to(self.device).float()

    def recruit_hub(self, x):
        """
        Create a new hub initialized to the current episode pattern.
        """
        x = self._normalize(x.unsqueeze(0)).squeeze(0)
        self.templates = torch.cat([self.templates, x.unsqueeze(0)], dim=0)
        return self.templates.shape[0] - 1

    def compute_bottom_up_drive(self, x):
        """
        Recognition drive = dot product between input and all hub templates.
        Since templates are normalized, this is similarity-like.
        """
        if self.templates.shape[0] == 0:
            return torch.empty(0, device=self.device)

        x = self._normalize(x.unsqueeze(0)).squeeze(0)
        drive = self.gain * torch.matmul(self.templates, x)  # [num_hubs]
        return drive

    def settle_competition(self, x, num_steps=20):
        """
        Simple winner-take-all style competition with lateral inhibition.
        Not biologically exact, but matches the paper's intuition.
        """
        drive = self.compute_bottom_up_drive(x)

        if drive.numel() == 0:
            return None, None

        acts = torch.zeros_like(drive)

        for _ in range(num_steps):
            total_other = acts.sum() - acts
            net = drive - self.inhibition_strength * total_other
            acts = torch.relu(net)

        winner = torch.argmax(acts).item()
        return winner, acts

    def predict_from_hub(self, hub_index):
        """
        Winning hub writes expected pattern into expectation layers.
        Since recognition and prediction share the same template,
        expectation is just the hub template itself.
        """
        return self.templates[hub_index].clone()

    def compute_pool_similarities(self, x, y):
        """
        Per-pool cosine similarity between input x and predicted y.
        Returns tensor [num_pools].
        """
        sims = []

        for start, end in self.pool_slices:
            xp = x[start:end]
            yp = y[start:end]

            x_norm = torch.norm(xp, p=2)
            y_norm = torch.norm(yp, p=2)

            if x_norm < 1e-8 or y_norm < 1e-8:
                sim = torch.tensor(0.0, device=self.device)
            else:
                sim = torch.dot(xp, yp) / (x_norm * y_norm + 1e-8)

            sims.append(sim)

        return torch.stack(sims)

    def snap_or_recruit(self, x):
        """
        Process one episode:
        - competition
        - expectation
        - per-pool similarity
        - vigilance check
        - recruit or update
        """
        x = self._ensure_1d(x)

        # If no hubs exist yet, recruit first one
        if self.templates.shape[0] == 0:
            winner = self.recruit_hub(x)
            return {
                "action": "recruit",
                "winner": winner,
                "pool_similarities": None,
            }

        winner, activities = self.settle_competition(x)
        y = self.predict_from_hub(winner)
        pool_sims = self.compute_pool_similarities(x, y)

        passes = pool_sims >= self.vigilance

        if torch.all(passes):
            self.update_hub(winner, x)
            return {
                "action": "snap",
                "winner": winner,
                "pool_similarities": pool_sims.detach().cpu(),
                "activities": activities.detach().cpu(),
            }
        else:
            new_hub = self.recruit_hub(x)
            return {
                "action": "recruit",
                "winner": new_hub,
                "previous_winner": winner,
                "pool_similarities": pool_sims.detach().cpu(),
                "activities": activities.detach().cpu(),
            }

    def update_hub(self, hub_index, x):
        """
        Move winning hub toward current episode, then renormalize.
        """
        x = self._normalize(x.unsqueeze(0)).squeeze(0)
        old = self.templates[hub_index]
        new = (1.0 - self.learning_rate) * old + self.learning_rate * x
        self.templates[hub_index] = self._normalize(new.unsqueeze(0)).squeeze(0)

    def process_dataset(self, X):
        """
        X: tensor [num_samples, total_dim]
        """
        results = []
        for i in range(X.shape[0]):
            result = self.snap_or_recruit(X[i])
            results.append(result)
        return results