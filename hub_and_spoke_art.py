import torch
import torch.nn.functional as F


class UnsupervisedConceptModel:
    def __init__(
        self,
        pool_sizes,
        baseline_vigilance=0.8,
        learning_rate=0.1,
        gain=1.0,
        inhibition_strength=0.2,
        device="cpu",
    ):
        """
        pool_sizes:
            List like [20, 20, 20], one size per feature pool.

        baseline_vigilance:
            Default vigilance used when no episode-specific vigilance is supplied.
            Can be:
              - float/int: same threshold for every pool
              - list/tuple/tensor: one threshold per pool

        learning_rate:
            Update rate used when an episode snaps to an existing hub.

        gain:
            Gain on bottom-up drive into hubs.

        inhibition_strength:
            Strength of lateral inhibition during hub competition.
        """
        self.pool_sizes = list(pool_sizes)
        self.num_pools = len(pool_sizes)
        self.total_dim = sum(pool_sizes)
        self.learning_rate = learning_rate
        self.gain = gain
        self.inhibition_strength = inhibition_strength
        self.device = device

        self.baseline_vigilance = self._format_vigilance(baseline_vigilance)

        # Hub templates: [num_hubs, total_dim]
        self.templates = torch.empty(
            (0, self.total_dim),
            dtype=torch.float32,
            device=self.device,
        )

        # Slice boundaries for each pool
        self.pool_slices = []
        start = 0
        for size in self.pool_sizes:
            end = start + size
            self.pool_slices.append((start, end))
            start = end

    def _normalize(self, x):
        return F.normalize(x, p=2, dim=-1, eps=1e-8)

    def _ensure_1d(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        if x.dim() != 1 or x.shape[0] != self.total_dim:
            raise ValueError(
                f"Expected input of shape [{self.total_dim}], got {tuple(x.shape)}"
            )

        return x.to(self.device).float()

    def _format_vigilance(self, vigilance):
        """
        Convert vigilance into a tensor of shape [num_pools].

        Allowed forms:
          - scalar: broadcast to all pools
          - list/tuple/tensor of length num_pools
        """
        if vigilance is None:
            return self.baseline_vigilance.clone()

        if isinstance(vigilance, (float, int)):
            v = torch.full(
                (self.num_pools,),
                float(vigilance),
                dtype=torch.float32,
                device=self.device,
            )
            return v

        if isinstance(vigilance, (list, tuple)):
            if len(vigilance) != self.num_pools:
                raise ValueError(
                    f"Expected vigilance length {self.num_pools}, got {len(vigilance)}"
                )
            return torch.tensor(
                vigilance,
                dtype=torch.float32,
                device=self.device,
            )

        if isinstance(vigilance, torch.Tensor):
            vigilance = vigilance.to(self.device).float().flatten()
            if vigilance.numel() == 1:
                return torch.full(
                    (self.num_pools,),
                    float(vigilance.item()),
                    dtype=torch.float32,
                    device=self.device,
                )
            if vigilance.numel() != self.num_pools:
                raise ValueError(
                    f"Expected vigilance length {self.num_pools}, got {vigilance.numel()}"
                )
            return vigilance

        raise TypeError(
            "vigilance must be None, scalar, list/tuple, or torch.Tensor"
        )

    def recruit_hub(self, x):
        """
        Create a new hub initialized to the current episode pattern.
        Template is normalized to unit L2 norm.
        """
        x = self._ensure_1d(x)
        x = self._normalize(x.unsqueeze(0)).squeeze(0)
        self.templates = torch.cat([self.templates, x.unsqueeze(0)], dim=0)
        return self.templates.shape[0] - 1

    def compute_bottom_up_drive(self, x):
        """
        Recognition drive = gain * dot(template, input)

        Since templates and input are normalized, this behaves like cosine-like
        similarity over the full concatenated representation.
        """
        x = self._ensure_1d(x)

        if self.templates.shape[0] == 0:
            return torch.empty(0, dtype=torch.float32, device=self.device)

        x = self._normalize(x.unsqueeze(0)).squeeze(0)
        drive = self.gain * torch.matmul(self.templates, x)
        return drive

    def settle_competition(self, x, num_steps=20):
        """
        Simple hub competition:
          acts <- relu(drive - inhibition * summed_other_activity)

        Returns:
            winner_index, activities
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

        Recognition and prediction share the same underlying template.
        """
        return self.templates[hub_index].clone()

    def compute_pool_similarities(self, x, y):
        """
        Compute per-pool cosine similarity between:
          x = data-layer pattern
          y = expectation-layer pattern

        Returns:
            tensor of shape [num_pools]
        """
        x = self._ensure_1d(x)
        y = self._ensure_1d(y)

        sims = []

        for start, end in self.pool_slices:
            xp = x[start:end]
            yp = y[start:end]

            x_norm = torch.norm(xp, p=2)
            y_norm = torch.norm(yp, p=2)

            if x_norm < 1e-8 or y_norm < 1e-8:
                sim = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            else:
                sim = torch.dot(xp, yp) / (x_norm * y_norm + 1e-8)

            sims.append(sim)

        return torch.stack(sims)

    def update_hub(self, hub_index, x):
        """
        Move winning hub toward current episode, then renormalize.
        """
        x = self._ensure_1d(x)
        x = self._normalize(x.unsqueeze(0)).squeeze(0)

        old = self.templates[hub_index]
        new = (1.0 - self.learning_rate) * old + self.learning_rate * x
        self.templates[hub_index] = self._normalize(new.unsqueeze(0)).squeeze(0)

    def snap_or_recruit(self, x, vigilance=None):
        """
        Process one episode.

        vigilance:
            Episode-specific vigilance input.
            Allowed forms:
              - None: use baseline vigilance
              - scalar: same vigilance for every pool
              - vector/list/tensor length num_pools: per-pool vigilance

        Processing:
          1. competition among existing hubs
          2. winner predicts expectation pattern
          3. compute per-pool cosine similarity
          4. compare against episode vigilance
          5. snap if all pools pass; otherwise recruit new hub
        """
        x = self._ensure_1d(x)
        episode_vigilance = self._format_vigilance(vigilance)

        # If no hubs exist, recruit first hub immediately
        if self.templates.shape[0] == 0:
            winner = self.recruit_hub(x)
            return {
                "action": "recruit",
                "winner": winner,
                "previous_winner": None,
                "pool_similarities": None,
                "pool_passes": None,
                "vigilance_used": episode_vigilance.detach().cpu(),
                "activities": None,
            }

        winner, activities = self.settle_competition(x)
        y = self.predict_from_hub(winner)
        pool_sims = self.compute_pool_similarities(x, y)

        passes = pool_sims >= episode_vigilance

        if torch.all(passes):
            self.update_hub(winner, x)
            return {
                "action": "snap",
                "winner": winner,
                "previous_winner": winner,
                "pool_similarities": pool_sims.detach().cpu(),
                "pool_passes": passes.detach().cpu(),
                "vigilance_used": episode_vigilance.detach().cpu(),
                "activities": activities.detach().cpu(),
            }
        else:
            new_hub = self.recruit_hub(x)
            return {
                "action": "recruit",
                "winner": new_hub,
                "previous_winner": winner,
                "pool_similarities": pool_sims.detach().cpu(),
                "pool_passes": passes.detach().cpu(),
                "vigilance_used": episode_vigilance.detach().cpu(),
                "activities": activities.detach().cpu(),
            }

    def process_dataset(self, X, vigilances=None):
        """
        Process a dataset of episodes.

        X:
            Tensor or array of shape [num_samples, total_dim]

        vigilances:
            Controls vigilance per episode.

            Allowed forms:
              - None:
                    use baseline vigilance for every episode
              - scalar:
                    same vigilance for every episode and every pool
              - list/tensor length num_pools:
                    same per-pool vigilance for every episode
              - list of length num_samples:
                    each element can be:
                      - scalar
                      - per-pool vector
              - tensor of shape [num_samples, num_pools]
                    one vigilance vector per episode

        Returns:
            list of result dicts
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        X = X.to(self.device).float()

        if X.dim() != 2 or X.shape[1] != self.total_dim:
            raise ValueError(
                f"Expected X shape [num_samples, {self.total_dim}], got {tuple(X.shape)}"
            )

        results = []

        for i in range(X.shape[0]):
            if vigilances is None:
                v_i = None

            elif isinstance(vigilances, (float, int)):
                v_i = vigilances

            elif isinstance(vigilances, torch.Tensor):
                if vigilances.dim() == 1:
                    if vigilances.numel() == self.num_pools:
                        # same per-pool vigilance for all episodes
                        v_i = vigilances
                    elif vigilances.numel() == X.shape[0]:
                        # one scalar vigilance per episode
                        v_i = vigilances[i]
                    else:
                        raise ValueError(
                            "1D vigilance tensor must have length num_pools "
                            "or num_samples"
                        )
                elif vigilances.dim() == 2:
                    if vigilances.shape != (X.shape[0], self.num_pools):
                        raise ValueError(
                            f"Expected vigilance tensor shape "
                            f"({X.shape[0]}, {self.num_pools}), got {tuple(vigilances.shape)}"
                        )
                    v_i = vigilances[i]
                else:
                    raise ValueError("vigilances tensor must be 1D or 2D")

            elif isinstance(vigilances, (list, tuple)):
                if len(vigilances) == self.num_pools and all(
                    isinstance(v, (int, float)) for v in vigilances
                ):
                    # same per-pool vigilance for all episodes
                    v_i = vigilances
                elif len(vigilances) == X.shape[0]:
                    # one vigilance specification per episode
                    v_i = vigilances[i]
                else:
                    raise ValueError(
                        "List vigilance must be either:\n"
                        f"  - length {self.num_pools} (same per-pool vigilance for all episodes), or\n"
                        f"  - length {X.shape[0]} (one vigilance spec per episode)"
                    )
            else:
                raise TypeError(
                    "vigilances must be None, scalar, list/tuple, or torch.Tensor"
                )

            result = self.snap_or_recruit(X[i], vigilance=v_i)
            results.append(result)

        return results

    def get_num_hubs(self):
        return self.templates.shape[0]

    def reset(self):
        """
        Clear all learned hubs.
        """
        self.templates = torch.empty(
            (0, self.total_dim),
            dtype=torch.float32,
            device=self.device,
        )