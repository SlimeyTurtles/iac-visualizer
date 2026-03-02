import torch
import pandas as pd
from typing import Dict, List, Tuple, Optional


class RumelhartModel:
    """
    Rumelhart-style concept -> property network.

    - Inputs:  localist concept units (one-hot; implemented by indexing a row)
    - Hidden:  distributed representation (sigmoid)
    - Outputs: localist property units (sigmoid; multi-label)

    Learning: classic error-correction / backprop with a fixed small step (eta),
              implemented explicitly (no optimizer).
    """

    def __init__(
        self,
        csv_path: str,
        hidden_dim: int = 8,
        eta: float = 0.01,          # the "tiny nudge" amount (learning rate)
        seed: int = 42,
        device: Optional[str] = None,
    ):
        torch.manual_seed(seed)

        df = pd.read_csv(csv_path)
        if df.shape[1] < 2:
            raise ValueError("CSV must have: concept column + at least one property column.")

        concept_col = df.columns[0]
        property_cols = list(df.columns[1:])

        self.concepts: List[str] = df[concept_col].astype(str).tolist()
        self.properties: List[str] = property_cols

        self.n_concepts = len(self.concepts)
        self.n_properties = len(self.properties)
        self.hidden_dim = hidden_dim
        self.eta = eta

        self.concept_to_idx = {c: i for i, c in enumerate(self.concepts)}
        self.property_to_idx = {p: j for j, p in enumerate(self.properties)}

        # Targets: (n_concepts, n_properties), values 0/1
        Y = torch.tensor(df[property_cols].values, dtype=torch.float32)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.Y = Y.to(self.device)

        # Weights (small random init, like classic connectionist demos)
        # Input->Hidden: (n_concepts, hidden_dim) since input is one-hot
        self.W_in = torch.randn(self.n_concepts, hidden_dim, device=self.device) * 0.01
        self.b_h = torch.zeros(hidden_dim, device=self.device)

        # Hidden->Output: (hidden_dim, n_properties)
        self.W_out = torch.randn(hidden_dim, self.n_properties, device=self.device) * 0.01
        self.b_out = torch.zeros(self.n_properties, device=self.device)

    # ---------- Forward ----------
    def _sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + torch.exp(-x))

    def forward_from_index(self, concept_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          h: (hidden_dim,)
          y: (n_properties,)
        """
        h_net = self.W_in[concept_idx] + self.b_h
        h = self._sigmoid(h_net)

        y_net = h @ self.W_out + self.b_out
        y = self._sigmoid(y_net)
        return h, y

    def predict_properties(self, concept_name: str) -> Dict[str, float]:
        idx = self._require_concept(concept_name)
        _, y = self.forward_from_index(idx)
        vals = y.detach().cpu().tolist()
        return {p: vals[j] for j, p in enumerate(self.properties)}

    def hidden_representation(self, concept_name: str) -> torch.Tensor:
        idx = self._require_concept(concept_name)
        h, _ = self.forward_from_index(idx)
        return h.detach().clone()

    # ---------- Error-correction learning (manual backprop) ----------
    def train(self, epochs: int = 20000, verbose_every: int = 200, shuffle: bool = True) -> List[float]:
        """
        Classic online error-correction:
        - present one concept
        - compute output error
        - backpropagate error to hidden
        - nudge weights by eta
        """
        losses: List[float] = []

        for epoch in range(1, epochs + 1):
            if shuffle:
                order = torch.randperm(self.n_concepts, device=self.device).tolist()
            else:
                order = list(range(self.n_concepts))

            total_loss = 0.0

            for i in order:
                # Forward
                h, y = self.forward_from_index(i)
                t = self.Y[i]  # target properties

                # Squared error (classic in early connectionist demos)
                # E = 0.5 * sum_k (t_k - y_k)^2
                e = (t - y)
                loss = 0.5 * torch.sum(e * e)
                total_loss += float(loss.detach().cpu())

                # --- Backprop deltas (sigmoid derivative) ---
                # output delta: δ_out = (t - y) * y*(1-y)
                delta_out = (t - y) * (y * (1.0 - y))  # (n_properties,)

                # hidden delta: δ_h = (W_out @ δ_out) * h*(1-h)
                # (hidden_dim,)
                delta_h = (self.W_out @ delta_out) * (h * (1.0 - h))

                # --- Weight nudges ---
                # W_out += eta * (h[:,None] @ delta_out[None,:])
                # b_out += eta * delta_out
                self.W_out += self.eta * torch.ger(h, delta_out)
                self.b_out += self.eta * delta_out

                # For one-hot input, only the row W_in[i] gets updated:
                # W_in[i] += eta * delta_h
                # b_h += eta * delta_h
                self.W_in[i] += self.eta * delta_h
                self.b_h += self.eta * delta_h

            avg_loss = total_loss / self.n_concepts
            losses.append(avg_loss)

            if verbose_every and (epoch % verbose_every == 0 or epoch == 1 or epoch == epochs):
                print(f"Epoch {epoch:>4}/{epochs} | avg_loss={avg_loss:.6f}")

        return losses

    # ---------- Rumelhart-style generalization: add new concept, freeze hidden->out ----------
    def add_new_concept(self, new_concept_name: str, seed: int = 123) -> None:
        if new_concept_name in self.concept_to_idx:
            raise ValueError(f"Concept '{new_concept_name}' already exists.")
        torch.manual_seed(seed)

        self.concept_to_idx[new_concept_name] = self.n_concepts
        self.concepts.append(new_concept_name)

        new_row = torch.randn(1, self.hidden_dim, device=self.device) * 0.01
        self.W_in = torch.cat([self.W_in, new_row], dim=0)

        new_targets = torch.zeros(1, self.n_properties, device=self.device)
        self.Y = torch.cat([self.Y, new_targets], dim=0)

        self.n_concepts += 1

    def train_new_concept_input_to_hidden_only(
        self,
        new_concept_name: str,
        known_properties: Dict[str, int],
        steps: int = 600,
        verbose_every: int = 100,
    ) -> List[float]:
        """
        Teach ONLY a few facts (e.g., "is_a_bird"=1, "is_an_animal"=1, ...),
        while freezing hidden->output weights.

        Only W_in[row_of_new_concept] and b_h are updated.
        """
        i = self._require_concept(new_concept_name)

        t = torch.zeros(self.n_properties, device=self.device)
        for prop, val in known_properties.items():
            if prop not in self.property_to_idx:
                raise ValueError(f"Unknown property '{prop}'. Available: {self.properties}")
            t[self.property_to_idx[prop]] = float(val)

        losses: List[float] = []

        for step in range(1, steps + 1):
            # Forward with frozen W_out/b_out (we simply do not update them)
            h, y = self.forward_from_index(i)

            e = (t - y)
            loss = 0.5 * torch.sum(e * e)
            losses.append(float(loss.detach().cpu()))

            # output delta (still computed; used to send error back)
            delta_out = (t - y) * (y * (1.0 - y))
            delta_h = (self.W_out @ delta_out) * (h * (1.0 - h))

            # Update ONLY new concept input->hidden row + hidden bias
            self.W_in[i] += self.eta * delta_h
            self.b_h += self.eta * delta_h

            if verbose_every and (step % verbose_every == 0 or step == 1 or step == steps):
                print(f"[{new_concept_name}] step {step:>4}/{steps} | loss={losses[-1]:.6f}")

        return losses

    # ---------- Helpers ----------
    def _require_concept(self, concept_name: str) -> int:
        if concept_name not in self.concept_to_idx:
            raise ValueError(f"Unknown concept '{concept_name}'. Available: {self.concepts}")
        return self.concept_to_idx[concept_name]


if __name__ == "__main__":
    # Train base network
    m = RumelhartModel("concept_properties.csv", hidden_dim=8, eta=0.01)
    m.train(epochs=2000, verbose_every=200)

    # Add sparrow and teach ONLY ISA facts (like the book demo)
    m.add_new_concept("sparrow")
    m.train_new_concept_input_to_hidden_only(
        "sparrow",
        known_properties={"is_a_bird": 1, "is_an_animal": 1, "is_living_thing": 1},
        steps=600,
        verbose_every=100,
    )

    # See if it generalizes to other bird properties
    print("\nTop predicted properties for sparrow:")
    preds = m.predict_properties("sparrow")
    for k, v in sorted(preds.items(), key=lambda kv: kv[1], reverse=True)[:12]:
        print(f"{k:>20}: {v:.3f}")