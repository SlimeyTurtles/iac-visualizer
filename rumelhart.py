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
        learning_rate: float = 0.01,
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
        self.learning_rate = learning_rate

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

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch forward: returns H (n_concepts, hidden_dim), Y_pred (n_concepts, n_properties)."""
        H = torch.sigmoid(self.W_in + self.b_h)
        Y_pred = torch.sigmoid(H @ self.W_out + self.b_out)
        return H, Y_pred

    def predict_properties(self) -> pd.DataFrame:
        """Returns predictions for all concepts as a DataFrame."""
        _, Y_pred = self.forward()
        return pd.DataFrame(
            Y_pred.detach().cpu().numpy(),
            index=self.concepts,
            columns=self.properties,
        )

    def hidden_representations(self) -> torch.Tensor:
        """Returns hidden representations for all concepts: (n_concepts, hidden_dim)."""
        H, _ = self.forward()
        return H.detach().clone()

    def train(self, epochs: int = 20000, verbose_every: int = 200) -> List[float]:
        """Batch gradient descent over all concepts."""
        losses: List[float] = []

        for epoch in range(1, epochs + 1):
            H, Y_pred = self.forward()

            # Squared error
            E = self.Y - Y_pred
            loss = 0.5 * torch.sum(E ** 2)
            avg_loss = float(loss.detach().cpu()) / self.n_concepts
            losses.append(avg_loss)

            # Backprop as matrices
            delta_out = E * (Y_pred * (1.0 - Y_pred))                # (n_concepts, n_properties)
            delta_h = (delta_out @ self.W_out.T) * (H * (1.0 - H))   # (n_concepts, hidden_dim)

            # Update weights
            self.W_out += self.learning_rate * (H.T @ delta_out)
            self.b_out += self.learning_rate * delta_out.sum(dim=0)
            self.W_in  += self.learning_rate * delta_h
            self.b_h   += self.learning_rate * delta_h.sum(dim=0)

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
        if new_concept_name not in self.concept_to_idx:
            raise ValueError(f"Unknown concept '{new_concept_name}'")
        i = self.concept_to_idx[new_concept_name]

        t = torch.zeros(self.n_properties, device=self.device)
        for prop, val in known_properties.items():
            if prop not in self.property_to_idx:
                raise ValueError(f"Unknown property '{prop}'. Available: {self.properties}")
            t[self.property_to_idx[prop]] = float(val)

        losses: List[float] = []

        for step in range(1, steps + 1):
            # Forward single concept (frozen W_out/b_out)
            h = torch.sigmoid(self.W_in[i] + self.b_h)
            y = torch.sigmoid(h @ self.W_out + self.b_out)

            e = (t - y)
            loss = 0.5 * torch.sum(e * e)
            losses.append(float(loss.detach().cpu()))

            # output delta (still computed; used to send error back)
            delta_out = (t - y) * (y * (1.0 - y))
            delta_h = (self.W_out @ delta_out) * (h * (1.0 - h))

            # Update ONLY new concept input->hidden row + hidden bias
            self.W_in[i] += self.learning_rate * delta_h
            self.b_h += self.learning_rate * delta_h

            if verbose_every and (step % verbose_every == 0 or step == 1 or step == steps):
                print(f"[{new_concept_name}] step {step:>4}/{steps} | loss={losses[-1]:.6f}")

        return losses


if __name__ == "__main__":
    # Train base network
    m = RumelhartModel("concept_properties.csv", hidden_dim=8, learning_rate=0.01)
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
    preds = m.predict_properties().loc["sparrow"].sort_values(ascending=False)
    for prop, val in preds.head(12).items():
        print(f"{prop:>20}: {val:.3f}")