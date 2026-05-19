import torch
import pandas as pd

class IACModel:
    def __init__(self, csv_path, excitatory=0.1, inhibitory=-0.2, decay=0.05):
        # 1. Hyperparameters
        self.excitatory_w = excitatory
        self.inhibitory_w = inhibitory
        self.decay_w = decay
        self.max_a = 1.0
        self.min_a = -0.2

        # 2. Data Loading & Indexing
        self.df = pd.read_csv(csv_path)
        self.all_nodes = []
        for col in self.df.columns:
            self.all_nodes.extend(self.df[col].unique().tolist())
        
        self.idx_lookup = {name: i for i, name in enumerate(self.all_nodes)}
        self.n = len(self.all_nodes)
        
        # 3. Hardware & Tensors
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.activations = torch.zeros(self.n, device=self.device)
        self.weights = torch.zeros((self.n, self.n), device=self.device)
        
        # 4. Initialize Network
        self._init_weights()

    def _init_weights(self):
        """Builds the internal excitatory and inhibitory connections."""
        # Fill Excitatory Weights (Person <-> Feature)
        for _, row in self.df.iterrows():
            name_idx = self.idx_lookup[row['Name']]
            for feature in row[1:]:
                feat_idx = self.idx_lookup[feature]
                self.weights[name_idx, feat_idx] = self.excitatory_w
                self.weights[feat_idx, name_idx] = self.excitatory_w

        # Fill Inhibitory Weights (Pool-based competition)
        for col in self.df.columns:
            pool_indices = [self.idx_lookup[val] for val in self.df[col].unique()]
            for i in pool_indices:
                for j in pool_indices:
                    if i != j:
                        self.weights[i, j] = self.inhibitory_w

    def step(self, external_input_vector):
        """Performs one update cycle and returns the new activation state."""
        # Calculate Net Input: (W * a) + External
        net_input = torch.mv(self.weights, self.activations) + external_input_vector
        
        # Logic Masks
        pos_mask = (net_input > 0).float()
        neg_mask = (net_input <= 0).float()
        
        # IAC Update Rules
        delta_pos = net_input * (self.max_a - self.activations) * pos_mask
        delta_neg = net_input * (self.activations - self.min_a) * neg_mask
        decay_term = self.decay_w * self.activations
        
        # Update internal state
        self.activations = self.activations + delta_pos + delta_neg - decay_term
        return self.activations

    def reset(self):
        """Clears current activation state."""
        self.activations = torch.zeros(self.n, device=self.device)

    def get_activation_dict(self):
        """Returns a human-readable dictionary of {node_name: value}."""
        vals = self.activations.tolist()
        return {name: vals[i] for name, i in self.idx_lookup.items()}