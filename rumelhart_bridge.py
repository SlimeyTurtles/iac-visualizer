import sys
import json
import torch
from rumelhart import RumelhartModel


# Global model instance for persistence during training
_model = None


def get_network_info(csv_path='concept_properties.csv', hidden_dim=8):
    """Returns network structure."""
    model = RumelhartModel(csv_path, hidden_dim=hidden_dim)

    return {
        "concepts": model.concepts,
        "properties": model.properties,
        "n_concepts": model.n_concepts,
        "n_properties": model.n_properties,
        "hidden_dim": model.hidden_dim,
        "targets": model.Y.cpu().tolist()
    }


def train_model(params, csv_path='concept_properties.csv'):
    """Train the model and return loss history plus final state."""
    global _model

    hidden_dim = int(params.get('hidden_dim', 8))
    eta = float(params.get('eta', 0.01))
    epochs = int(params.get('epochs', 2000))

    _model = RumelhartModel(csv_path, hidden_dim=hidden_dim, learning_rate=eta)

    # Train and collect detailed history
    losses = []
    weight_history = []
    hidden_history = []
    prediction_history = []

    # Record initial state
    record_state(losses, weight_history, hidden_history, prediction_history, 0.0)

    for epoch in range(1, epochs + 1):
        order = torch.randperm(_model.n_concepts, device=_model.device).tolist()
        total_loss = 0.0

        for i in order:
            h, y = _model.forward_from_index(i)
            t = _model.Y[i]
            e = (t - y)
            loss = 0.5 * torch.sum(e * e)
            total_loss += float(loss.detach().cpu())

            # Backprop
            delta_out = (t - y) * (y * (1.0 - y))
            delta_h = (_model.W_out @ delta_out) * (h * (1.0 - h))

            _model.W_out += _model.learning_rate * torch.ger(h, delta_out)
            _model.b_out += _model.learning_rate * delta_out
            _model.W_in[i] += _model.learning_rate * delta_h
            _model.b_h += _model.learning_rate * delta_h

        avg_loss = total_loss / _model.n_concepts

        # Record state at intervals
        if epoch % max(1, epochs // 100) == 0 or epoch == epochs:
            record_state(losses, weight_history, hidden_history, prediction_history, avg_loss)

    return {
        "losses": losses,
        "weight_history": weight_history,
        "hidden_history": hidden_history,
        "prediction_history": prediction_history,
        "concepts": _model.concepts,
        "properties": _model.properties,
        "hidden_dim": _model.hidden_dim,
        "final_predictions": get_all_predictions()
    }


def record_state(losses, weight_history, hidden_history, prediction_history, loss):
    """Record current model state."""
    global _model

    losses.append(loss)

    # Record weights
    weight_history.append({
        "W_in": _model.W_in.cpu().tolist(),
        "W_out": _model.W_out.cpu().tolist(),
        "b_h": _model.b_h.cpu().tolist(),
        "b_out": _model.b_out.cpu().tolist()
    })

    # Record hidden representations for each concept
    hidden_reps = []
    for i in range(_model.n_concepts):
        h, _ = _model.forward_from_index(i)
        hidden_reps.append(h.detach().cpu().tolist())
    hidden_history.append(hidden_reps)

    # Record predictions
    preds = []
    for i in range(_model.n_concepts):
        _, y = _model.forward_from_index(i)
        preds.append(y.detach().cpu().tolist())
    prediction_history.append(preds)


def get_all_predictions():
    """Get predictions for all concepts."""
    global _model
    if _model is None:
        return {}

    result = {}
    for concept in _model.concepts:
        preds = _model.predict_properties(concept)
        result[concept] = preds
    return result


def predict_concept(concept_name, csv_path='concept_properties.csv', hidden_dim=8):
    """Predict properties for a single concept."""
    global _model

    if _model is None:
        _model = RumelhartModel(csv_path, hidden_dim=hidden_dim)

    if concept_name not in _model.concept_to_idx:
        return {"error": f"Unknown concept: {concept_name}"}

    preds = _model.predict_properties(concept_name)
    hidden = _model.hidden_representation(concept_name).cpu().tolist()

    return {
        "concept": concept_name,
        "predictions": preds,
        "hidden_representation": hidden
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No command specified"}))
        sys.exit(1)

    command = sys.argv[1]

    if command == "info":
        params = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        hidden_dim = int(params.get('hidden_dim', 8))
        result = get_network_info(hidden_dim=hidden_dim)
        print(json.dumps(result))

    elif command == "train":
        params = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        result = train_model(params)
        print(json.dumps(result))

    elif command == "predict":
        concept_name = sys.argv[2] if len(sys.argv) > 2 else ""
        params = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
        result = predict_concept(concept_name, hidden_dim=int(params.get('hidden_dim', 8)))
        print(json.dumps(result))

    else:
        print(json.dumps({"error": f"Unknown command: {command}"}))
