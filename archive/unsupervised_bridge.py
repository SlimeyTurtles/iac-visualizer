import sys
import json
import torch
from unsupervised import UnsupervisedConceptModel
from unsupervised_data import get_dataset_json, get_dataset_tensor


def get_network_info():
    """Returns dataset structure and model info."""
    data = get_dataset_json()

    return {
        "pool_sizes": data["pool_sizes"],
        "pool_names": data["pool_names"],
        "features": data["features"],
        "episodes": data["episodes"],
        "episode_names": data["episode_names"],
        "total_dim": sum(data["pool_sizes"])
    }


def run_training(params):
    """
    Process all episodes through the model and return detailed history.
    """
    data = get_dataset_tensor()
    X = data["X"]

    vigilance = float(params.get("vigilance", 0.7))
    learning_rate = float(params.get("learning_rate", 0.1))
    gain = float(params.get("gain", 1.0))
    inhibition = float(params.get("inhibition", 0.2))
    num_passes = int(params.get("num_passes", 3))

    model = UnsupervisedConceptModel(
        pool_sizes=data["pool_sizes"],
        vigilance=vigilance,
        learning_rate=learning_rate,
        gain=gain,
        inhibition_strength=inhibition
    )

    # Process episodes multiple times to allow learning
    history = []
    hub_history = []  # Track hub templates over time

    for pass_num in range(num_passes):
        for i in range(X.shape[0]):
            episode = X[i]
            result = model.snap_or_recruit(episode)

            # Record state
            step_record = {
                "pass": pass_num,
                "episode_idx": i,
                "episode_name": data["episode_names"][i],
                "action": result["action"],
                "winner": result["winner"],
                "num_hubs": model.templates.shape[0],
            }

            if result["pool_similarities"] is not None:
                step_record["pool_similarities"] = result["pool_similarities"].tolist()

            if "activities" in result and result["activities"] is not None:
                step_record["activities"] = result["activities"].tolist()

            if "previous_winner" in result:
                step_record["previous_winner"] = result["previous_winner"]

            history.append(step_record)

            # Record hub templates
            hub_history.append({
                "step": len(history) - 1,
                "templates": model.templates.cpu().tolist()
            })

    # Final hub assignments for each episode
    final_assignments = []
    for i in range(X.shape[0]):
        episode = X[i]
        if model.templates.shape[0] > 0:
            winner, activities = model.settle_competition(episode)
            prediction = model.predict_from_hub(winner)
            pool_sims = model.compute_pool_similarities(episode, prediction)
            final_assignments.append({
                "episode_idx": i,
                "episode_name": data["episode_names"][i],
                "hub": winner,
                "activities": activities.cpu().tolist() if activities is not None else None,
                "pool_similarities": pool_sims.cpu().tolist(),
                "prediction": prediction.cpu().tolist()
            })
        else:
            final_assignments.append({
                "episode_idx": i,
                "episode_name": data["episode_names"][i],
                "hub": None,
                "activities": None,
                "pool_similarities": None,
                "prediction": None
            })

    return {
        "history": history,
        "hub_history": hub_history,
        "final_templates": model.templates.cpu().tolist(),
        "final_assignments": final_assignments,
        "num_hubs": model.templates.shape[0],
        "pool_sizes": data["pool_sizes"],
        "pool_names": data["pool_names"],
        "features": data["features"],
        "episodes": data["episodes"],
        "episode_names": data["episode_names"],
        "X": X.tolist()
    }


def process_single_episode(episode_idx, params):
    """Process a single episode and return the result."""
    data = get_dataset_tensor()
    X = data["X"]

    if episode_idx < 0 or episode_idx >= X.shape[0]:
        return {"error": f"Invalid episode index: {episode_idx}"}

    vigilance = float(params.get("vigilance", 0.7))
    learning_rate = float(params.get("learning_rate", 0.1))
    gain = float(params.get("gain", 1.0))
    inhibition = float(params.get("inhibition", 0.2))

    # Initialize fresh model
    model = UnsupervisedConceptModel(
        pool_sizes=data["pool_sizes"],
        vigilance=vigilance,
        learning_rate=learning_rate,
        gain=gain,
        inhibition_strength=inhibition
    )

    episode = X[episode_idx]
    result = model.snap_or_recruit(episode)

    return {
        "episode_idx": episode_idx,
        "episode_name": data["episode_names"][episode_idx],
        "action": result["action"],
        "winner": result["winner"],
        "pool_similarities": result["pool_similarities"].tolist() if result["pool_similarities"] is not None else None,
        "template": model.templates[result["winner"]].cpu().tolist() if model.templates.shape[0] > 0 else None
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No command specified"}))
        sys.exit(1)

    command = sys.argv[1]

    if command == "info":
        result = get_network_info()
        print(json.dumps(result))

    elif command == "train":
        params = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
        result = run_training(params)
        print(json.dumps(result))

    elif command == "process":
        episode_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        params = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
        result = process_single_episode(episode_idx, params)
        print(json.dumps(result))

    else:
        print(json.dumps({"error": f"Unknown command: {command}"}))
