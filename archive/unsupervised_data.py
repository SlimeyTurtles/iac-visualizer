"""
Dataset generator for the Unsupervised Concept Model.

Creates feature patterns across multiple pools (color, shape, size, texture)
that the model can learn to categorize without supervision.
"""

import torch
import json


def generate_object_dataset():
    """
    Generate a dataset of simple objects with features across 4 pools.

    Pools:
    - Color: red, green, blue, yellow (4 units)
    - Shape: circle, square, triangle, star (4 units)
    - Size: small, medium, large (3 units)
    - Texture: smooth, rough, bumpy (3 units)

    Total: 14 units

    Objects are created with consistent feature combinations that form
    natural categories (e.g., "fruits" might be red/round/small/smooth).
    """
    pool_sizes = [4, 4, 3, 3]  # color, shape, size, texture
    pool_names = ["Color", "Shape", "Size", "Texture"]

    # Feature names within each pool
    features = {
        "Color": ["red", "green", "blue", "yellow"],
        "Shape": ["circle", "square", "triangle", "star"],
        "Size": ["small", "medium", "large"],
        "Texture": ["smooth", "rough", "bumpy"]
    }

    # Define episodes (objects) with their feature patterns
    # Each episode is a dict mapping pool -> active feature
    episodes = [
        # Category: "Apples" (red, circle, small/medium, smooth)
        {"name": "apple1", "Color": "red", "Shape": "circle", "Size": "small", "Texture": "smooth"},
        {"name": "apple2", "Color": "red", "Shape": "circle", "Size": "medium", "Texture": "smooth"},
        {"name": "apple3", "Color": "green", "Shape": "circle", "Size": "small", "Texture": "smooth"},

        # Category: "Oranges" (yellow, circle, medium, bumpy)
        {"name": "orange1", "Color": "yellow", "Shape": "circle", "Size": "medium", "Texture": "bumpy"},
        {"name": "orange2", "Color": "yellow", "Shape": "circle", "Size": "small", "Texture": "bumpy"},

        # Category: "Blocks" (various colors, square, various sizes, smooth)
        {"name": "block1", "Color": "red", "Shape": "square", "Size": "large", "Texture": "smooth"},
        {"name": "block2", "Color": "blue", "Shape": "square", "Size": "medium", "Texture": "smooth"},
        {"name": "block3", "Color": "yellow", "Shape": "square", "Size": "small", "Texture": "smooth"},
        {"name": "block4", "Color": "green", "Shape": "square", "Size": "large", "Texture": "smooth"},

        # Category: "Rocks" (various colors, triangle, various sizes, rough)
        {"name": "rock1", "Color": "blue", "Shape": "triangle", "Size": "small", "Texture": "rough"},
        {"name": "rock2", "Color": "blue", "Shape": "triangle", "Size": "medium", "Texture": "rough"},
        {"name": "rock3", "Color": "green", "Shape": "triangle", "Size": "large", "Texture": "rough"},

        # Category: "Stars" (yellow, star, various, smooth)
        {"name": "star1", "Color": "yellow", "Shape": "star", "Size": "small", "Texture": "smooth"},
        {"name": "star2", "Color": "yellow", "Shape": "star", "Size": "large", "Texture": "smooth"},

        # Mixed/edge cases
        {"name": "gem1", "Color": "blue", "Shape": "star", "Size": "small", "Texture": "smooth"},
        {"name": "ball1", "Color": "green", "Shape": "circle", "Size": "large", "Texture": "rough"},
    ]

    return {
        "pool_sizes": pool_sizes,
        "pool_names": pool_names,
        "features": features,
        "episodes": episodes
    }


def episode_to_vector(episode, features, pool_sizes, sigma=1.0):
    """
    Convert an episode dict to a feature vector with Gaussian activation.

    Instead of one-hot encoding, features follow a Gaussian distribution
    centered on the active feature. Adjacent features get partial activation
    based on their distance from the center.

    Args:
        episode: dict mapping pool name to active feature
        features: dict mapping pool name to list of feature names
        pool_sizes: list of pool sizes
        sigma: standard deviation of the Gaussian (controls spread)
    """
    import math

    pool_names = list(features.keys())
    vector = []

    for i, pool_name in enumerate(pool_names):
        pool_features = features[pool_name]
        pool_vec = [0.0] * pool_sizes[i]

        if pool_name in episode:
            active_feature = episode[pool_name]
            if active_feature in pool_features:
                center_idx = pool_features.index(active_feature)

                # Apply Gaussian activation to all features in the pool
                for j in range(len(pool_features)):
                    distance = abs(j - center_idx)
                    # Gaussian: exp(-d^2 / (2 * sigma^2))
                    activation = math.exp(-(distance ** 2) / (2 * sigma ** 2))
                    pool_vec[j] = activation

        vector.extend(pool_vec)

    return vector


def get_dataset_tensor():
    """Returns the dataset as a torch tensor and metadata."""
    data = generate_object_dataset()

    vectors = []
    for ep in data["episodes"]:
        vec = episode_to_vector(ep, data["features"], data["pool_sizes"])
        vectors.append(vec)

    X = torch.tensor(vectors, dtype=torch.float32)

    return {
        "X": X,
        "pool_sizes": data["pool_sizes"],
        "pool_names": data["pool_names"],
        "features": data["features"],
        "episode_names": [ep["name"] for ep in data["episodes"]],
        "episodes": data["episodes"]
    }


def get_dataset_json():
    """Returns the dataset as JSON-serializable dict."""
    data = get_dataset_tensor()

    return {
        "X": data["X"].tolist(),
        "pool_sizes": data["pool_sizes"],
        "pool_names": data["pool_names"],
        "features": data["features"],
        "episode_names": data["episode_names"],
        "episodes": data["episodes"]
    }


if __name__ == "__main__":
    data = get_dataset_json()
    print(f"Dataset: {len(data['episodes'])} episodes")
    print(f"Pool sizes: {data['pool_sizes']} (total: {sum(data['pool_sizes'])})")
    print(f"Pool names: {data['pool_names']}")
    print("\nEpisodes:")
    for i, ep in enumerate(data["episodes"]):
        print(f"  {i}: {ep['name']} - {data['X'][i]}")
