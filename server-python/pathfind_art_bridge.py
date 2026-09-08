import sys
import json
from pathfind_art import ARTPathfinder


def build_model(params):
    return ARTPathfinder(
        num_states=int(params['numStates']),
        dim=int(params.get('dim', 4)),
        k=int(params.get('k', 3)),
        seed=int(params['seed']),
        vigilance=float(params.get('vigilance', 0.9)),
        lr=float(params.get('lr', 0.3)),
        noise=float(params.get('noise', 0.03)),
        samples=int(params.get('samples', 5)),
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: pathfind_art_bridge.py <network|search> <params-json> [endpoints-json]"}))
        sys.exit(1)

    command = sys.argv[1]
    params = json.loads(sys.argv[2])

    if command == "network":
        model = build_model(params)
        print(json.dumps(model.network_info()))

    elif command == "search":
        endpoints = json.loads(sys.argv[3])
        model = build_model(params)
        result = model.network_info()
        result.update(model.search(endpoints[0], endpoints[1],
                                   max_depth=int(params.get('maxDepth', 20))))
        print(json.dumps(result))

    else:
        print(json.dumps({"error": f"Unknown command: {command}"}))
