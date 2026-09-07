import sys
import json
from pathfind import PathfindIACModel


def build_model(params):
    return PathfindIACModel(
        num_scenarios=int(params['numScenarios']),
        num_actions=int(params['numActions']),
        density=float(params['density']),
        seed=int(params['seed']),
        weight_mode=params['weightMode'],
        excitatory=float(params['excitatory']),
        inhibitory=float(params['inhibitory']),
        decay=float(params['decay']),
        max_connections=int(params.get('maxConnections', 0)),
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: pathfind_bridge.py <network|run> <params-json> [clamped-json]"}))
        sys.exit(1)

    command = sys.argv[1]
    params = json.loads(sys.argv[2])

    if command == "network":
        model = build_model(params)
        print(json.dumps(model.network_info()))

    elif command == "run":
        clamped = json.loads(sys.argv[3])
        model = build_model(params)
        history = model.run(clamped, int(params['steps']))
        result = model.network_info()
        result["history"] = history
        print(json.dumps(result))

    else:
        print(json.dumps({"error": f"Unknown command: {command}"}))
