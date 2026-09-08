import sys
import json
from pathfind_pe import PathfindPEModel


def build_model(params):
    return PathfindPEModel(
        num_scenarios=int(params['numScenarios']),
        num_actions=int(params['numActions']),
        dim=int(params.get('dim', 2)),
        k=int(params.get('k', 3)),
        seed=int(params['seed']),
        weight_mode=params['weightMode'],
        excitatory=float(params['excitatory']),
        inhibitory=float(params['inhibitory']),
        decay=float(params['decay']),
        rectify=bool(params.get('rectify', True)),
        inhibition_mode=params.get('inhibitionMode', 'sibling'),
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: pathfind_pe_bridge.py <network|run> <params-json> [clamped-json]"}))
        sys.exit(1)

    command = sys.argv[1]
    params = json.loads(sys.argv[2])

    if command == "network":
        model = build_model(params)
        print(json.dumps(model.network_info()))

    elif command == "run":
        clamped = json.loads(sys.argv[3])
        model = build_model(params)
        history = model.run(clamped, int(params['steps']),
                            dir_inhibition=float(params.get('dirInh', 1.0)))
        result = model.network_info()
        result["history"] = history
        result["similarity_field"] = [round(v, 4) for v in
                                      model.similarity_field(clamped).tolist()]
        print(json.dumps(result))

    else:
        print(json.dumps({"error": f"Unknown command: {command}"}))
