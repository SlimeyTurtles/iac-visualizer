import sys
import json
import random
from pathfind_mlp import PathfindMLPModel


def build_model(params, lesions=None):
    return PathfindMLPModel(
        width=int(params['width']),
        depth=int(params['depth']),
        conn_p=float(params['connP']),
        seed=int(params['seed']),
        w_mean=float(params.get('wMean', 0.5)),
        w_sd=float(params.get('wSd', 0.25)),
        w_lo=float(params.get('wLo', 0.0)),
        w_hi=float(params.get('wHi', 1.0)),
        excitatory=float(params.get('excitatory', 0.1)),
        inhibitory=float(params.get('inhibitory', -0.05)),
        decay=float(params.get('decay', 0.05)),
        rectify=bool(params.get('rectify', True)),
        lesions=lesions,
    )


def pick_goal(model, params, seed):
    goal_mode = str(params.get('goalMode', 'random'))
    top = [n for n in model.layers[-1] if n not in model.lesions]
    if not top:
        return None
    if goal_mode == 'random':
        return random.Random(seed * 7919 + 13).choice(top)
    i = max(1, min(len(top), int(goal_mode)))
    return top[i - 1]


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: pathfind_mlp_bridge.py <network|run|sweep> <params-json> [extra-json]"}))
        sys.exit(1)

    command = sys.argv[1]
    params = json.loads(sys.argv[2])
    lesions = params.get('lesions', [])

    if command == "network":
        model = build_model(params, lesions)
        print(json.dumps(model.network_info()))

    elif command == "run":
        extra = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
        goal = extra.get('goal')
        model = build_model(params, lesions)
        history = model.run(goal, int(params.get('steps', 200)))
        result = model.network_info()
        result["goal"] = goal
        result["history"] = history
        result["analysis"] = model.analyze(goal, history[-1])
        print(json.dumps(result))

    elif command == "sweep":
        extra = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
        seed_start = int(extra.get('seedStart', 0))
        seed_count = int(extra.get('seedCount', 100))
        steps = int(params.get('steps', 200))
        rows = []
        for seed in range(seed_start, seed_start + seed_count):
            p = dict(params)
            p['seed'] = seed
            model = build_model(p, lesions)
            goal = pick_goal(model, params, seed)
            if goal is None:
                continue
            history = model.run(goal, steps)
            a = model.analyze(goal, history[-1])
            rows.append({
                "seed": seed,
                "goal": goal,
                "edges": len(model.connections),
                "outcome": a["outcome"],
                "reachable": a["reachable"],
                "efficiency": a["efficiency"],
                "found_mean": a["found_mean"],
                "optimal_mean": a["optimal_mean"],
                "path_count": a["path_count"],
                "argmax_match": a["argmax_match"],
                "argmax_coherence": a["argmax_coherence"],
                "found_path": a["found_path"],
                "optimal_path": a["optimal_path"],
            })
        print(json.dumps({"rows": rows}))

    else:
        print(json.dumps({"error": f"Unknown command: {command}"}))
