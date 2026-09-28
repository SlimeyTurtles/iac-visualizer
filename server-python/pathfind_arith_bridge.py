import sys
import json
import random

from pathfind_arith import PathfindArithModel


def build_model(params):
    return PathfindArithModel(
        a_max=int(params.get('aMax', 10)),
        b_max=int(params.get('bMax', 10)),
        c_max=int(params.get('cMax', 5)),
        excitatory=float(params.get('excitatory', 0.05)),
        inhibitory=float(params.get('inhibitory', -0.001)),
        decay=float(params.get('decay', 0.1)),
        w_max=float(params.get('wMax', 0.15)),
        eta=float(params.get('eta', 0.02)),
        norm=float(params.get('norm', 1.0)),
        rectify=bool(params.get('rectify', True)),
        inh_mode=str(params.get('inhMode', 'global')),
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: pathfind_arith_bridge.py <network|run|train> <params-json> [extra-json]"}))
        sys.exit(1)

    command = sys.argv[1]
    params = json.loads(sys.argv[2])
    extra = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}

    model = build_model(params)
    model.load_learned(extra.get('learned'))
    steps = int(params.get('steps', 80))
    max_steps = int(params.get('maxPathSteps', 60))
    gate = str(params.get('learnGate', 'always'))

    if command == "network":
        print(json.dumps(model.network_info()))

    elif command == "run":
        start = tuple(extra['start'])
        goal = tuple(extra['goal'])
        learn = bool(extra.get('learn', True))
        history_every = int(params.get('historyEvery', 2))
        history, analysis = model.run_episode(
            start, goal, steps, learn=learn, gate=gate,
            history_every=history_every, max_steps=max_steps)
        print(json.dumps({
            "start": list(start), "goal": list(goal),
            "history": history,
            "analysis": analysis,
            "learned": model.export_learned(),
        }))

    elif command == "train":
        episodes = int(extra.get('episodes', 100))
        seed = int(extra.get('seed', 42))
        start_min = extra.get('startMin')
        start_max = extra.get('startMax')
        rng = random.Random(seed)
        rows = []
        for ep in range(episodes):
            start, goal = model.sample_task(rng, start_min, start_max)
            _, a = model.run_episode(start, goal, steps, learn=True, gate=gate,
                                     history_every=0, max_steps=max_steps)
            rows.append({
                "start": list(start), "goal": list(goal),
                "outcome": a["outcome"],
                "found_len": a["found_len"],
                "optimal_len": a["optimal_len"],
                "efficiency": a["efficiency"],
            })
        print(json.dumps({
            "rows": rows,
            "learned": model.export_learned(),
        }))

    else:
        print(json.dumps({"error": f"Unknown command: {command}"}))
