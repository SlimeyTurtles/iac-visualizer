import sys
import json
import torch
from iac import IACModel


def get_network_info(csv_path):
    """Returns network structure without running simulation."""
    import pandas as pd
    df = pd.read_csv(csv_path)

    columns = df.columns.tolist()
    node_to_pool = {}
    pools = {}

    for col in columns:
        unique_vals = df[col].unique().tolist()
        pools[col] = unique_vals
        for val in unique_vals:
            node_to_pool[val] = col

    # Build connection list (excitatory connections between pools)
    connections = []
    for _, row in df.iterrows():
        name = row['Name']
        for col in columns[1:]:  # Skip 'Name' column
            feature = row[col]
            connections.append({"source": name, "target": feature, "type": "excitatory"})

    all_nodes = []
    for col in columns:
        all_nodes.extend(df[col].unique().tolist())

    return {
        "nodes": all_nodes,
        "columns": columns,
        "pools": pools,
        "node_to_pool": node_to_pool,
        "connections": connections
    }


def run_simulation(selected_nodes, params, csv_path='jets_sharks.csv'):
    """Runs the IAC simulation and returns history."""
    model = IACModel(
        csv_path,
        excitatory=float(params['excitatory']),
        inhibitory=float(params['inhibitory']),
        decay=float(params['decay'])
    )

    ext = torch.zeros(model.n, device=model.device)
    for name in selected_nodes:
        if name in model.idx_lookup:
            ext[model.idx_lookup[name]] = 1.0

    history = []
    for _ in range(int(params['steps'])):
        state = model.step(ext)
        history.append(state.cpu().tolist())

    # Get network info
    info = get_network_info(csv_path)

    return {
        "nodes": model.all_nodes,
        "columns": info["columns"],
        "pools": info["pools"],
        "node_to_pool": info["node_to_pool"],
        "connections": info["connections"],
        "history": history,
        "idx_lookup": model.idx_lookup
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No command specified"}))
        sys.exit(1)

    command = sys.argv[1]

    if command == "info":
        # Just get network structure
        result = get_network_info('jets_sharks.csv')
        print(json.dumps(result))

    elif command == "run":
        # Run simulation
        selected_nodes = json.loads(sys.argv[2])
        params = json.loads(sys.argv[3])
        result = run_simulation(selected_nodes, params)
        print(json.dumps(result))

    else:
        print(json.dumps({"error": f"Unknown command: {command}"}))
