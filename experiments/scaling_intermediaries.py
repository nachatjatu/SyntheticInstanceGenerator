import sys
import json
import pickle
import numpy as np
import pandas as pd
import os

sys.path.insert(0, "src")

from farmers_intermediaries import Instance
from road_graphs import RoadGraph
from pricing import Optimizer
from instance_generator import InstanceGenerator


# =========================
# CLI ARGUMENTS
# =========================
if len(sys.argv) != 2:
    raise ValueError("Please provide n_id as a single argument.")

n_id = int(sys.argv[1])


# =========================
# CONFIG
# =========================
N_INTS_LIST = [n_int for n_int in range(12, 30+1, 3)]
NUM_SEEDS = 4

GRAPH_PATH = "data/graph_0-14960_00.pickle"


FARMERS_PATH = "data/farmers.csv"
INTS_PATH = "data/ints.csv"
MILLS_PATH = "data/mills.csv"


# =========================
# LOAD STATIC DATA
# =========================
farmers_df = pd.read_csv(FARMERS_PATH)
ints_df = pd.read_csv(INTS_PATH)
mills_df = pd.read_csv(MILLS_PATH)

with open(GRAPH_PATH, "rb") as f:
    GRAPH = pickle.load(f)

# Initialize generator once
instance_generator = InstanceGenerator(farmers_df, ints_df, GRAPH_PATH)


# =========================
# CORE BUILDERS
# =========================
def build_instance(instance_id, n_ints, seed):
    """
    Generate a fresh instance and attach the road graph.
    """
    instance_generator.gen_ints(n_ints, seed=seed)

    instance_dict = instance_generator.gen_instance(
        instance_id,
        write=False,
        plot=False,
        seed=seed
    )

    platform = Instance.from_dict(instance_dict)
    platform.set_graph(RoadGraph(GRAPH))

    return platform, instance_dict


def reset_fixed_costs(platform):
    return {
        intermediary.id: (
            platform.dist_to_mill[intermediary.id] * 4
            + np.random.normal(0, 100000)
        )
        for intermediary in platform.intermediaries
    }



def run_single_simulation(platform):
    """
    Runs one optimization on a (possibly perturbed) platform.
    """
    platform.set_graph(RoadGraph(GRAPH))

    sampled_epsilon = np.random.choice([0,1,2,3,4,5,6,7,8,9])

    epsilon = {int.id: sampled_epsilon for int in platform.intermediaries}

    het_costs = reset_fixed_costs(platform)

    parameters = {
        "epsilon": epsilon,
        "solver": "gurobi",
        "het_costs": het_costs,
    }

    opt = Optimizer(platform, parameters)
    base_matchings = opt.base_matchings

    summary_vanilla = opt.solve("heuristic_optimized", options={
        "structured_farmer_prices": False,
        "domination": False,})
    
    opt = Optimizer(platform, parameters, base_matchings=base_matchings)
    
    summary_structured = opt.solve("heuristic_optimized", options={
        "structured_farmer_prices": True,
        "domination": False,})
    
    opt = Optimizer(platform, parameters, base_matchings=base_matchings)
    
    summary_domination = opt.solve("heuristic_optimized", options={
        "structured_farmer_prices": False,
        "domination": True,})

    return {
        "cost": het_costs,
        "epsilon": epsilon,
        "farmer_quantities": {f.id: f.quantity for f in platform.farmers},
        "summary_vanilla": summary_vanilla.to_dict(),
        "summary_structured": summary_structured.to_dict(),
        "summary_domination": summary_domination.to_dict(),
        "farmer_dirt_to_mill": {f.id: f.dirt_to_mill for f in platform.farmers},
        "farmer_paved_to_mill": {f.id: f.paved_to_mill for f in platform.farmers},
    }


# =========================
# JSON SERIALIZATION
# =========================
def convert(obj):
    if isinstance(obj, (np.float64, np.int64)):
        return float(obj)
    elif isinstance(obj, np.str_):
        return str(obj)
    elif isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


# =========================
# MAIN EXPERIMENT LOOP
# =========================
def main():
    
    # 1. Fast cycle: cycling through the intermediaries
    n_ints_idx = n_id % len(N_INTS_LIST)
    n_ints = N_INTS_LIST[n_ints_idx]

    # 2. Slow cycle: change seed only after we've finished all n_ints for the current seed
    instance_seed = (n_id // len(N_INTS_LIST)) % NUM_SEEDS

    print(f"Starting task n_id={n_id} with instance {instance_seed} and N_INTS={n_ints}")

    platform, instance_dict = build_instance(n_id, n_ints, seed=instance_seed)
    
    sim_result = run_single_simulation(platform)

    # Add metadata
    sim_result.update({
        "instance_id": n_id,
        "n_id": n_id,
        "n_ints": n_ints,
    })

    results_path = f"data/results_scaling_ints/instance_{instance_seed}/{n_id}.json"


    dir_name = os.path.dirname(results_path)

    # Create the directories
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    # Save results
    with open(results_path, "w") as f:
        json.dump(sim_result, f, indent=4, default=convert)

    print(f"Results saved to {results_path}")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()