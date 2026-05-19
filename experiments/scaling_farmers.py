import sys
import json
import pickle
import numpy as np
import pandas as pd
import random

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

def set_reproducible_state(seed_val):
    np.random.seed(seed_val)
    random.seed(seed_val)
    
set_reproducible_state(n_id)


# =========================
# CONFIG
# =========================
SCALE_FACTORS_LIST = [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
N_INTS = 15

GRAPH_PATH = "data/graph_0-14960_00.pickle"
RESULTS_PATH = f"data/results_scaling_ints/{n_id}.json"

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
def build_instance(instance_id, scale_factor):
    """
    Generate a fresh instance and attach the road graph.
    """
    instance_generator.gen_ints(N_INTS)

    instance_dict = instance_generator.gen_instance(
        instance_id,
        write=False,
        plot=False,
        scale_factor=scale_factor
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



def run_single_simulation(instance_dict, platform):
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
    scaling_factor_idx = (n_id - 1) % len(SCALE_FACTORS_LIST)
    selected_scale_factor = SCALE_FACTORS_LIST[scaling_factor_idx]

    print(f"Starting task n_id={n_id} with scale_factor={selected_scale_factor}")

    instance_id = n_id

    platform, instance_dict = build_instance(instance_id, selected_scale_factor)

    sim_result = run_single_simulation(instance_dict, platform)

    # Add metadata
    sim_result.update({
        "instance_id": instance_id,
        "n_id": n_id,
        "scale_factor": selected_scale_factor,
    })


    # Save results
    with open(RESULTS_PATH, "w") as f:
        json.dump(sim_result, f, indent=4, default=convert)

    print(f"Results saved to {RESULTS_PATH}")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()