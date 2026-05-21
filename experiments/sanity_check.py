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


# =========================
# CONFIG
# =========================
SIM_SIZE = 10
N_INTS = 14

GRAPH_PATH = "data/graph_0-14960_00.pickle"
RESULTS_PATH = f"data/results_sc/{n_id}.json"

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
# STOCHASTIC COMPONENTS
# =========================
def reset_quantities(platform, rng):
    return {
        farmer.id: min(
            max(
                np.floor((farmer.quantity + rng.uniform(-0.5, 0.5)) * 10) / 10,
                0.1,
            ),
            9.0,
        )
        for farmer in platform.farmers
    }


def reset_fixed_costs(platform, rng):
    return {
        intermediary.id: (
            platform.dist_to_mill[intermediary.id] * 4
            + rng.normal(0, 100000)
        )
        for intermediary in platform.intermediaries
    }


def sample_epsilon(platform, rng):
    return {
        intermediary.id: rng.uniform(0, 6.0)
        for intermediary in platform.intermediaries
    }


# =========================
# CORE BUILDERS
# =========================
def build_instance(instance_id, seed=n_id):
    """
    Generate a fresh instance and attach the road graph.
    """
    instance_generator.gen_ints(N_INTS, seed)

    instance_dict = instance_generator.gen_instance(
        instance_id,
        write=False,
        plot=False,
        seed=seed
    )

    platform = Instance.from_dict(instance_dict)
    platform.set_graph(RoadGraph(GRAPH))

    return platform, instance_dict


def apply_quantity_perturbation(instance_dict, platform, rng):
    """
    Rebuild platform with perturbed quantities.
    """
    quantities = reset_quantities(platform, rng)
    return Instance.from_dict(instance_dict, opt_quantities=quantities)


def run_single_simulation(instance_dict, platform, rng):
    """
    Runs one optimization on a (possibly perturbed) platform.
    """
    platform = apply_quantity_perturbation(instance_dict, platform, rng)

    platform.set_graph(RoadGraph(GRAPH))

    epsilon = sample_epsilon(platform, rng)
    het_costs = reset_fixed_costs(platform, rng)

    parameters = {
        "epsilon": epsilon,
        "solver": "gurobi",
        "het_costs": het_costs,
    }

    optimizer = Optimizer(platform, parameters)

    summary = optimizer.solve(
        "heuristic_optimized",
        options={
            "structured_farmer_prices": False,
            "domination": False,
        },
    )

    # Collect outputs
    return {
        "epsilon": epsilon,
        "cost": het_costs,
        "farmer_quantities": {f.id: f.quantity for f in platform.farmers},
        "summary_vanilla": summary.to_dict(),
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

    results = []

    print(f"Starting experiment batch n_id={n_id}")

    platform, instance_dict = build_instance(n_id, seed=n_id)

    rng = np.random.default_rng(n_id)

    for sim_n in range(1, SIM_SIZE + 1):
        print(f"--- Simulation {sim_n}/{SIM_SIZE} ---")

        instance_id = f"{n_id}_{sim_n}"

        # Run one stochastic solve
        sim_result = run_single_simulation(instance_dict, platform, rng)

        # Add metadata
        sim_result.update({
            "instance_id": instance_id,
            "n_id": n_id,
            "sim_n": sim_n,
        })

        results.append(sim_result)

        print(f"Completed simulation {sim_n}")
        print()

    # Save results
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=4, default=convert)

    print(f"Results saved to {RESULTS_PATH}")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()