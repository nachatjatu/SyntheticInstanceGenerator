#!/bin/bash
#SBATCH --job-name=pricing_exp6
#SBATCH --array=0-195
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --output=%x_%a.out
#SBATCH --error=%x_%a.err

# Load required modules
ml gurobi/11.0.2
ml python/3.12

# cd to repo root (wherever sbatch was submitted from)
cd $SLURM_SUBMIT_DIR

source .venv/bin/activate

# Run experiment for this array index
python3 experiments/experiment_6.py $SLURM_ARRAY_TASK_ID
