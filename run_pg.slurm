#!/bin/bash
# Submit from /projects/e32706/way5959/MLDS_hw1 after installing .venv.
# Examples:
#   sbatch run_pg.slurm --env cartpole --baseline none --episodes 1000
#   sbatch run_pg.slurm --env pong --baseline value --episodes 3000
#SBATCH --account=e32706
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -e
cd /projects/e32706/way5959/MLDS_hw1
mkdir -p logs
source .conda_env/bin/activate
python -u policy_gradient.py "$@"
