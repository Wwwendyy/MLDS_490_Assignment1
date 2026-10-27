#!/bin/bash
#SBATCH --job-name=pong_pg
#SBATCH --account=e32706
#SBATCH --partition=gengpu,gengpu-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/e32706/way5959/MLDS_hw1/logs/%x-%j.out

set -eo pipefail

cd /projects/e32706/way5959/MLDS_hw1
mkdir -p logs results

module load anaconda3/2022.05
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

PYTHON=/projects/e32706/way5959/MLDS_hw1/.conda_env/bin/python
echo "Host: $(hostname)"
echo "Python: $PYTHON"
"$PYTHON" -c "import numpy, torch; print('NumPy:', numpy.__version__, numpy.__file__); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

"$PYTHON" -u policy_gradient.py "$@"
