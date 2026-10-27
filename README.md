# MLDS HW1 starter implementation

This package implements the shuttle MDP (finite-horizon enumeration, value iteration, policy iteration) and REINFORCE for CartPole-v1 and ALE/Pong-v5, with either no baseline or a learned state-value baseline.

## Important modeling convention

For Shuttle, the state is the queue at a dispatch decision. Dispatching boards up to 15 people, the immediate cost is $2 per person remaining plus $100 if a shuttle is dispatched, arrivals are then sampled uniformly from {1,2,3,4,5}, and the next queue is clipped at 200. Overflow is treated as lost arrivals. The same convention is used in all three algorithms. These assumptions should be stated in the report.

For Pong, the learned policy has exactly two choices: RIGHT (ALE action 2) and LEFT (ALE action 3). A fixed FIRE (action 1) is issued once after reset to start the game; it is not selected by the network. This startup action is recorded in `summary.json` and should be mentioned in the report.

## Install on Quest

Run these commands inside your Quest project directory. GPU training should run on a Slurm compute node, not the login node.

```bash
cd /projects/e32706/way5959/MLDS_hw1
module load anaconda3/2022.05
conda create -p "$PWD/.conda_env" python=3.11 -y
conda activate "$PWD/.conda_env"
python -m pip install --upgrade pip
pip install -r requirements.txt
mkdir -p logs results
```

For PyTorch/CUDA compatibility, record `python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'` in your experiment notes. If it prints `False` on a GPU node, use the PyTorch install command matching Quest's CUDA setup before running Pong.

## Run Shuttle

```bash
python shuttle_mdp.py --out results/shuttle --horizon 500
python validate_shuttle.py
```

Outputs include three PNG figures, `shuttle_values.csv`, and `shuttle_policy.csv`. The validation script checks probability normalization, the capped state boundary, the dispatch cost, and a dispatch-cost sensitivity experiment.

## Run CartPole

Smoke run (debug only; do not use as final result):

```bash
python policy_gradient.py --env cartpole --baseline none --episodes 10 --eval-episodes 5 --out results/debug_cartpole
```

Full experiments (each writes a 500-episode evaluation by default):

```bash
python policy_gradient.py --env cartpole --baseline none --episodes 1000 --eval-episodes 500 --seed 42 --out results/cartpole_pg_seed42
python policy_gradient.py --env cartpole --baseline value --episodes 1000 --eval-episodes 500 --seed 42 --out results/cartpole_baseline_seed42
```

## Run Pong

Use a GPU compute node. The required 500-episode evaluation can take substantial time.

```bash
python policy_gradient.py --env pong --baseline none --episodes 3000 --eval-episodes 500 --seed 42 --out results/pong_pg_seed42
python policy_gradient.py --env pong --baseline value --episodes 3000 --eval-episodes 500 --seed 42 --out results/pong_baseline_seed42
```

Alternatively submit each run through Slurm (first create `logs/`):

```bash
mkdir -p logs
sbatch run_pg.slurm --env pong --baseline none --episodes 3000 --eval-episodes 500 --seed 42 --out results/pong_pg_seed42
sbatch run_pg.slurm --env pong --baseline value --episodes 3000 --eval-episodes 500 --seed 42 --out results/pong_baseline_seed42
squeue -u "$USER"
```

For CartPole, replace `pong` by `cartpole` and use `--episodes 1000`.

## Generated outputs

Each policy-gradient run writes:

- `training_rewards.csv`: episode reward and 100-episode moving average;
- `training_curve.png`: required training plot;
- `evaluation_rewards.csv`: per-episode scores for the evaluation rollouts;
- `evaluation_histogram.png`: evaluation reward histogram;
- `summary.json`: mean, standard deviation, hyperparameters, seed, and device;
- `policy.pt`: trained policy weights.

The script defaults to stochastic policy sampling during evaluation. Add `--deterministic-eval` only if you want an additional argmax-policy analysis; report which mode was used. The implementation uses the required learning rate 0.01 and gamma 0.95 (CartPole) / 0.99 (Pong) by default.

## Validation items to discuss in the report

1. Shuttle: probability rows sum to one; queue never leaves 0..200; changing dispatch cost changes the dispatch threshold in the expected direction; VI and PI policies agree and values are close.
2. CartPole/Pong: record environment ID, action/observation shape, seed, terminated/truncated handling, and confirm Pong actions map only to 2 or 3 after fixed startup FIRE.
3. Compare baseline vs no-baseline training curves and evaluation standard deviations. For stronger evidence, repeat each final setting with several seeds and report mean and spread across seeds.
4. A short smoke run confirms the pipeline executes, but is not a substitute for algorithm-behavior validation or the required 500-episode evaluation.
