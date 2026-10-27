# MLDS 490 - Assignment 1 Report

**Student:** GitHub username `Wwwendyy`  
**Repository:** [Wwwendyy/MLDS_490_Assignment1](https://github.com/Wwwendyy/MLDS_490_Assignment1)  
**Commit reviewed:** `d19cb83` (`Add MLDS Assignment 1 code and results`)

## 1. Shuttle dispatch MDP

### Model

The state is the number of customers waiting when a dispatch decision is made, (n\in\{0,\ldots,200\}\). The two actions are wait and dispatch. A dispatch boards up to 15 customers. The immediate cost is

\[
c(n,\text{wait})=2n,\qquad c(n,\text{dispatch})=100+2\max(n-15,0).
\]

After the decision, arrivals (A\) are uniform on \(\{1,2,3,4,5\}\), and the next queue is \(n'=\min(200,n-15\mathbf 1_{\{\text{dispatch}\}}+A)\). Arrivals over capacity are lost. The discount factor is \(\gamma=0.95\); the finite-horizon calculation uses \(T=500\) and terminal value zero. All algorithms use this same cost and transition convention. Values are discounted expected costs, so smaller is better.

### Methods and results

Finite-horizon enumeration computes the cost-to-go backward from horizon 500. Value iteration applies the Bellman optimality operator until the maximum update is below \(10^{-10}\); it converged in 519 iterations (final maximum delta \(9.913\times10^{-11}\)). Policy iteration converged in 6 policy improvements. Both infinite-horizon methods agree on the policy at all 201 states; their maximum value difference is \(1.865\times10^{-9}\). At every queue state, the finite-horizon value and converged infinite-horizon value differ by at most \(3.08\times10^{-9}\), indicating that the 500-step horizon is sufficient here.

The optimal policy waits for queues 0 through 12 and dispatches at queue 13 or above. The value increases with the queue because more customers incur waiting cost. This threshold behavior matches the capacity and fixed dispatch cost.

![Finite-horizon enumeration value by initial queue](https://raw.githubusercontent.com/Wwwendyy/MLDS_490_Assignment1/main/results/shuttle/shuttle_enumeration_value.png)

*Figure 1. Finite-horizon cost-to-go at time 0, for horizon 500.*

![Infinite-horizon value iteration and policy iteration](https://raw.githubusercontent.com/Wwwendyy/MLDS_490_Assignment1/main/results/shuttle/shuttle_infinite_value.png)

*Figure 2. Infinite-horizon optimal cost from value iteration and policy iteration.*

![Optimal Shuttle policy](https://raw.githubusercontent.com/Wwwendyy/MLDS_490_Assignment1/main/results/shuttle/shuttle_policy.png)

*Figure 3. Optimal action by queue; wait=0 and dispatch=1.*

### Edge cases and validation

`validate_shuttle.py` checks that each transition row is a probability distribution, probabilities are nonnegative, and all successor states remain in 0..200. It also checks the empty-queue costs, saturation at state 200, dispatch successors at the capacity boundary, and dispatch-cost sensitivity. Raising the fixed dispatch cost from $20 to $250 moves the first dispatch-optimal state from 7 to 15, as expected. At state 200, waiting remains at 200; dispatch leaves 185 onboard and produces a next queue from 186 to 190 after arrivals.

## 2. REINFORCE without a baseline

### Implementation

The policy-gradient implementation uses discounted returns \(G_t=\sum_{k\ge0}\gamma^k r_{t+k}\) and minimizes the REINFORCE loss \(-\sum_t\log\pi_\theta(a_t\mid s_t)G_t\). CartPole uses a 4-128-128-2 multilayer perceptron. Pong observations are cropped, downsampled to 80 by 80 grayscale frames and passed to a convolutional policy network. The Pong policy chooses between LEFT (ALE action 3) and RIGHT (ALE action 2); a fixed FIRE action (ALE action 1) starts each game after reset and is not learned. Training and evaluation sample from the categorical policy. Pong reports full-game episodic scores.

The reported final experiments use seed 42, learning rate 0.001, 1,000 training episodes for CartPole and 3,000 for Pong, followed by 500 stochastic evaluation episodes. CartPole uses \(\gamma=0.95\), Pong \(\gamma=0.99\). The 0.001 learning rate was used for the final comparisons; at 0.01, the saved CartPole runs remained near the minimum episode return, while the 0.001 runs performed substantially better.

### Evaluation results

| Environment | Train episodes | Evaluation episodes | Evaluation mean | Evaluation SD | Last-100 training mean |
|---|---:|---:|---:|---:|---:|
| CartPole-v1 | 1,000 | 500 | 67.300 | 11.510 | 115.760 |
| ALE/Pong-v5 | 3,000 | 500 | -20.470 | 0.758 | -20.470 |

The CartPole policy learns some balancing behavior, but its average evaluation episode is well below the 500-step cap. In Pong, the no-baseline policy remains near the game’s worst score (approximately -21); this run does not demonstrate a competitive Pong policy.

![CartPole REINFORCE without baseline training curve](https://raw.githubusercontent.com/Wwwendyy/MLDS_490_Assignment1/main/results/cartpole_pg_lr001_seed42/training_curve.png)

*Figure 4. CartPole training reward and 100-episode moving average, without a baseline.*

![CartPole REINFORCE without baseline evaluation histogram](https://raw.githubusercontent.com/Wwwendyy/MLDS_490_Assignment1/main/results/cartpole_pg_lr001_seed42/evaluation_histogram.png)

*Figure 5. CartPole stochastic evaluation returns, 500 episodes, without a baseline.*

![Pong REINFORCE without baseline training curve](https://raw.githubusercontent.com/Wwwendyy/MLDS_490_Assignment1/main/results/pong_pg_lr001_seed42_retry/training_curve.png)

*Figure 6. Pong training reward and 100-episode moving average, without a baseline.*

![Pong REINFORCE without baseline evaluation histogram](https://raw.githubusercontent.com/Wwwendyy/MLDS_490_Assignment1/main/results/pong_pg_lr001_seed42_retry/evaluation_histogram.png)

*Figure 7. Pong stochastic evaluation returns, 500 episodes, without a baseline.*

## 3. REINFORCE with a learned value baseline

### Implementation

The baseline version adds a state-value head to the policy network. It estimates \(V_\phi(s_t)\) and forms an advantage estimate \(A_t=G_t-V_\phi(s_t)\), with the value estimate detached in the policy loss. The policy loss is \(-\sum_t\log\pi_\theta(a_t\mid s_t)A_t\); a mean-squared error value loss trains the baseline, weighted by 0.5. Gradients are clipped to norm 1.0. The baseline changes the learning signal by subtracting a state-dependent estimate of expected return; it does not change the policy’s action space or the environment reward.

### Evaluation results

| Environment | Train episodes | Evaluation episodes | Evaluation mean | Evaluation SD | Last-100 training mean |
|---|---:|---:|---:|---:|---:|
| CartPole-v1 | 1,000 | 500 | 164.234 | 25.772 | 224.030 |
| ALE/Pong-v5 | 3,000 | 500 | -13.354 | 3.215 | -13.020 |

For this seed and these settings, adding the learned value baseline improves both evaluation means and the last-100 training means. CartPole improves by 96.934 evaluation reward points; Pong improves by 7.116 points. The Pong score is still negative, so this is a partial improvement rather than a solved game. Evaluation standard deviation is higher with the baseline in both comparisons (CartPole: 25.772 vs. 11.510; Pong: 3.215 vs. 0.758). These are dispersions of episodic evaluation returns, not estimates of gradient variance; the results therefore do not show lower evaluation-return variability with the baseline. A single seed cannot establish a general baseline effect.

![CartPole REINFORCE with learned value baseline training curve](https://raw.githubusercontent.com/Wwwendyy/MLDS_490_Assignment1/main/results/cartpole_baseline_lr001_seed42/training_curve.png)

*Figure 8. CartPole training reward and 100-episode moving average, with learned value baseline.*

![CartPole REINFORCE with learned value baseline evaluation histogram](https://raw.githubusercontent.com/Wwwendyy/MLDS_490_Assignment1/main/results/cartpole_baseline_lr001_seed42/evaluation_histogram.png)

*Figure 9. CartPole stochastic evaluation returns, 500 episodes, with learned value baseline.*

![Pong REINFORCE with learned value baseline training curve](https://raw.githubusercontent.com/Wwwendyy/MLDS_490_Assignment1/main/results/pong_pg_value_lr001_seed42/training_curve.png)

*Figure 10. Pong training reward and 100-episode moving average, with learned value baseline.*

![Pong REINFORCE with learned value baseline evaluation histogram](https://raw.githubusercontent.com/Wwwendyy/MLDS_490_Assignment1/main/results/pong_pg_value_lr001_seed42/evaluation_histogram.png)

*Figure 11. Pong stochastic evaluation returns, 500 episodes, with learned value baseline.*

## 4. Additional validation and limitations

The repository includes meaningful Shuttle checks and a short policy-gradient smoke run. Saved summaries confirm environment IDs, seeds, episode counts, discount factors, learning rates, stochastic evaluation, device, and the Pong startup FIRE setting. The environment loop ends an episode on either `terminated` or `truncated`. The Pong implementation restricts learned actions to ALE actions 2 and 3 after the fixed startup action. Debug runs and failed Slurm attempts are excluded from the reported results.

The policy-gradient comparisons use only one seed per setting. The evaluation SD describes variation across episodes for that trained policy; it is not a confidence interval across independent training runs. Results may vary with initialization and sampling. For a stronger comparison, repeat each configuration with at least five seeds, report mean and SD across seeds, and include the per-seed evaluation returns. Also compare stochastic and deterministic evaluation as a separate analysis, verify Pong action traces and reset/startup behavior, and add controlled checks for both termination and time-limit truncation.

## 5. Reproduction instructions

The following commands were run from the repository root in the Quest project. Install dependencies from `requirements.txt` in the project environment first. Pong training should run on a GPU compute node via Slurm, not on a login node.

```bash
python shuttle_mdp.py --out results/shuttle --horizon 500
python validate_shuttle.py

python policy_gradient.py --env cartpole --baseline none --episodes 1000 --eval-episodes 500 --seed 42 --lr 0.001 --out results/cartpole_pg_lr001_seed42
python policy_gradient.py --env cartpole --baseline value --episodes 1000 --eval-episodes 500 --seed 42 --lr 0.001 --out results/cartpole_baseline_lr001_seed42

sbatch run_pong.slurm --env pong --baseline none --episodes 3000 --eval-episodes 500 --seed 42 --lr 0.001 --out results/pong_pg_lr001_seed42_retry
sbatch run_pong.slurm --env pong --baseline value --episodes 3000 --eval-episodes 500 --seed 42 --lr 0.001 --out results/pong_pg_value_lr001_seed42
```

The Slurm script assumes the Quest project path and `.conda_env` shown in the repository setup. Each policy-gradient output directory contains `training_rewards.csv`, `training_curve.png`, `evaluation_rewards.csv`, `evaluation_histogram.png`, `summary.json`, and `policy.pt`. The selected run summaries and figures are committed to the GitHub repository.

## Repository

Source code, saved results, figures, and logs: [https://github.com/Wwwendyy/MLDS_490_Assignment1](https://github.com/Wwwendyy/MLDS_490_Assignment1).
