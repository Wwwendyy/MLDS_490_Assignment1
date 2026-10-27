"""Finite-horizon enumeration, value iteration, and policy iteration for HW1."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


N_STATES = 201
N_ACTIONS = 2  # 0 = wait, 1 = dispatch
ARRIVALS = np.arange(1, 6, dtype=np.int64)
ARRIVAL_PROBS = np.full(5, 0.2, dtype=np.float64)


def build_mdp(capacity=15, dispatch_cost=100.0, waiting_cost=2.0, max_waiting=200):
    """Return immediate cost C[s,a] and transition tensor P[s,a,s'].

    Timing convention: dispatch is decided at the start of an epoch. The
    immediate waiting cost is charged on people remaining after dispatch;
    arrivals then occur and join the next epoch. Excess arrivals at the stated
    station capacity are lost (state is clipped at max_waiting).
    """
    states = max_waiting + 1
    costs = np.zeros((states, N_ACTIONS), dtype=np.float64)
    transitions = np.zeros((states, N_ACTIONS, states), dtype=np.float64)

    for s in range(states):
        for a in range(N_ACTIONS):
            remaining = max(0, s - (capacity if a == 1 else 0))
            costs[s, a] = waiting_cost * remaining + (dispatch_cost if a == 1 else 0.0)
            for arrival, prob in zip(ARRIVALS, ARRIVAL_PROBS):
                next_s = min(max_waiting, remaining + int(arrival))
                transitions[s, a, next_s] += prob
    return costs, transitions


def finite_horizon(costs, transitions, gamma=0.95, horizon=500):
    """Cost-minimizing dynamic programming with terminal value V_T=0."""
    values = np.zeros((horizon + 1, costs.shape[0]), dtype=np.float64)
    policies = np.zeros((horizon, costs.shape[0]), dtype=np.int64)
    for t in range(horizon - 1, -1, -1):
        q = costs + gamma * np.einsum("sak,k->sa", transitions, values[t + 1])
        policies[t] = np.argmin(q, axis=1)
        values[t] = np.min(q, axis=1)
    return values, policies


def value_iteration(costs, transitions, gamma=0.95, tol=1e-10, max_iter=100_000):
    values = np.zeros(costs.shape[0], dtype=np.float64)
    for iteration in range(1, max_iter + 1):
        q = costs + gamma * np.einsum("sak,k->sa", transitions, values)
        updated = np.min(q, axis=1)
        delta = float(np.max(np.abs(updated - values)))
        values = updated
        if delta < tol:
            break
    else:
        raise RuntimeError("Value iteration did not converge")
    policy = np.argmin(costs + gamma * np.einsum("sak,k->sa", transitions, values), axis=1)
    return values, policy, iteration, delta


def policy_iteration(costs, transitions, gamma=0.95, max_iter=1_000):
    states = costs.shape[0]
    policy = np.zeros(states, dtype=np.int64)
    for iteration in range(1, max_iter + 1):
        p_pi = transitions[np.arange(states), policy]
        c_pi = costs[np.arange(states), policy]
        values = np.linalg.solve(np.eye(states) - gamma * p_pi, c_pi)
        q = costs + gamma * np.einsum("sak,k->sa", transitions, values)
        improved = np.argmin(q, axis=1)
        if np.array_equal(improved, policy):
            return values, policy, iteration
        policy = improved
    raise RuntimeError("Policy iteration did not stabilize")


def save_results(out_dir: Path, gamma=0.95, horizon=500):
    out_dir.mkdir(parents=True, exist_ok=True)
    costs, transitions = build_mdp()
    finite_v, finite_pi = finite_horizon(costs, transitions, gamma, horizon)
    vi_v, vi_pi, vi_iters, vi_delta = value_iteration(costs, transitions, gamma)
    pi_v, pi_pi, pi_iters = policy_iteration(costs, transitions, gamma)
    states = np.arange(N_STATES)

    np.savetxt(out_dir / "shuttle_values.csv",
               np.column_stack([states, finite_v[0], vi_v, pi_v]), delimiter=",",
               header="waiting_customers,finite_horizon_V0,value_iteration_V,policy_iteration_V", comments="")
    np.savetxt(out_dir / "shuttle_policy.csv",
               np.column_stack([states, vi_pi, pi_pi]), delimiter=",",
               header="waiting_customers,value_iteration_action,policy_iteration_action", comments="")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(states, finite_v[0], label=f"Enumeration, T={horizon}")
    ax.set(xlabel="Customers waiting", ylabel="Minimum discounted cost", title="Shuttle finite-horizon value at t=0")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "shuttle_enumeration_value.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(states, vi_v, label="Value iteration")
    ax.plot(states, pi_v, "--", label="Policy iteration")
    ax.set(xlabel="Customers waiting", ylabel="Minimum discounted cost", title="Shuttle infinite-horizon value")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "shuttle_infinite_value.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.step(states, vi_pi, where="post", label="Value iteration")
    ax.step(states, pi_pi, where="post", linestyle="--", label="Policy iteration")
    ax.set(xlabel="Customers waiting", ylabel="Action (0=wait, 1=dispatch)",
           title="Shuttle optimal policy", yticks=[0, 1], yticklabels=["Wait", "Dispatch"])
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "shuttle_policy.png", dpi=180); plt.close(fig)

    print(f"VI iterations={vi_iters}, final max delta={vi_delta:.3e}")
    print(f"PI iterations={pi_iters}")
    print(f"Policies agree: {np.array_equal(vi_pi, pi_pi)}")
    print(f"Max value difference (VI vs PI): {np.max(np.abs(vi_v - pi_v)):.3e}")
    print(f"Saved figures and CSVs to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/shuttle")
    parser.add_argument("--horizon", type=int, default=500)
    args = parser.parse_args()
    save_results(Path(args.out), horizon=args.horizon)
