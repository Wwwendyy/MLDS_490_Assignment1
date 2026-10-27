"""Behavioral checks and small sensitivity experiments for shuttle_mdp.py."""
from __future__ import annotations

import numpy as np
from shuttle_mdp import build_mdp, value_iteration


def dispatch_threshold(policy):
    selected = np.flatnonzero(policy == 1)
    return int(selected[0]) if len(selected) else None


def main():
    gamma = 0.95
    base_cost, base_p = build_mdp()

    # Probability mass and state-boundary checks.
    assert np.allclose(base_p.sum(axis=-1), 1.0)
    assert np.all(base_p >= 0.0)
    next_states = np.arange(base_p.shape[-1])
    assert next_states.max() == 200
    assert np.all(base_p[:, :, 201:] == 0.0)
    # No one boards when dispatching from an empty station; dispatch fixed cost applies.
    assert base_cost[0, 0] == 0.0
    assert base_cost[0, 1] == 100.0
    # At capacity, wait keeps the next state capped; dispatch removes 15 before arrivals.
    assert base_p[200, 0, 200] == 1.0
    assert base_p[200, 1, 186:191].sum() == 1.0

    low_cf_cost, low_cf_p = build_mdp(dispatch_cost=20.0)
    high_cf_cost, high_cf_p = build_mdp(dispatch_cost=250.0)
    _, low_cf_policy, _, _ = value_iteration(low_cf_cost, low_cf_p, gamma)
    _, high_cf_policy, _, _ = value_iteration(high_cf_cost, high_cf_p, gamma)
    low_threshold = dispatch_threshold(low_cf_policy)
    high_threshold = dispatch_threshold(high_cf_policy)
    print("Shuttle transition rows sum to 1 and remain within the 0..200 state space.")
    print(f"Low dispatch cost ($20): first dispatch state = {low_threshold}")
    print(f"High dispatch cost ($250): first dispatch state = {high_threshold}")
    print("Expected behavior: raising dispatch cost should not make dispatch optimal at a lower queue.")
    assert low_threshold is not None and high_threshold is not None
    assert high_threshold >= low_threshold


if __name__ == "__main__":
    main()
