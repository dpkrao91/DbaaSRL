"""Smoke tests covering the workload, environment, and agents."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agents import LinearQAgent, LinearQConfig, TileCoder
from baselines import ThresholdScaler
from environment import DBaaSEnv, EnvConfig
from workload_generator import DEFAULT_TENANTS, generate_tenant_trace


def test_tile_coder_shape():
    tc = TileCoder(low=[0, 0], high=[10, 10], bins=4, num_tilings=3)
    idx = tc(np.array([5.0, 7.0]))
    assert idx.shape == (3,)
    assert (idx >= 0).all() and (idx < tc.num_tiles).all()


def test_linear_q_learns_on_constant_reward():
    obs_low = np.array([0.0])
    obs_high = np.array([1.0])
    agent = LinearQAgent(obs_low, obs_high, n_actions=2,
                         config=LinearQConfig(alpha=0.5, gamma=0.0,
                                              epsilon_start=0.0, epsilon_end=0.0,
                                              bins=4, num_tilings=2))
    obs = np.array([0.5])
    # Reward exceeds optimistic-initialisation value so action 1 is preferred.
    for _ in range(50):
        agent.update(obs, 1, reward=5.0, next_obs=obs, done=True)
    q = agent.q_values(obs)
    assert q[1] > q[0], f"Expected action 1 preferred, got {q}"


def test_environment_reward_signs():
    df = generate_tenant_trace(DEFAULT_TENANTS[0], n_minutes=200, seed=1)
    agg = pd.DataFrame({"minute": df["minute"], "total_qps": df["qps"]})
    env = DBaaSEnv(agg, EnvConfig())
    obs = env.reset()
    rewards = []
    for _ in range(50):
        obs, r, done, info = env.step(0)  # always smallest ACU
        rewards.append(r)
        if done:
            break
    assert all(r <= 0 for r in rewards), "Costs should be non-positive"


def test_threshold_scaler_increases_under_load():
    scaler = ThresholdScaler(acu_levels=(0.5, 1.0, 2.0, 4.0, 8.0))
    # High mean QPS, low ACU -> should scale up
    start = scaler.current_idx
    for _ in range(10):
        scaler.act(np.array([10_000.0, 10_000.0, 0.0, 1.0, 0.5, 1.5]))
    assert scaler.current_idx > start


if __name__ == "__main__":
    test_tile_coder_shape()
    test_linear_q_learns_on_constant_reward()
    test_environment_reward_signs()
    test_threshold_scaler_increases_under_load()
    print("All tests passed.")
