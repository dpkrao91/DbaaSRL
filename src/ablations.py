"""Real ablation experiments for the Linear-Q agent.

Runs three configurations against the same train and eval traces:

  1. "default":            B=4 bins, K=3 tilings, random tie-break, random init
  2. "no_random_tiebreak": replaces random argmax tie-break with np.argmax
  3. "bins_8":             B=8 bins (64x larger state space)

Each is trained and evaluated identically to ``train.py``'s LinearQ branch.
Results are written to ``results/ablations.csv``.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from agents import LinearQAgent, LinearQConfig
from environment import DBaaSEnv, EnvConfig
from train import evaluate, obs_bounds, load_workload


# ---------------------------------------------------------------- variants


class NoRandomTiebreakLinearQ(LinearQAgent):
    """LinearQ variant using deterministic first-index argmax tie-break."""

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:  # type: ignore[override]
        import random as _random
        if not greedy and self.rng.random() < self._epsilon():
            return self.rng.randrange(self.n_actions)
        q = self.q_values(obs)
        return int(np.argmax(q))


# ---------------------------------------------------------------- runner


def train_and_eval(agent, train_env, eval_env, episodes, episode_length, label, online=True):
    t0 = time.time()
    train_rewards = []
    for ep in range(episodes):
        obs = train_env.reset()
        total = 0.0
        for _ in range(episode_length):
            a = agent.act(obs)
            nobs, r, done, _ = train_env.step(a)
            agent.update(obs, a, r, nobs, done)
            obs = nobs
            total += r
            if done:
                break
        train_rewards.append(total)
    train_t = time.time() - t0

    trace = evaluate(eval_env, agent, online=online)
    return {
        "variant": label,
        "params": agent.num_parameters(),
        "train_seconds": train_t,
        "final_train_reward_med20": float(np.median(train_rewards[-20:])),
        "mean_latency_ms": float(trace["latency_ms"].mean()),
        "p99_latency_ms": float(trace["latency_ms"].quantile(0.99)),
        "sla_violation_rate": float((trace["latency_ms"] > 100.0).mean()),
        "mean_acu": float(trace["acu"].mean()),
        "total_cost": float(trace["cost"].sum()),
    }


def main():
    seed = 42
    episodes = 100
    episode_length = 720

    train_wl = load_workload("data/workload_aggregate.csv")
    eval_wl = load_workload("data/workload_aggregate_eval.csv")

    env_cfg = EnvConfig()
    train_env = DBaaSEnv(train_wl, env_cfg)
    eval_env = DBaaSEnv(eval_wl, env_cfg)
    low, high = obs_bounds(train_env)
    n_actions = train_env.n_actions
    decay = int(episodes * episode_length * 0.5)

    rows = []

    # 1) default
    cfg = LinearQConfig(seed=seed, epsilon_decay_steps=decay)
    agent = LinearQAgent(low, high, n_actions, cfg)
    rows.append(train_and_eval(agent,
                               DBaaSEnv(train_wl, env_cfg),
                               DBaaSEnv(eval_wl, env_cfg),
                               episodes, episode_length, "default"))
    print(rows[-1])

    # 2) no random tie-break
    cfg = LinearQConfig(seed=seed, epsilon_decay_steps=decay)
    agent = NoRandomTiebreakLinearQ(low, high, n_actions, cfg)
    rows.append(train_and_eval(agent,
                               DBaaSEnv(train_wl, env_cfg),
                               DBaaSEnv(eval_wl, env_cfg),
                               episodes, episode_length, "no_random_tiebreak"))
    print(rows[-1])

    # 3) bins=8 (much larger state space)
    cfg = LinearQConfig(seed=seed, epsilon_decay_steps=decay, bins=8, num_tilings=3)
    agent = LinearQAgent(low, high, n_actions, cfg)
    rows.append(train_and_eval(agent,
                               DBaaSEnv(train_wl, env_cfg),
                               DBaaSEnv(eval_wl, env_cfg),
                               episodes, episode_length, "bins_8"))
    print(rows[-1])

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/ablations.csv", index=False)
    print("\n=== Ablation Summary ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
