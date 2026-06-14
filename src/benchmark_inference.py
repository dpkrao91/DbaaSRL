"""Benchmark per-step inference latency for each agent.

This script is what backs the paper's central "lightweight" claim:
how many microseconds does each agent need to choose an action?
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from agents import DQNAgent, DQNConfig, LinearQAgent, LinearQConfig
from baselines import ARIMAScaler, ThresholdScaler


N_TRIALS = 50_000
ACU_LEVELS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)


def benchmark(agent, label: str, obs_samples: np.ndarray) -> dict:
    # Warm-up to avoid first-call overhead.
    for i in range(100):
        agent.act(obs_samples[i % len(obs_samples)], greedy=True)
    t0 = time.perf_counter()
    for i in range(N_TRIALS):
        agent.act(obs_samples[i % len(obs_samples)], greedy=True)
    elapsed = time.perf_counter() - t0
    per_step_us = elapsed * 1e6 / N_TRIALS
    return {"agent": label, "per_step_us": per_step_us, "n_trials": N_TRIALS}


def main():
    rng = np.random.default_rng(0)
    obs_samples = np.column_stack(
        [
            rng.uniform(0, 2000, 1000),
            rng.uniform(0, 2500, 1000),
            rng.uniform(-200, 200, 1000),
            rng.choice(ACU_LEVELS, 1000),
            rng.uniform(0, 1, 1000),
            rng.uniform(0, 2, 1000),
        ]
    ).astype(np.float32)

    low = np.array([0, 0, -2000, 0, 0, 0], dtype=np.float32)
    high = np.array([2500, 2500, 200, 64, 1.0, 2.0], dtype=np.float32)

    linq = LinearQAgent(low, high, len(ACU_LEVELS), LinearQConfig(seed=0))
    dqn = DQNAgent(6, len(ACU_LEVELS), DQNConfig(seed=0))
    thr = ThresholdScaler(ACU_LEVELS)
    ari = ARIMAScaler(ACU_LEVELS)

    rows = [
        benchmark(linq, "LinearQ", obs_samples),
        benchmark(dqn,  "DQN",     obs_samples),
        benchmark(thr,  "Threshold", obs_samples),
        benchmark(ari,  "ARIMA",   obs_samples),
    ]
    df = pd.DataFrame(rows)
    df["relative_speedup_vs_dqn"] = df.loc[df["agent"] == "DQN", "per_step_us"].iloc[0] / df["per_step_us"]
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/inference_benchmark.csv", index=False)
    print(df.to_string(index=False))
    with open("results/inference_benchmark.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
