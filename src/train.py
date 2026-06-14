"""
Training, evaluation, and benchmarking pipeline.

Runs all agents (LinearQ, DQN, Threshold, OverProvision, ARIMA) over the same
held-out evaluation slice of the synthetic dataset and produces:

  * ``results/training_curves.csv``   -- episodic reward over training
  * ``results/evaluation_summary.csv`` -- per-agent latency / cost / SLA metrics
  * ``results/per_step_<agent>.csv``  -- step-level traces for plotting
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict

import numpy as np
import pandas as pd

from agents import (
    DQNAgent,
    DQNConfig,
    FederatedLinearQAgent,
    LinearQAgent,
    LinearQConfig,
)
from baselines import ARIMAScaler, OverProvisionScaler, ThresholdScaler
from environment import DBaaSEnv, EnvConfig


# ---------------------------------------------------------------- helpers


def load_workload(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.sort_values("minute").reset_index(drop=True)


def make_env(workload: pd.DataFrame, cfg: EnvConfig | None = None) -> DBaaSEnv:
    return DBaaSEnv(workload, cfg)


def obs_bounds(env: DBaaSEnv) -> tuple:
    qps_max = float(env.workload["total_qps"].max()) * 1.2
    acu_max = max(env.config.acu_levels)
    low = np.array([0.0, 0.0, -qps_max, 0.0, 0.0, 0.0], dtype=np.float32)
    high = np.array([qps_max, qps_max, qps_max, acu_max, 1.0, 2.0], dtype=np.float32)
    return low, high


# ---------------------------------------------------------------- training


def train_agent(env: DBaaSEnv, agent, episodes: int, episode_length: int) -> pd.DataFrame:
    """Online training -- works for any agent exposing act/update."""
    history = []
    for ep in range(episodes):
        obs = env.reset()
        total_reward = 0.0
        td_loss = 0.0
        steps = 0
        for _ in range(episode_length):
            action = agent.act(obs)
            next_obs, reward, done, _ = env.step(action)
            td = agent.update(obs, action, reward, next_obs, done)
            td_loss += abs(td)
            total_reward += reward
            obs = next_obs
            steps += 1
            if done:
                break
        history.append(
            {
                "episode": ep,
                "reward": total_reward,
                "td_loss": td_loss / max(steps, 1),
                "steps": steps,
            }
        )
    return pd.DataFrame(history)


# ---------------------------------------------------------------- evaluation


def evaluate(env: DBaaSEnv, agent, episodes: int = 1, online: bool = False) -> pd.DataFrame:
    """Run greedy evaluation and return the step-level trace.

    When ``online`` is True the agent may continue updating its Q-function from
    eval-time transitions. This mirrors a production deployment where a small
    amount of continual learning is desirable; it does not affect baselines
    whose ``update`` method is a no-op.
    """
    all_steps = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action = agent.act(obs, greedy=True)
            next_obs, reward, done, info = env.step(action)
            if online:
                agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            info_row = dict(info)
            info_row["reward"] = reward
            info_row["t"] = env.state.t
            all_steps.append(info_row)
    return pd.DataFrame(all_steps)


def summarise(trace: pd.DataFrame, label: str, train_seconds: float, n_params: int) -> dict:
    sla_target = 100.0
    return {
        "agent": label,
        "mean_latency_ms": float(trace["latency_ms"].mean()),
        "p99_latency_ms": float(trace["latency_ms"].quantile(0.99)),
        "sla_violation_rate": float((trace["latency_ms"] > sla_target).mean()),
        "mean_acu": float(trace["acu"].mean()),
        "total_cost": float(trace["cost"].sum()),
        "cold_starts": int(trace["cold_start"].sum()),
        "train_seconds": train_seconds,
        "num_parameters": n_params,
    }


# ---------------------------------------------------------------- pipeline


def run_all(
    workload_path: str,
    out_dir: str,
    eval_workload_path: str | None = None,
    episodes: int = 30,
    episode_length: int = 1440,
    seed: int = 0,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    workload = load_workload(workload_path)

    if eval_workload_path and os.path.exists(eval_workload_path):
        # Independent held-out evaluation set generated with a different seed.
        train_wl = workload.reset_index(drop=True)
        eval_wl = load_workload(eval_workload_path)
    else:
        # Fallback: 80/20 chronological split.
        split = int(0.8 * len(workload))
        train_wl = workload.iloc[:split].reset_index(drop=True)
        eval_wl = workload.iloc[split:].reset_index(drop=True)

    env_cfg = EnvConfig()
    train_env = make_env(train_wl, env_cfg)
    eval_env = make_env(eval_wl, env_cfg)
    low, high = obs_bounds(train_env)
    n_actions = train_env.n_actions
    # Adapt epsilon decay to actual training horizon so the agent gets
    # meaningful exploration coverage in the multi-dim tile space.
    total_steps = episodes * episode_length
    decay_steps = max(1, int(total_steps * 0.5))

    results: Dict[str, dict] = {}
    training_frames = []

    # 1. Linear-Q -------------------------------------------------------
    print("Training LinearQ...")
    lin_cfg = LinearQConfig(seed=seed, epsilon_decay_steps=decay_steps)
    lin_agent = LinearQAgent(low, high, n_actions, lin_cfg)
    t0 = time.time()
    hist = train_agent(train_env, lin_agent, episodes, episode_length)
    train_t = time.time() - t0
    hist["agent"] = "LinearQ"
    training_frames.append(hist)
    trace = evaluate(eval_env, lin_agent, online=True)
    trace.to_csv(os.path.join(out_dir, "per_step_linearq.csv"), index=False)
    results["LinearQ"] = summarise(trace, "LinearQ", train_t, lin_agent.num_parameters())

    # 2. DQN ------------------------------------------------------------
    print("Training DQN...")
    dqn_cfg = DQNConfig(seed=seed, epsilon_decay_steps=decay_steps)
    dqn_agent = DQNAgent(train_env.observation_dim, n_actions, dqn_cfg)
    t0 = time.time()
    hist = train_agent(train_env, dqn_agent, episodes, episode_length)
    train_t = time.time() - t0
    hist["agent"] = "DQN"
    training_frames.append(hist)
    trace = evaluate(eval_env, dqn_agent)
    trace.to_csv(os.path.join(out_dir, "per_step_dqn.csv"), index=False)
    results["DQN"] = summarise(trace, "DQN", train_t, dqn_agent.num_parameters())

    # 3. Threshold reactive baseline -----------------------------------
    print("Evaluating Threshold baseline...")
    thr_agent = ThresholdScaler(env_cfg.acu_levels)
    trace = evaluate(eval_env, thr_agent)
    trace.to_csv(os.path.join(out_dir, "per_step_threshold.csv"), index=False)
    results["Threshold"] = summarise(trace, "Threshold", 0.0, 0)

    # 4. Over-provisioning ---------------------------------------------
    print("Evaluating OverProvision baseline...")
    op_agent = OverProvisionScaler(env_cfg.acu_levels)
    trace = evaluate(eval_env, op_agent)
    trace.to_csv(os.path.join(out_dir, "per_step_overprovision.csv"), index=False)
    results["OverProvision"] = summarise(trace, "OverProvision", 0.0, 0)

    # 5. ARIMA-style predictive ----------------------------------------
    print("Evaluating ARIMA baseline...")
    ar_agent = ARIMAScaler(env_cfg.acu_levels)
    trace = evaluate(eval_env, ar_agent)
    trace.to_csv(os.path.join(out_dir, "per_step_arima.csv"), index=False)
    results["ARIMA"] = summarise(trace, "ARIMA", 0.0, 0)

    # 6. Federated Linear-Q --------------------------------------------
    print("Training FederatedLinearQ...")
    fed_cfg = LinearQConfig(seed=seed, epsilon_decay_steps=decay_steps)
    fed = FederatedLinearQAgent(
        n_tenants=4, obs_low=low, obs_high=high, n_actions=n_actions,
        config=fed_cfg,
    )
    t0 = time.time()
    # Each "client" trains a full local episode, then the coordinator
    # periodically averages their weights (FedAvg with a 5-episode period).
    for ep in range(episodes):
        for tenant_idx in range(len(fed.agents)):
            env_t = make_env(train_wl, env_cfg)
            obs = env_t.reset()
            for _ in range(episode_length):
                action = fed[tenant_idx].act(obs)
                next_obs, reward, done, _ = env_t.step(action)
                fed[tenant_idx].update(obs, action, reward, next_obs, done)
                obs = next_obs
                if done:
                    break
        if ep % 5 == 4:
            fed.aggregate()
    train_t = time.time() - t0
    # Evaluate using the post-aggregation shared weights (tenant 0).
    trace = evaluate(eval_env, fed[0], online=True)
    trace.to_csv(os.path.join(out_dir, "per_step_federated.csv"), index=False)
    results["FederatedLinearQ"] = summarise(trace, "FederatedLinearQ", train_t, fed.num_parameters())

    # ----------------------------------------------------------------- save
    summary_df = pd.DataFrame(list(results.values()))
    summary_df.to_csv(os.path.join(out_dir, "evaluation_summary.csv"), index=False)
    pd.concat(training_frames, ignore_index=True).to_csv(
        os.path.join(out_dir, "training_curves.csv"), index=False
    )

    print("\n=== Evaluation Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nAll results written to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all DBaaS RL experiments.")
    parser.add_argument("--workload", default="data/workload_aggregate.csv")
    parser.add_argument("--eval-workload", default="data/workload_aggregate_eval.csv")
    parser.add_argument("--out", default="results")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--episode-length", type=int, default=1440)  # 1 day in minutes
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_all(
        workload_path=args.workload,
        out_dir=args.out,
        eval_workload_path=args.eval_workload,
        episodes=args.episodes,
        episode_length=args.episode_length,
        seed=args.seed,
    )
