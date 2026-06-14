"""
Serverless DBaaS Environment for Reinforcement Learning.

This module implements a discrete-time simulator of an Aurora-Serverless-v2-like
elastic database instance. The agent observes recent workload statistics and
selects a target capacity (in Aurora Capacity Units, ACUs). The environment
returns a reward that balances:

  * SLA latency violations (penalty)
  * Cloud cost (penalty proportional to ACU-minutes)
  * Cold-start delay when capacity is increased from a low baseline

The state, action, and reward design follow a Markov Decision Process (MDP)
formulation suitable for tabular or shallow neural agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd


# Discrete action set: each value is the *target* ACU count.
# Aurora Serverless v2 currently allows 0.5..128 ACUs in 0.5 increments.
DEFAULT_ACU_LEVELS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)

# Each ACU handles roughly 50 QPS at moderate utilisation, derived from
# public AWS benchmark documentation for db.r6g instances.
QPS_PER_ACU = 50.0

# Cold-start latency penalty applied when scaling up from a paused state.
# AWS reports 5-30 seconds for v2; we use the optimistic end.
COLD_START_SECONDS = 5.0

# Pricing: 1 ACU-hour ~= $0.12 on us-east-1 list price.
COST_PER_ACU_MINUTE = 0.12 / 60.0


@dataclass
class EnvConfig:
    """Hyperparameters controlling the simulation."""

    window_minutes: int = 10        # observation lookback window
    sla_latency_ms: float = 100.0   # cluster-wide SLA target
    sla_penalty: float = 25.0       # reward penalty per normalised ms over SLA
    cost_weight: float = 1.0        # multiplier on $ cost in reward
    cold_start_penalty: float = 10.0 # one-shot penalty when scaling from <1 ACU
    acu_levels: Tuple[float, ...] = DEFAULT_ACU_LEVELS


@dataclass
class EnvState:
    """Internal environment state at each step."""

    t: int = 0
    current_acu: float = 2.0
    last_action: int = 0
    history: list = field(default_factory=list)


class DBaaSEnv:
    """Gym-style environment for the serverless DBaaS scaling MDP."""

    def __init__(self, workload_df: pd.DataFrame, config: EnvConfig | None = None):
        self.workload = workload_df.copy().reset_index(drop=True)
        self.config = config or EnvConfig()
        self.state = EnvState(current_acu=self.config.acu_levels[2])
        self.n_actions = len(self.config.acu_levels)
        # Observation: [mean_qps, max_qps, qps_trend, current_acu, time_of_day, utilisation]
        self.observation_dim = 6

    # ----------------------------------------------------------------- API

    def reset(self) -> np.ndarray:
        self.state = EnvState(current_acu=self.config.acu_levels[2])
        return self._observation()

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Apply an action and advance the simulator by one minute."""
        cfg = self.config
        previous_acu = self.state.current_acu
        target_acu = cfg.acu_levels[int(action_idx)]

        # Cold-start dynamics: scaling from a near-paused state incurs a penalty
        # and slower convergence.
        cold_start = previous_acu < 1.0 and target_acu >= 1.0
        if cold_start:
            # Capacity arrives partway through this minute.
            effective_acu = 0.5 * previous_acu + 0.5 * target_acu
        else:
            # Vertical scaling is rapid in v2 -- effectively immediate.
            effective_acu = target_acu

        self.state.current_acu = target_acu

        # Realised workload for this minute
        row = self.workload.iloc[self.state.t]
        qps = float(row["total_qps"])
        capacity = effective_acu * QPS_PER_ACU
        utilisation = min(qps / max(capacity, 1e-6), 0.999)

        # M/M/c-style latency curve
        latency_ms = 5.0 + 45.0 * (utilisation / (1 - utilisation)) ** 0.5
        latency_ms = float(np.clip(latency_ms, 5.0, 5000.0))
        sla_violation = max(0.0, latency_ms - cfg.sla_latency_ms)

        # Reward components -- scaled to comparable magnitudes for stable learning.
        # SLA penalty saturates at the full violation magnitude (capped).
        sla_violation_norm = min(sla_violation / 100.0, 50.0)  # cap extreme penalties
        sla_cost = cfg.sla_penalty * sla_violation_norm
        money_cost = cfg.cost_weight * effective_acu * COST_PER_ACU_MINUTE * 50.0
        cold_cost = cfg.cold_start_penalty if cold_start else 0.0

        reward = -(sla_cost + money_cost + cold_cost)

        info = {
            "qps": qps,
            "acu": effective_acu,
            "latency_ms": latency_ms,
            "sla_violation_ms": sla_violation,
            "cost": money_cost,
            "cold_start": cold_cost > 0,
        }
        self.state.history.append(info)
        self.state.last_action = int(action_idx)
        self.state.t += 1
        done = self.state.t >= len(self.workload) - 1

        return self._observation(), float(reward), done, info

    # ------------------------------------------------------------- helpers

    def _observation(self) -> np.ndarray:
        """Compact observation vector from the lookback window."""
        cfg = self.config
        t = self.state.t
        lo = max(0, t - cfg.window_minutes)
        window = self.workload.iloc[lo : t + 1]["total_qps"].values
        if len(window) == 0:
            window = np.array([float(self.workload.iloc[0]["total_qps"])])
        mean_qps = float(np.mean(window))
        max_qps = float(np.max(window))
        if len(window) >= 2:
            trend = float(window[-1] - window[0]) / max(len(window), 1)
        else:
            trend = 0.0
        minute_of_day = (t % (24 * 60)) / (24 * 60)
        capacity = max(self.state.current_acu, 0.5) * QPS_PER_ACU
        utilisation_obs = float(min(mean_qps / capacity, 2.0))
        return np.array(
            [mean_qps, max_qps, trend, self.state.current_acu,
             minute_of_day, utilisation_obs],
            dtype=np.float32,
        )

    def history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.state.history)
