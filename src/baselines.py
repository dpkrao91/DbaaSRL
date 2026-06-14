"""
Baseline scaling policies used as comparison points for the RL agent.

  * ``ThresholdScaler`` -- mimics Aurora Serverless v2 / RDS reactive autoscaling
    using utilisation thresholds.

  * ``OverProvisionScaler`` -- always allocates the maximum ACU level. Acts as
    an upper-bound on SLA compliance and a lower-bound on cost-efficiency.

  * ``ARIMAScaler`` -- a moving-average predictive scaler representing the
    "traditional ML" comparison point. Heavier to run but does not learn from
    reward feedback.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np


class ThresholdScaler:
    """Reactive autoscaler with scale-up/down CPU utilisation thresholds."""

    def __init__(
        self,
        acu_levels: Sequence[float],
        scale_up_util: float = 0.70,
        scale_down_util: float = 0.30,
        cooldown_minutes: int = 5,
    ):
        self.acu_levels = list(acu_levels)
        self.scale_up_util = scale_up_util
        self.scale_down_util = scale_down_util
        self.cooldown = cooldown_minutes
        self.current_idx = 2
        self.last_scale_t = -10**6
        self.t = 0

    def act(self, obs: np.ndarray, greedy: bool = True) -> int:
        mean_qps, _, _, current_acu, _, _ = obs
        capacity = max(current_acu, 0.5) * 50.0
        util = float(mean_qps) / capacity
        if self.t - self.last_scale_t >= self.cooldown:
            if util > self.scale_up_util and self.current_idx < len(self.acu_levels) - 1:
                self.current_idx += 1
                self.last_scale_t = self.t
            elif util < self.scale_down_util and self.current_idx > 0:
                self.current_idx -= 1
                self.last_scale_t = self.t
        self.t += 1
        return self.current_idx

    def update(self, *args, **kwargs) -> float:  # noqa: D401
        """No-op -- threshold scalers do not learn."""
        return 0.0


class OverProvisionScaler:
    """Always-max-ACU scaler. Gives an SLA upper bound at maximum cost."""

    def __init__(self, acu_levels: Sequence[float]):
        self.idx = len(acu_levels) - 1

    def act(self, obs: np.ndarray, greedy: bool = True) -> int:
        return self.idx

    def update(self, *args, **kwargs) -> float:
        return 0.0


class ARIMAScaler:
    """Moving-average + linear-trend predictive scaler.

    Used as a stand-in for an ARIMA(p, d, q) predictor: at each step it fits
    a simple AR(1)+trend model over the recent window and provisions enough
    ACUs to handle the predicted peak with a 20% safety margin.

    Heavier than a tile-coded RL agent because it refits each step, mirroring
    how production ML-based autoscalers work today.
    """

    def __init__(self, acu_levels: Sequence[float], window: int = 30, safety: float = 1.2):
        self.acu_levels = list(acu_levels)
        self.window = window
        self.safety = safety
        self.history: deque = deque(maxlen=window)

    def _predict(self) -> float:
        if len(self.history) < 3:
            return float(self.history[-1]) if self.history else 100.0
        y = np.array(self.history, dtype=np.float32)
        x = np.arange(len(y), dtype=np.float32)
        # Least-squares linear fit -> extrapolate one step ahead.
        slope, intercept = np.polyfit(x, y, 1)
        return float(slope * (len(y) + 1) + intercept)

    def act(self, obs: np.ndarray, greedy: bool = True) -> int:
        mean_qps, max_qps, _, _, _, _ = obs
        self.history.append(float(max_qps))
        predicted = self._predict() * self.safety
        required_acu = predicted / 50.0
        # Snap up to the next discrete ACU level.
        for i, level in enumerate(self.acu_levels):
            if level >= required_acu:
                return i
        return len(self.acu_levels) - 1

    def update(self, *args, **kwargs) -> float:
        return 0.0
