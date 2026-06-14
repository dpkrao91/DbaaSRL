"""
Reinforcement Learning Agents for Serverless DBaaS Scaling.

We implement three agents:

  1. ``LinearQAgent`` -- the core proposal of this paper.
     A tile-coded linear function approximator trained with Q-learning.
     Inference cost is O(num_tiles_active * num_actions), making it suitable
     to run *inside* the serverless control plane without negating cost
     savings. Memory footprint is < 1 MB.

  2. ``DQNAgent`` -- a deep neural baseline (one hidden layer of 64 units)
     used purely as a heavier-weight comparison.

  3. ``FederatedLinearQAgent`` -- a federated extension where each tenant
     trains a local LinearQAgent and a coordinator periodically averages
     weights (FedAvg-style).

All agents share a common ``act / update`` interface.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


# ---------------------------------------------------------------- tile coding

class TileCoder:
    """Lightweight uniform-grid tile coder.

    Maps a continuous observation vector to a set of active binary features.
    With ``num_tilings`` overlapping offset grids and ``bins`` per dimension,
    the agent gets generalisation across nearby states for free.

    Implementation note: the encoding is vectorised across tilings so a single
    call performs ``O(num_tilings * dim)`` arithmetic operations without
    Python-level loops, keeping per-action inference in the low microseconds
    range even for high-dim observations.
    """

    def __init__(
        self,
        low: Sequence[float],
        high: Sequence[float],
        bins: int = 8,
        num_tilings: int = 4,
        seed: int = 0,
    ):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.span = np.maximum(self.high - self.low, 1e-6)
        self.bins = bins
        self.num_tilings = num_tilings
        self.dim = len(low)
        rng = np.random.default_rng(seed)
        # Random per-tiling offsets for asymmetric tilings.
        self.offsets = rng.uniform(0, 1, size=(num_tilings, self.dim)).astype(np.float32)
        self.offsets /= self.bins
        self.tiles_per_tiling = bins ** self.dim
        self.num_tiles = self.tiles_per_tiling * num_tilings
        # Pre-computed power vector for radix conversion of multi-dim indices
        # into a single flat tile index.
        self.power = np.array(
            [bins ** (self.dim - d - 1) for d in range(self.dim)],
            dtype=np.int64,
        )
        self.tiling_offsets = np.arange(num_tilings, dtype=np.int64) * self.tiles_per_tiling

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Return indices of active tiles (length == num_tilings)."""
        norm = (obs - self.low) / self.span
        norm = np.clip(norm, 0.0, 0.9999)
        # Shape (num_tilings, dim) after broadcasting offsets onto obs.
        shifted = np.clip(norm[None, :] + self.offsets, 0.0, 0.9999)
        cells = (shifted * self.bins).astype(np.int64)
        flat = (cells * self.power).sum(axis=1)
        return flat + self.tiling_offsets


# ---------------------------------------------------------------- Linear Q

@dataclass
class LinearQConfig:
    alpha: float = 0.25
    gamma: float = 0.9
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 3_000
    bins: int = 4
    num_tilings: int = 3
    seed: int = 0


class LinearQAgent:
    """Tile-coded linear Q-learning agent."""

    def __init__(
        self,
        obs_low: Sequence[float],
        obs_high: Sequence[float],
        n_actions: int,
        config: LinearQConfig | None = None,
    ):
        self.cfg = config or LinearQConfig()
        self.n_actions = n_actions
        self.tc = TileCoder(
            obs_low, obs_high,
            bins=self.cfg.bins,
            num_tilings=self.cfg.num_tilings,
            seed=self.cfg.seed,
        )
        # Small random init breaks the argmax tie-break bias toward action 0.
        np_rng = np.random.default_rng(self.cfg.seed)
        self.W = np_rng.normal(
            0.0, 0.01, size=(n_actions, self.tc.num_tiles)
        ).astype(np.float32)
        self.step_count = 0
        self.rng = random.Random(self.cfg.seed)
        self._np_rng = np_rng

    # ------------------------------------------------------------------

    def _epsilon(self) -> float:
        frac = min(1.0, self.step_count / self.cfg.epsilon_decay_steps)
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        active = self.tc(obs)
        return self.W[:, active].sum(axis=1)

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self._epsilon():
            return self.rng.randrange(self.n_actions)
        q = self.q_values(obs)
        # Random tie-breaking prevents the agent from getting stuck on action 0
        # when many Q-values are equal at uninitialised / unvisited tiles.
        max_q = q.max()
        best = np.where(q >= max_q - 1e-8)[0]
        return int(best[self._np_rng.integers(0, len(best))])

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> float:
        active = self.tc(obs)
        q_sa = self.W[action, active].sum()
        if done:
            target = reward
        else:
            next_q = self.q_values(next_obs)
            target = reward + self.cfg.gamma * float(np.max(next_q))
        td_error = target - q_sa
        # Linear-Q with tile coding: gradient is +1 on active tiles.
        self.W[action, active] += (self.cfg.alpha / self.cfg.num_tilings) * td_error
        self.step_count += 1
        return float(td_error)

    # ------------------------------------------------------------------

    def num_parameters(self) -> int:
        return int(self.W.size)

    def memory_bytes(self) -> int:
        return int(self.W.nbytes)


# ---------------------------------------------------------------- DQN baseline

class _NumpyMLP:
    """Tiny 1-hidden-layer MLP implemented in NumPy for fair zero-dep comparison."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        scale = math.sqrt(2.0 / n_in)
        self.W1 = rng.normal(0, scale, size=(n_in, n_hidden)).astype(np.float32)
        self.b1 = np.zeros(n_hidden, dtype=np.float32)
        self.W2 = rng.normal(0, math.sqrt(2.0 / n_hidden), size=(n_hidden, n_out)).astype(np.float32)
        self.b2 = np.zeros(n_out, dtype=np.float32)

    def forward(self, x: np.ndarray):
        z1 = x @ self.W1 + self.b1
        h1 = np.maximum(z1, 0.0)
        y = h1 @ self.W2 + self.b2
        return y, (x, z1, h1)

    def backward(self, cache, dY, lr: float):
        x, z1, h1 = cache
        dW2 = h1.T @ dY
        db2 = dY.sum(axis=0)
        dH1 = dY @ self.W2.T
        dZ1 = dH1 * (z1 > 0)
        dW1 = x.T @ dZ1
        db1 = dZ1.sum(axis=0)
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def num_parameters(self) -> int:
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size


@dataclass
class DQNConfig:
    hidden: int = 64
    lr: float = 1e-3
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5_000
    batch_size: int = 32
    buffer_size: int = 5_000
    target_sync: int = 200
    seed: int = 0


class DQNAgent:
    """Replay-buffer DQN baseline in pure NumPy."""

    def __init__(self, obs_dim: int, n_actions: int, config: DQNConfig | None = None):
        self.cfg = config or DQNConfig()
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.q = _NumpyMLP(obs_dim, self.cfg.hidden, n_actions, seed=self.cfg.seed)
        self.target = _NumpyMLP(obs_dim, self.cfg.hidden, n_actions, seed=self.cfg.seed)
        self._sync_target()
        self.buffer: List[tuple] = []
        self.step_count = 0
        self.rng = random.Random(self.cfg.seed)
        self.np_rng = np.random.default_rng(self.cfg.seed)

    def _sync_target(self):
        self.target.W1 = self.q.W1.copy()
        self.target.b1 = self.q.b1.copy()
        self.target.W2 = self.q.W2.copy()
        self.target.b2 = self.q.b2.copy()

    def _epsilon(self) -> float:
        frac = min(1.0, self.step_count / self.cfg.epsilon_decay_steps)
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self._epsilon():
            return self.rng.randrange(self.n_actions)
        q, _ = self.q.forward(obs.reshape(1, -1))
        return int(np.argmax(q[0]))

    def update(self, obs, action, reward, next_obs, done) -> float:
        self.buffer.append((obs.copy(), action, reward, next_obs.copy(), done))
        if len(self.buffer) > self.cfg.buffer_size:
            self.buffer.pop(0)

        self.step_count += 1
        if len(self.buffer) < self.cfg.batch_size:
            return 0.0
        if self.step_count % self.cfg.target_sync == 0:
            self._sync_target()

        batch_idx = self.np_rng.integers(0, len(self.buffer), size=self.cfg.batch_size)
        batch = [self.buffer[i] for i in batch_idx]
        obs_b = np.stack([b[0] for b in batch])
        act_b = np.array([b[1] for b in batch])
        rew_b = np.array([b[2] for b in batch], dtype=np.float32)
        nxt_b = np.stack([b[3] for b in batch])
        done_b = np.array([b[4] for b in batch], dtype=np.float32)

        q_pred, cache = self.q.forward(obs_b)
        q_next, _ = self.target.forward(nxt_b)
        target_vals = rew_b + (1 - done_b) * self.cfg.gamma * q_next.max(axis=1)
        dY = np.zeros_like(q_pred)
        for i, a in enumerate(act_b):
            dY[i, a] = (q_pred[i, a] - target_vals[i]) / self.cfg.batch_size
        self.q.backward(cache, dY, self.cfg.lr)
        return float(np.mean((q_pred[np.arange(len(act_b)), act_b] - target_vals) ** 2))

    def num_parameters(self) -> int:
        return self.q.num_parameters()

    def memory_bytes(self) -> int:
        return self.q.num_parameters() * 4  # float32


# ---------------------------------------------------------------- Federated

class FederatedLinearQAgent:
    """FedAvg across per-tenant LinearQAgents.

    Each tenant owns a local agent. Calling ``aggregate()`` averages the weight
    matrices, simulating a privacy-preserving server-side update where only
    weights (not raw queries) leave the tenant's region.
    """

    def __init__(
        self,
        n_tenants: int,
        obs_low: Sequence[float],
        obs_high: Sequence[float],
        n_actions: int,
        config: LinearQConfig | None = None,
    ):
        base = config or LinearQConfig()
        self.agents = []
        for i in range(n_tenants):
            # Distinct seed per client so exploration trajectories differ;
            # this is what makes weight aggregation actually useful.
            client_cfg = LinearQConfig(
                alpha=base.alpha,
                gamma=base.gamma,
                epsilon_start=base.epsilon_start,
                epsilon_end=base.epsilon_end,
                epsilon_decay_steps=base.epsilon_decay_steps,
                bins=base.bins,
                num_tilings=base.num_tilings,
                seed=base.seed + 1000 * (i + 1),
            )
            self.agents.append(LinearQAgent(obs_low, obs_high, n_actions, client_cfg))

    def aggregate(self):
        stacked = np.stack([a.W for a in self.agents], axis=0)
        mean_W = stacked.mean(axis=0)
        for a in self.agents:
            a.W = mean_W.copy()

    def __getitem__(self, idx: int) -> LinearQAgent:
        return self.agents[idx]

    def num_parameters(self) -> int:
        return self.agents[0].num_parameters()
