"""
Synthetic Multi-Tenant Workload Generator for Serverless DBaaS.

Generates realistic time-series workload traces that simulate the kind of
bursty, multi-tenant database traffic observed in cloud DBaaS systems such
as Amazon Aurora Serverless v2. Traces combine the following components:

  * Diurnal sinusoidal base (business-hour cycle)
  * Weekly seasonality (weekend dips)
  * Per-tenant baseline (tenant heterogeneity)
  * Poisson-distributed micro-burst arrivals
  * Heavy-tailed (Pareto) flash-crowd spikes
  * Slow drift / non-stationarity
  * Gaussian measurement noise

The output is a CSV per tenant plus a combined aggregate file used by the
RL training pipeline.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TenantProfile:
    """Per-tenant workload profile parameters."""

    tenant_id: str
    baseline_qps: float        # average queries per second
    diurnal_amplitude: float   # business-hour multiplier
    burst_prob: float          # probability of a micro-burst per minute
    burst_scale: float         # mean magnitude of a micro-burst
    flash_prob: float          # probability of a flash crowd per day
    workload_type: str         # "oltp", "olap", or "mixed"
    sla_ms: float              # target P99 latency in milliseconds


# A heterogeneous tenant population. The mix loosely follows public AWS
# Aurora Serverless v2 customer surveys, weighted toward OLTP traffic.
DEFAULT_TENANTS = [
    TenantProfile("t01_ecommerce",      120.0, 1.8, 0.04, 60.0,  0.05, "oltp",  50.0),
    TenantProfile("t02_saas_crm",        80.0, 1.6, 0.03, 40.0,  0.03, "oltp",  80.0),
    TenantProfile("t03_analytics",       30.0, 1.2, 0.01, 25.0,  0.02, "olap", 500.0),
    TenantProfile("t04_gaming",         200.0, 2.2, 0.08, 90.0,  0.08, "oltp",  30.0),
    TenantProfile("t05_iot_telemetry",  150.0, 1.1, 0.02, 30.0,  0.01, "oltp", 100.0),
    TenantProfile("t06_media",           60.0, 1.5, 0.05, 50.0,  0.10, "mixed", 150.0),
    TenantProfile("t07_fintech",        100.0, 1.7, 0.02, 35.0,  0.01, "oltp",  40.0),
    TenantProfile("t08_ml_pipeline",     20.0, 1.0, 0.01, 80.0,  0.04, "olap", 800.0),
]


def _diurnal(t_minutes: np.ndarray, amplitude: float) -> np.ndarray:
    """Daily sinusoid peaking around 14:00 local time."""
    hours = (t_minutes / 60.0) % 24.0
    return 1.0 + amplitude * 0.5 * (np.sin(2 * np.pi * (hours - 8) / 24.0) + 1) - amplitude * 0.5


def _weekly(t_minutes: np.ndarray) -> np.ndarray:
    """Weekend traffic dips to roughly 60% of weekday volume."""
    day_of_week = (t_minutes / (60 * 24)) % 7
    weekend = (day_of_week >= 5).astype(float)
    return 1.0 - 0.4 * weekend


def _bursts(n: int, prob: float, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Poisson micro-bursts on a per-minute granularity."""
    arrivals = rng.binomial(1, prob, size=n)
    magnitudes = rng.exponential(scale, size=n) * arrivals
    # Burst decay over 3-5 minutes via simple convolution
    kernel = np.exp(-np.arange(5) / 2.0)
    kernel /= kernel.sum()
    return np.convolve(magnitudes, kernel, mode="same")


def _flash_crowds(n: int, daily_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Heavy-tailed flash crowd events, sparse but large."""
    n_days = max(1, n // (60 * 24))
    series = np.zeros(n)
    n_events = rng.poisson(daily_prob * n_days)
    for _ in range(n_events):
        start = rng.integers(0, n)
        # Pareto-distributed magnitude (alpha=1.5 -> heavy tail)
        magnitude = (rng.pareto(1.5) + 1) * 200
        duration = int(rng.uniform(15, 90))  # 15-90 minutes
        decay = np.exp(-np.arange(duration) / (duration / 3))
        end = min(start + duration, n)
        series[start:end] += magnitude * decay[: end - start]
    return series


def generate_tenant_trace(
    profile: TenantProfile,
    n_minutes: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a single-tenant time-series workload trace."""
    rng = np.random.default_rng(seed + hash(profile.tenant_id) % 10_000)
    t = np.arange(n_minutes)

    diurnal = _diurnal(t, profile.diurnal_amplitude)
    weekly = _weekly(t)
    bursts = _bursts(n_minutes, profile.burst_prob, profile.burst_scale, rng)
    flash = _flash_crowds(n_minutes, profile.flash_prob, rng)

    # Slow non-stationary drift -- captures gradual tenant growth
    drift = 1.0 + 0.15 * np.sin(2 * np.pi * t / (n_minutes / 3))

    qps = profile.baseline_qps * diurnal * weekly * drift + bursts + flash
    qps += rng.normal(0, profile.baseline_qps * 0.05, size=n_minutes)
    qps = np.clip(qps, 0, None)

    # Derive a synthetic CPU-load and latency signal from QPS using an
    # M/M/c-style approximation: latency rises sharply as utilisation
    # approaches capacity. We pretend each ACU handles ~50 QPS.
    capacity_qps = 50.0 * 4.0  # baseline of 4 ACUs reserved
    utilisation = np.clip(qps / capacity_qps, 0, 0.999)
    latency_ms = 5.0 + 45.0 * (utilisation / (1 - utilisation)) ** 0.5
    latency_ms = np.clip(latency_ms, 5.0, 5000.0)

    cpu_pct = np.clip(utilisation * 100 + rng.normal(0, 3, n_minutes), 0, 100)
    mem_pct = np.clip(40 + 0.4 * cpu_pct + rng.normal(0, 5, n_minutes), 0, 100)

    return pd.DataFrame(
        {
            "minute": t,
            "tenant_id": profile.tenant_id,
            "workload_type": profile.workload_type,
            "sla_ms": profile.sla_ms,
            "qps": qps.round(2),
            "cpu_pct": cpu_pct.round(2),
            "mem_pct": mem_pct.round(2),
            "latency_p99_ms": latency_ms.round(2),
        }
    )


def generate_full_dataset(
    n_days: int = 14,
    out_dir: str = "data",
    seed: int = 42,
) -> str:
    """Generate per-tenant CSVs and a combined dataset.

    Returns the path to the combined CSV.
    """
    os.makedirs(out_dir, exist_ok=True)
    n_minutes = n_days * 24 * 60

    frames = []
    for profile in DEFAULT_TENANTS:
        df = generate_tenant_trace(profile, n_minutes, seed=seed)
        path = os.path.join(out_dir, f"workload_{profile.tenant_id}.csv")
        df.to_csv(path, index=False)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined_path = os.path.join(out_dir, "workload_combined.csv")
    combined.to_csv(combined_path, index=False)

    # Aggregate per-minute view used for cluster-wide RL state.
    agg = (
        combined.groupby("minute")
        .agg(
            total_qps=("qps", "sum"),
            mean_cpu=("cpu_pct", "mean"),
            max_cpu=("cpu_pct", "max"),
            mean_latency=("latency_p99_ms", "mean"),
            max_latency=("latency_p99_ms", "max"),
            n_tenants=("tenant_id", "nunique"),
        )
        .reset_index()
    )
    agg_path = os.path.join(out_dir, "workload_aggregate.csv")
    agg.to_csv(agg_path, index=False)

    print(f"Wrote {len(DEFAULT_TENANTS)} tenant files + combined ({len(combined):,} rows) "
          f"and aggregate ({len(agg):,} rows) to {out_dir}/")
    return combined_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic DBaaS workload traces.")
    parser.add_argument("--days", type=int, default=14, help="Number of simulated days.")
    parser.add_argument("--out", type=str, default="data", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    generate_full_dataset(args.days, args.out, args.seed)
