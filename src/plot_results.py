"""Generate publication-quality figures from saved experiment results."""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)

AGENT_COLORS = {
    "LinearQ":          "#1f77b4",
    "DQN":              "#d62728",
    "Threshold":        "#7f7f7f",
    "OverProvision":    "#9467bd",
    "ARIMA":            "#2ca02c",
    "FederatedLinearQ": "#ff7f0e",
}


def plot_workload_overview(workload_csv: str, out_path: str) -> None:
    df = pd.read_csv(workload_csv)
    fig, ax = plt.subplots(figsize=(7, 3.0))
    hours = df["minute"] / 60.0
    ax.plot(hours, df["total_qps"], lw=0.7, color="#1f77b4")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Aggregate QPS (all tenants)")
    ax.set_title("Synthetic Multi-Tenant Workload Trace")
    ax.grid(alpha=0.3)
    fig.savefig(out_path)
    plt.close(fig)


def plot_training_curves(curves_csv: str, out_path: str) -> None:
    df = pd.read_csv(curves_csv)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for agent, group in df.groupby("agent"):
        # Use a larger smoothing window since DQN's per-episode return is
        # very noisy; show raw points as a faint scatter underneath.
        smooth = group["reward"].rolling(7, min_periods=1, center=True).median()
        color = AGENT_COLORS.get(agent, None)
        ax.scatter(group["episode"], group["reward"],
                   s=8, color=color, alpha=0.25, zorder=1)
        ax.plot(group["episode"], smooth,
                label=agent, color=color, lw=1.8, zorder=3)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episodic Return")
    ax.set_title("Learning Curves (raw points + 7-episode median)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.savefig(out_path)
    plt.close(fig)


def plot_evaluation_bars(summary_csv: str, out_path: str) -> None:
    df = pd.read_csv(summary_csv).sort_values("agent").reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))

    metrics = [
        ("sla_violation_rate", "SLA Violation Rate", "Fraction"),
        ("total_cost",         "Cumulative Cost",    "Cost (scaled ACU-min)"),
        ("p99_latency_ms",     "P99 Latency",        "Milliseconds"),
    ]
    for ax, (col, title, ylabel) in zip(axes, metrics):
        colors = [AGENT_COLORS.get(a, "#444") for a in df["agent"]]
        ax.bar(df["agent"], df[col], color=colors, edgecolor="black", lw=0.5)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_step_traces(results_dir: str, out_path: str) -> None:
    agents = ["linearq", "dqn", "threshold", "arima"]
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.5), sharex=True)
    for agent in agents:
        f = os.path.join(results_dir, f"per_step_{agent}.csv")
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        if len(df) > 720:
            df = df.iloc[:720]
        label = {"linearq": "LinearQ", "dqn": "DQN",
                 "threshold": "Threshold", "arima": "ARIMA"}[agent]
        color = AGENT_COLORS[label]
        axes[0].plot(df.index, df["acu"],     lw=1.0, label=label, color=color)
        axes[1].plot(df.index, df["latency_ms"], lw=1.0, label=label, color=color)
    axes[0].set_ylabel("ACU allocated")
    axes[0].set_title("Capacity Allocation Over Time (first 12h of eval)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right", ncol=4)

    axes[1].set_ylabel("P99 latency (ms)")
    axes[1].set_xlabel("Minute index (eval set)")
    axes[1].axhline(100, ls="--", color="red", lw=0.8, label="SLA target")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right", ncol=5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_pareto(summary_csv: str, out_path: str) -> None:
    df = pd.read_csv(summary_csv)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for _, row in df.iterrows():
        color = AGENT_COLORS.get(row["agent"], "#444")
        ax.scatter(row["total_cost"], row["sla_violation_rate"],
                   s=80, color=color, edgecolor="black", zorder=3)
        ax.annotate(row["agent"],
                    (row["total_cost"], row["sla_violation_rate"]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=8.5)
    ax.set_xlabel("Cumulative Cost (scaled)")
    ax.set_ylabel("SLA Violation Rate")
    ax.set_title("Cost-vs-SLA Pareto Frontier")
    ax.grid(alpha=0.3)
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--workload", default="data/workload_aggregate.csv")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    plot_workload_overview(args.workload, os.path.join(args.out, "workload_overview.png"))
    plot_training_curves(os.path.join(args.results, "training_curves.csv"),
                         os.path.join(args.out, "training_curves.png"))
    plot_evaluation_bars(os.path.join(args.results, "evaluation_summary.csv"),
                         os.path.join(args.out, "evaluation_bars.png"))
    plot_step_traces(args.results, os.path.join(args.out, "step_traces.png"))
    plot_pareto(os.path.join(args.results, "evaluation_summary.csv"),
                os.path.join(args.out, "pareto.png"))
    print(f"Figures written to {args.out}/")


if __name__ == "__main__":
    main()
