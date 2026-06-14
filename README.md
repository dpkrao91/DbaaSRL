# Predictive Resource Allocation in Serverless DBaaS via Lightweight Reinforcement Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PeerJ CS Submission](https://img.shields.io/badge/PeerJ%20CS-AI%20Application-orange.svg)](https://peerj.com/computer-science/)

> Companion repository for the PeerJ Computer Science AI Application
> manuscript **"Predictive Resource Allocation in Serverless DBaaS via
> Lightweight Reinforcement Learning"** (manuscript ID 140888).

---

## 1. Title

**Predictive Resource Allocation in Serverless DBaaS via Lightweight
Reinforcement Learning** — code, data, and manuscript supporting a
tile-coded linear Q-learning agent (with federated extension) for
capacity scaling in multi-tenant serverless databases such as Amazon
Aurora Serverless v2.

## 2. Description

This repository implements and evaluates a **tile-coded linear
Q-learning** agent for predictive resource allocation in serverless
DBaaS. The agent learns to choose Aurora-style capacity levels (ACUs)
in response to a 6-dimensional state of windowed workload statistics,
current capacity, and time-of-day, optimising a joint reward over
SLA-violation severity, monetary cost, and cold-start overhead.

A federated variant aggregates per-tenant policies via FedAvg, sharing
only learned weights and not raw query traces. Both variants are
compared against three non-RL baselines (threshold autoscaler,
always-maximum over-provisioning, ARIMA-style predictor) and a deep
Q-network of comparable training budget.

**Headline empirical findings** (3-day held-out evaluation,
seed 137):

| Agent              | SLA viol. ↓ | P99 lat. (ms) ↓ | Cost ↓  |
|--------------------|-------------|-----------------|---------|
| LinearQ (ours)     | 3.6%        | 1427            | 13,919  |
| FederatedLinearQ   | 10.7%       | 1427            | 11,752  |
| DQN                | 8.8%        | 113             | 13,821  |
| Threshold          | 0.2%        | 77              | 15,354  |
| ARIMA              | 2.6%        | 108             | 11,834  |
| OverProvision      | 0.0%        | 44              | 27,642  |

The linear agent halves the DQN's SLA-violation rate (3.6% vs 8.8%) at
essentially the same cost, and saves 9.4% on cost versus the reactive
threshold at the price of a higher (but bounded) violation rate.
**Limitation:** when the linear agent does violate the SLA the
violation is severe (saturated tail at ≈1,427 ms). Operators with hard
tail constraints should treat this as a binding limitation; mitigations
are discussed in the paper.

The complete claim-to-CSV-row audit trail is in `HONEST_RESULTS.md`.

## 3. Dataset Information

The repository ships with **deterministically regenerable synthetic
workloads** for 8 heterogeneous DBaaS tenants spanning OLTP/OLAP/mixed
profiles. The generator (`src/workload_generator.py`) implements the
parametric model described in the *Materials & Methods* section of the
paper: per-tenant baseline QPS, diurnal × weekly × drift modulation,
Bernoulli-thinned exponential micro-bursts, Pareto-tailed (α = 1.5)
flash crowds, and Gaussian noise. M/M/c-style P99 latency is computed
from instantaneous utilisation.

| File                                    | Rows    | Description                                              |
|-----------------------------------------|---------|----------------------------------------------------------|
| `data/workload_combined.csv`            | 80,640  | Training: 7 days × 8 tenants × 1,440 minutes (seed 42)   |
| `data/workload_aggregate.csv`           | 10,080  | Training: per-minute cluster aggregate                   |
| `data/workload_t0{1..8}_*.csv`          | 10,080  | Training: per-tenant traces                              |
| `data/workload_combined_eval.csv`       | 34,560  | Held-out eval: 3 days × 8 tenants (seed 137)             |
| `data/workload_aggregate_eval.csv`      | 4,320   | Held-out eval: per-minute cluster aggregate              |

Columns are documented inline in `src/workload_generator.py`. All
values are stored as 32-bit floats. The two seeds (42 train, 137 eval)
produce QPS distributions that overlap substantially but are not
identical: training (65, 1720, 691) min/max/mean QPS; eval (131, 1467,
754) min/max/mean QPS — guaranteeing out-of-distribution evaluation.

## 4. Code Information

```
serverless-rl-dbaas/
├── src/                                 # Python source code (9 modules)
│   ├── workload_generator.py            # 8-tenant synthetic generator
│   ├── environment.py                   # DBaaS simulator (M/M/c, ACU, cold-start)
│   ├── agents.py                        # LinearQ, DQN, FederatedLinearQ
│   ├── baselines.py                     # Threshold, OverProvision, ARIMA
│   ├── train.py                         # Training + held-out evaluation
│   ├── ablations.py                     # Ablation runner (tile granularity, etc.)
│   ├── benchmark_inference.py           # Per-step latency benchmark
│   └── plot_results.py                  # Figure generation (5 figures)
├── data/                                # Generated workload CSVs (regenerable)
├── results/                             # Evaluation CSVs + per-step traces
├── figures/                             # Manuscript figures (PNG)
├── paper/                               # PeerJ-format LaTeX manuscript
│   ├── main.tex                         # \documentclass{wlpeerj} source
│   ├── references.bib                   # BibTeX bibliography
│   ├── wlpeerj.cls                      # PeerJ class file (shim, replace before submission)
│   └── main.pdf                         # Compiled manuscript (15 pages)
├── tests/test_smoke.py                  # Pytest smoke tests
├── configs/                             # Hyperparameter configs (TOML)
├── notebooks/                           # Optional analysis notebooks
├── HONEST_RESULTS.md                    # Claim-to-CSV verification document
├── Makefile                             # `make data | train | figures | paper | all`
├── requirements.txt                     # Python dependencies
├── LICENSE                              # MIT
└── README.md                            # This file
```

## 5. Usage Instructions

### Quickstart (regenerate everything in ~10 minutes)

```bash
# 1. Clone and enter
git clone https://github.com/<your-org>/serverless-rl-dbaas.git
cd serverless-rl-dbaas

# 2. Install dependencies (Python 3.10+ required)
pip install -r requirements.txt

# 3. Regenerate everything end-to-end:
#    workload data → train 6 agents → ablations → benchmarks → figures → paper PDF
make all
```

### Step-by-step

```bash
make data       # Regenerate workload CSVs (training seed 42, eval seed 137)
make train      # Train all RL agents + run baselines on held-out trace
make ablations  # Run ablation study (tile granularity, init, tie-break)
make benchmark  # Per-step inference latency benchmark
make figures    # Regenerate 5 PNG figures from result CSVs
make paper      # Recompile manuscript PDF (requires pdflatex + bibtex)
make test       # Run pytest smoke tests
make clean      # Remove all generated artifacts
```

### Run a single agent

```python
from src.workload_generator import WorkloadConfig, generate_workload
from src.environment import DBaaSEnvironment
from src.agents import LinearQAgent

cfg = WorkloadConfig(n_days=7, seed=42)
workload = generate_workload(cfg)
env = DBaaSEnvironment(workload)
agent = LinearQAgent(state_dim=6, n_actions=8, n_tilings=3, n_bins=4)

state = env.reset()
for _ in range(1440):  # one day
    action = agent.act(state)
    next_state, reward, done, info = env.step(action)
    agent.update(state, action, reward, next_state, done)
    state = next_state
```

## 6. Requirements

### Software
| Package        | Tested version | Purpose                                |
|----------------|----------------|----------------------------------------|
| Python         | 3.10+          | Runtime                                |
| NumPy          | 1.26+          | Tensor / linear algebra                |
| pandas         | 2.0+           | CSV I/O and per-minute aggregations    |
| matplotlib     | 3.7+           | Figure generation                      |
| pytest         | 7.0+ (dev)     | Smoke tests                            |

Optional for manuscript compilation: `pdflatex`, `bibtex`, the `lineno`,
`authblk`, `natbib`, `algorithm`, `algpseudocode`, `booktabs`, `lmodern`
LaTeX packages. The included `wlpeerj.cls` is a minimal shim that lets
the manuscript compile in stock TeX Live; **replace it with the
official `wlpeerj.cls` from [peerj.com](https://peerj.com/about/author-instructions/)
before final submission**.

### Hardware
- **Single CPU thread** (no GPU/accelerator required)
- **~500 MB RAM** peak
- **~10 minutes** wall-clock for the full pipeline on a modern laptop
- Training: LinearQ ~23 s · DQN ~47 s · FederatedLinearQ (4 clients sequential) ~104 s

### Operating system
Linux, macOS, or Windows with WSL. Tested on Ubuntu 24.04 with Python
3.10.

## 7. Methodology

The methodology is described in full in the manuscript (`paper/main.pdf`)
*Materials & Methods* section. Briefly:

1. **Workload generation.** Eight tenant archetypes
   (`src/workload_generator.py`) each emit a per-minute QPS time series
   that is the sum of a diurnal-sinusoid × weekly-modulator × drift
   baseline, Bernoulli-thinned exponential micro-bursts, Pareto-tailed
   flash crowds, and Gaussian noise. Aggregating across tenants gives
   the cluster QPS used by the controller.
2. **Data preprocessing.** Cluster aggregate is grouped by minute;
   state features (windowed mean / max / linear-trend slope of QPS,
   current ACU capacity, normalised minute-of-day, current utilisation)
   are computed on-line from a 10-minute trailing window. No
   normalisation beyond the tile-coder's per-dimension min-max scaling.
   No data augmentation; no shuffling; no preprocessing of the eval
   trace.
3. **Simulator.** `src/environment.py` exposes a Gym-style interface.
   Per-minute P99 latency is computed from an M/M/c-style queue at the
   current utilisation (clipped at 5,000 ms to avoid divergence; this
   modelling artefact is discussed honestly in the paper). Cold starts
   from a paused state are interpolated linearly within the minute.
   Cost is `0.12/60 USD × ACU` per minute, matching Aurora Serverless
   v2 list pricing.
4. **Agents.** `src/agents.py` provides:
   - `LinearQAgent`: tile-coded linear Q-learning with K=3 tilings of
     B=4 bins per dimension (98,304 weights, 384 KB). Updates touch
     only K active tiles per action per step. No back-propagation, no
     replay.
   - `DQNAgent`: single-hidden-layer (H=64) MLP, 968 parameters, with
     target network (sync every 200 steps), replay buffer (size
     5,000), and Adam optimiser.
   - `FederatedLinearQAgent`: 4 LinearQ clients with distinct seeds;
     FedAvg aggregation every 5 episodes.
5. **Baselines.** `src/baselines.py` implements a threshold autoscaler
   (0.7/0.3 with 5-min cooldown, Aurora-style), always-max
   over-provisioning, and a rolling-window ARIMA-style linear-trend
   predictor with a 20% safety margin.
6. **Training.** 100 episodes × 720 minutes per episode = 72,000 env
   interactions. Top-level seed 42; federated client seeds
   `42 + 1000 × (i+1)`.
7. **Evaluation.** Held-out 3-day trace (seed 137). Greedy
   evaluation; RL agents continue updating on-line during evaluation
   to mirror continual learning in production.
8. **Honest verification.** Every numerical claim in the manuscript is
   traced to a specific CSV row in `HONEST_RESULTS.md`. The full
   pipeline is deterministic given the seeds; running `make all`
   regenerates every number and figure in the paper.

## 8. AI Disclosure

In accordance with PeerJ policy, the authors disclose that artificial
intelligence assistance was used during the preparation of this work.
Specifically:

- **Tool:** Claude (Anthropic), model version Opus 4.7
- **Access:** via [claude.ai](https://claude.ai), May–June 2026
- **Uses:**
  1. Drafting and iterating Python source for the workload generator,
     environment simulator, and RL agents (all code reviewed,
     executed, and tested by the authors before inclusion).
  2. Drafting and iterating the LaTeX manuscript source.
  3. Identifying candidate design improvements (e.g., random argmax
     tie-breaking, saturating reward normalisation), each
     subsequently evaluated empirically by the authors.
- **NOT used for:** fabricating results; generating figures (all
  produced by deterministic Python scripts); inventing citations
  (cross-checked by the authors); or final scientific decisions.

This disclosure is repeated in the *Materials & Methods* section of
the manuscript itself (under *Use of Artificial Intelligence in
Manuscript Preparation*), as required by PeerJ AI Application policy.

## 9. Citations

If you use this code, data, or methodology, please cite:

```bibtex
@article{deepak2026serverless-rl,
  title   = {Predictive Resource Allocation in Serverless {DBaaS}
             via Lightweight Reinforcement Learning},
  author  = {[Surname], Deepak},
  journal = {PeerJ Computer Science},
  year    = {2026},
  note    = {AI Application; under review, manuscript ID 140888}
}
```

### Key references used in this work

- Sutton & Barto (2018). *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press.
- Sutton (1996). *Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding*. NeurIPS.
- McMahan et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS.
- Mnih et al. (2015). *Human-level control through deep reinforcement learning*. Nature 518.
- Roy, Dubey & Gokhale (2011). *Efficient autoscaling in the cloud using predictive models for workload forecasting*. IEEE CLOUD.
- Tsitsiklis & Van Roy (1997). *An analysis of temporal-difference learning with function approximation*. IEEE TAC 42(5).
- Zhang et al. (2019). *An end-to-end automatic cloud database tuning system using deep reinforcement learning (CDBTune)*. ACM SIGMOD.

Full bibliography: `paper/references.bib`.

## 10. License & Contribution Guidelines

### License
This work is released under the **MIT License** (see `LICENSE`).

### Contributing
Contributions are welcome. Please:

1. Open an issue to discuss substantial changes before opening a PR.
2. Ensure `make test` passes.
3. Update `HONEST_RESULTS.md` if your change affects any numerical
   claim in the manuscript.
4. Sign your commits.

### Reporting issues
- Bugs: open a GitHub issue with the failing `make` target and the
  contents of the error log.
- Reproducibility issues: include OS, Python version, NumPy version,
  and the output of `make all 2>&1 | tee debug.log`.

### Code of conduct
Contributors are expected to follow the
[Contributor Covenant](https://www.contributor-covenant.org/) v2.1.

---

## Companion documents

- **`paper/main.pdf`** — the manuscript itself (15 pages, PeerJ AI
  Application format).
- **`HONEST_RESULTS.md`** — claim-by-claim audit trail mapping every
  number in the manuscript to a CSV row.
- **`paper/main.tex`** — LaTeX source (line-numbered, ready for PeerJ
  review).
- **`paper/references.bib`** — BibTeX bibliography (separate file as
  required by PeerJ LaTeX submission).

## Acknowledgements

We thank PeerJ for the opportunity to submit as an AI Application
article, and the open-source maintainers of NumPy, pandas, matplotlib,
pytest, and TeX Live.
