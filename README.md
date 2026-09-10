# ChurnOps

**A customer churn classifier built to demonstrate a full, production-style MLOps lifecycle — not to win a Kaggle leaderboard.**

> 🎓 Master's capstone project. The ML model is intentionally simple (XGBoost on tabular data, trains in seconds on CPU) so that effort goes into the *system* around it: experiment tracking, orchestration, CI/CD, infrastructure-as-code, deployment, and monitoring — the actual day-to-day of an ML Engineer.

---

## Problem Statement

Predict whether a telecom customer will churn, using the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (~7,000 rows, ~20 features: tenure, contract type, monthly charges, services subscribed, etc.).

Churn prediction was chosen deliberately: it's one of the most common real-world enterprise ML use cases (structurally identical to fraud detection, credit risk, and propensity modeling), trains fast enough to iterate constantly, and produces intuitive, interpretable features — which makes concepts like data drift and model monitoring easy to reason about later in the project.

## What This Project Demonstrates

| Capability | Status |
|---|---|
| Reproducible data pipeline (DVC + schema validation) | 🔲 Planned |
| Experiment tracking & model registry (MLflow) | 🔲 Planned |
| Hyperparameter optimization (Optuna) | 🔲 Planned |
| Pipeline orchestration (Airflow) | 🔲 Planned |
| Containerized training & serving (Docker) | 🔲 Planned |
| Model serving API (FastAPI) | 🔲 Planned |
| CI/CD (GitHub Actions) | 🔲 Planned |
| Infrastructure as Code (Terraform, AWS) | 🔲 Planned |
| Kubernetes deployment (kind + Helm) | 🔲 Planned |
| Monitoring & drift detection (Prometheus, Grafana, Evidently) | 🔲 Planned |
| Automated retraining loop | 🔲 Planned |

See [`docs/roadmap.md`](docs/roadmap.md) for the full phase-by-phase build plan, and [`docs/tool-stack.md`](docs/tool-stack.md) for why each tool was chosen over its alternatives.

## Architecture (target state)

```
 Git push
    │
    ▼
GitHub Actions (lint, test, build) ──► Container Registry
    │
    ▼
Airflow DAG: validate → train → tune → conditionally register
    │                                          │
    ▼                                          ▼
DVC (data versions)                   MLflow (tracking + registry)
                                               │
                                               ▼
                                  FastAPI serving app (Kubernetes)
                                               │
                              ┌────────────────┼────────────────┐
                              ▼                                 ▼
                    Prometheus + Grafana                Evidently (drift reports)
                    (service health)                    (model validity)
                                               │
                                               ▼
                          Drift/schedule trigger → back to Airflow DAG
```

## Tech Stack

| Layer | Tools |
|---|---|
| Language / env | Python 3.11, `uv` |
| Data versioning | DVC |
| Data validation | Pandera |
| Experiment tracking | MLflow |
| Hyperparameter tuning | Optuna |
| Orchestration | Apache Airflow |
| Containerization | Docker, Docker Compose |
| Serving | FastAPI |
| CI/CD | GitHub Actions |
| IaC / Cloud | Terraform, AWS |
| Container orchestration | Kubernetes (`kind`), Helm |
| Monitoring | Prometheus, Grafana, Evidently AI |
| Testing | pytest |
| Docs | MkDocs |

Full rationale for each choice lives in [`docs/tool-stack.md`](docs/tool-stack.md).

## Project Structure

```
churnops/
├── .github/workflows/     # CI/CD pipelines
├── airflow/dags/          # Pipeline DAGs
├── data/                  # DVC-tracked, not committed to git
├── docker/                # Dockerfile.train, Dockerfile.serve, docker-compose.yml
├── k8s/                   # Manifests + Helm chart
├── terraform/             # Infrastructure as Code
├── src/
│   ├── data/              # Ingestion, validation
│   ├── features/          # Feature engineering
│   ├── training/          # train.py, tune.py
│   ├── serving/            # FastAPI app
│   └── monitoring/          # Evidently jobs
├── tests/
├── docs/                    # MkDocs source + roadmap/tool-stack docs
├── notebooks/                # EDA only, minimal
├── pyproject.toml
└── README.md
```

## Getting Started

> ⚠️ Project is in early scaffolding (Phase 0). Setup steps below will fill in as each phase lands.

**Prerequisites:** Python 3.11+, Docker, [`uv`](https://docs.astral.sh/uv/)

```bash
# Clone and install dependencies
git clone https://github.com/<your-username>/churnops.git
cd churnops
uv sync

# Set up pre-commit hooks
uv run pre-commit install

# (Coming soon) Spin up the full local stack
docker compose up
```

## Development Workflow

```bash
uv run pytest              # run tests
uv run ruff check .        # lint
uv run ruff format .       # format
```

All PRs are gated by lint, type-check, and test checks via GitHub Actions (see `.github/workflows/`).

## Roadmap

This project is being built in ordered phases — see [`docs/roadmap.md`](docs/roadmap.md) for the full breakdown (data foundation → tracking → tuning → orchestration → serving → CI/CD → IaC → Kubernetes → monitoring → automated retraining).

## Dataset & License

- Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), originally published by IBM as a sample dataset.
- Code in this repository is licensed under the [MIT License](LICENSE).
