# MLOps Capstone Project — Full Roadmap
### "ChurnOps" — Customer Churn Prediction with a Production-Grade MLOps Stack

---

## 1. Project Concept

**Why churn prediction?**
- Small, well-known tabular dataset (e.g. [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), ~7,000 rows, ~20 features)
- XGBoost/LightGBM trains in **1–5 seconds on CPU** — you can run hundreds of experiments without ever touching a GPU
- It's one of the most common real-world enterprise ML use cases (churn, fraud, credit risk, propensity models are all structurally identical), so every piece of MLOps plumbing you build directly maps to industry
- Binary classification with clean, interpretable features → drift monitoring, data validation, and explainability all become intuitive to reason about

**Alternative if you want an NLP flavor instead:** SMS Spam / AG News text classification with TF-IDF + Logistic Regression — trains just as fast, same roadmap applies unchanged. Everything below is model-agnostic; swap the training script and keep the rest.

**Guiding principle:** the model is a *fixture*. Never spend more than ~5% of your total project time improving accuracy. All complexity budget goes into the surrounding system.

---

## 2. Tool Stack (mapped to what Berlin ML/MLOps job postings ask for)

| Lifecycle stage | Tool | Why this one |
|---|---|---|
| Dependency & env management | **uv** (or Poetry) | Modern standard, fast, reproducible lockfiles |
| Version control | **Git + GitHub** | Baseline for everything else |
| Code quality | **ruff, black, pre-commit, mypy** | Every real eng team gates on this |
| Data versioning | **DVC** | Git-style diffing for data/models, works with any storage backend (S3/local) |
| Data validation | **Great Expectations** or **Pandera** | Companies gate pipelines on data contracts, not just code tests |
| Experiment tracking | **MLflow** | The single most-requested tracking tool in job postings; self-hostable |
| Hyperparameter tuning | **Optuna** | Bayesian/TPE search, integrates directly with MLflow logging |
| Model registry | **MLflow Model Registry** | Staging → Production promotion workflow |
| Pipeline orchestration | **Apache Airflow** (Prefect as lighter alt.) | Most commonly named orchestrator in DE job ads; DAG-based retraining pipeline |
| Containerization | **Docker + Docker Compose** | Everything (training, serving, MLflow, monitoring) runs as services |
| Model serving | **FastAPI** (+ Uvicorn) | Ubiquitous, lightweight, easy to instrument |
| CI/CD | **GitHub Actions** | Lint/test/build/push/deploy pipeline |
| Container registry | **GitHub Container Registry (GHCR)** | Free, integrates natively with Actions |
| Infrastructure as Code | **Terraform** | Provision cloud resources declaratively |
| Cloud | **AWS (free tier)** | S3 for artifact/DVC storage, ECR, one small EC2 instance |
| Container orchestration | **Kubernetes via `kind`** (local) | Learn K8s without cloud cluster costs; Helm for packaging |
| Monitoring (infra) | **Prometheus + Grafana** | Standard observability stack, mentioned constantly in postings |
| Monitoring (ML-specific) | **Evidently AI** | Data drift / prediction drift reports, pure Python |
| Testing | **pytest** | Unit + data + model contract tests |
| Documentation | **MkDocs** | Auto-published architecture docs |
| (Stretch) Feature store | **Feast** | Only if you want to show you understand feature reuse across models |
| (Stretch) GitOps | **ArgoCD** | Sync K8s manifests from Git automatically |

---

## 3. Roadmap — Ordered Phases & Tasks

Each phase builds on the last. Treat each as a PR (or a few) into `main`, gated by your growing CI pipeline.

### Phase 0 — Project Scaffolding
- [ ] Create repo, choose license, write initial README with problem statement
- [ ] Set up `uv`/Poetry project, pin Python version
- [ ] Install and configure `pre-commit` (ruff, ruff-format, mypy, end-of-file-fixer)
- [ ] Define repo structure (see Section 4)
- [ ] Write a one-page "architecture decision record" (ADR) template — you'll use this at every major tool choice, exactly like a real team does

### Phase 1 — Reproducible Data Foundation
- [ ] Download raw dataset, store in `data/raw/` (not committed to Git)
- [ ] Initialize **DVC**, configure a remote (local dir or free S3 bucket)
- [ ] Write ingestion script; version the raw data with `dvc add`
- [ ] Add data validation with **Great Expectations/Pandera**: schema, null checks, value ranges, category checks
- [ ] Write a short EDA notebook (keep it minimal — this isn't the focus)
- [ ] Set up config management (Hydra or pydantic-settings) — no hardcoded paths/params anywhere

### Phase 2 — Training Pipeline + Experiment Tracking
- [ ] Write baseline training script (XGBoost, deterministic seed)
- [ ] Stand up **MLflow** tracking server (Docker service, local file or S3 backend store)
- [ ] Log params, metrics (F1, ROC-AUC, precision/recall), and artifacts (model, confusion matrix plot) per run
- [ ] Add `pytest` unit tests for: data transforms, feature engineering, training function I/O shapes
- [ ] Add a `Makefile` or CLI (Typer/Click) so every step is a single reproducible command

### Phase 3 — Hyperparameter Optimization
- [ ] Define search space and objective function for **Optuna**
- [ ] Use TPE sampler + pruning (e.g. `MedianPruner`) to cut wasted trials
- [ ] Log every trial as a nested MLflow run
- [ ] Compare runs in the MLflow UI; pick winning config by a defined metric threshold, not by eyeballing

### Phase 4 — Model Registry & Packaging
- [ ] Register best model in **MLflow Model Registry**
- [ ] Define stage transitions: `None → Staging → Production → Archived`
- [ ] Write a promotion script that only promotes if new model beats current Production model on a held-out eval set (a real "champion/challenger" gate)
- [ ] Containerize the training job itself (`Dockerfile.train`) so training is fully reproducible outside your machine

### Phase 5 — Orchestration
- [ ] Stand up **Airflow** via Docker Compose (webserver, scheduler, Postgres metadata DB)
- [ ] Build a DAG: `validate_data → train → evaluate → conditionally_register`
- [ ] Parameterize the DAG (e.g. trigger with different config overrides)
- [ ] Manually trigger a full pipeline run end-to-end and confirm idempotency (running twice doesn't corrupt state)

### Phase 6 — Model Serving
- [ ] Build a **FastAPI** service that loads the current "Production" model from the MLflow registry at startup
- [ ] Define request/response schemas with Pydantic (input validation is a real production requirement, not optional)
- [ ] Add `/health` and `/predict` endpoints; add `/reload` to hot-swap models without restart
- [ ] Containerize the serving app (`Dockerfile.serve`)
- [ ] Load-test locally with `locust` or `hey` to know your baseline latency/throughput

### Phase 7 — CI/CD Pipeline
- [ ] **PR workflow**: lint, type-check, unit tests, data-contract tests — block merge on failure
- [ ] **Main-branch workflow**: build Docker images for train + serve, push to GHCR with git-sha tags
- [ ] **Model CI**: on relevant changes, trigger a training run in CI and fail the build if eval metrics regress past a threshold
- [ ] Add branch protection rules; require passing checks before merge
- [ ] Introduce semantic versioning for both code releases and model versions (they should be traceable to each other)

### Phase 8 — Infrastructure as Code
- [ ] Write **Terraform** to provision: S3 bucket (DVC + MLflow artifact store), ECR/GHCR auth, one small EC2 instance (or skip EC2 and go straight to Phase 9's local K8s if you want zero cloud cost)
- [ ] Store Terraform state remotely (S3 + DynamoDB lock table) — this is how real teams avoid state corruption
- [ ] Document the infra in the ADR log

### Phase 9 — Kubernetes Deployment
- [ ] Spin up a local cluster with `kind` or `k3d`
- [ ] Write K8s manifests: `Deployment`, `Service`, `ConfigMap`, `HorizontalPodAutoscaler` for the serving app
- [ ] Package as a **Helm chart** (parameterize replica count, image tag, resource limits)
- [ ] Deploy MLflow + serving app to the cluster; verify via `kubectl port-forward`
- [ ] *(Stretch)* Add **ArgoCD** for GitOps: cluster state syncs automatically from a Git repo

### Phase 10 — Monitoring & Observability
- [ ] Instrument the FastAPI service with `prometheus-fastapi-instrumentator` (latency, request count, error rate)
- [ ] Deploy **Prometheus** to scrape metrics, **Grafana** for dashboards
- [ ] Add **Evidently AI** batch job: compare live prediction inputs against training data distribution, generate drift reports on a schedule
- [ ] Set up an alert (Grafana alert rule or a simple webhook to Slack/Discord) for drift or error-rate spikes

### Phase 11 — Automated Retraining Loop
- [ ] Define a retraining trigger: scheduled (weekly) OR drift-threshold-based (from Evidently output)
- [ ] Wire the trigger to the Airflow DAG from Phase 5
- [ ] Ensure the promotion gate from Phase 4 still applies — no model reaches Production without beating the incumbent
- [ ] *(Stretch)* Shadow-deploy new models: serve predictions from both champion and challenger, log both, compare before promoting

### Phase 12 — Polish, Docs, Security
- [ ] Draw a full architecture diagram (draw.io/Excalidraw) showing every component and data flow
- [ ] Publish docs with **MkDocs** (setup, architecture, runbook for "model is drifting, what do I do")
- [ ] Security pass: non-root Docker users, `.env`/GitHub Secrets for credentials, `trivy` or `pip-audit` dependency scanning in CI
- [ ] Write a project retrospective: what you'd do differently at 10x scale (this is a great interview talking point)

---

## 4. Suggested Repo Structure

```
churnops/
├── .github/workflows/          # CI/CD pipelines
├── airflow/dags/
├── data/                       # DVC-tracked, not committed to git
├── docker/                     # Dockerfile.train, Dockerfile.serve, docker-compose.yml
├── k8s/                        # manifests + helm chart
├── terraform/
├── src/
│   └── churnops/                # the importable package (src-layout, uv-managed)
│       ├── data/                # ingestion, validation
│       ├── features/
│       ├── training/            # train.py, tune.py (Optuna)
│       ├── serving/              # FastAPI app
│       └── monitoring/           # Evidently jobs
├── tests/
├── docs/                        # MkDocs source
├── notebooks/                    # EDA only, minimal
├── pyproject.toml
└── README.md
```

---

## 5. Time Investment (rough guide)

| Phases | Focus | Est. time |
|---|---|---|
| 0–2 | Foundation, tracking | 1 week |
| 3–4 | Tuning, registry | 3–4 days |
| 5–6 | Orchestration, serving | 1–1.5 weeks |
| 7–8 | CI/CD, IaC | 1 week |
| 9–10 | K8s, monitoring | 1–1.5 weeks |
| 11–12 | Retraining loop, polish | 1 week |

Total: roughly **5–7 weeks** part-time — very doable alongside thesis writing, and each phase is independently demoable in an interview.

---

## 6. Stretch Goals (if you have extra time)

- **Feast** feature store — shows you understand feature reuse/point-in-time correctness
- **A/B testing framework** for champion/challenger in production
- **Cost tracking** dashboard (even mocked) — a genuinely underrated skill companies value
- **KServe/Seldon Core** instead of raw FastAPI-on-K8s, for K8s-native model serving semantics
- **LLM-as-a-service add-on**: swap in a small classifier LLM to show you can extend the same MLOps skeleton to GenAI workloads
