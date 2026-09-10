# ChurnOps Tool Stack — Explained

This document goes through every tool from the roadmap: what it does in general, what job it does specifically in this project, why it was chosen over the alternatives, and a minimal code snippet so you can see it in action. Read it phase-by-phase alongside the roadmap, not all at once.

---

## 1. Environment & Code Quality

### uv (dependency & environment management)
**What it is in general:** A package/project manager for Python — installs dependencies, manages virtual environments, and produces a lockfile for exact reproducibility.

**Role in this project:** Every service (training, serving, Airflow tasks) needs the *exact same* dependency versions to avoid "works on my machine" bugs. `uv.lock` is the single source of truth, copied into every Docker image.

**Why this one:** The classic choice is **pip + virtualenv**, but it resolves dependencies slowly and has no lockfile by default. **Poetry** added lockfiles and a nicer CLI but is noticeably slower than `uv` (written in Rust) and has had packaging-standard friction. `uv` is now what most new Python infra projects reach for because it's a drop-in replacement that's 10-100x faster and still speaks the standard `pyproject.toml` format.

```bash
uv init churnops
uv add xgboost mlflow fastapi
uv sync --frozen   # installs exactly what's in uv.lock, used in Docker/CI
```

### ruff, black, mypy, pre-commit (code quality)
**What they are:** `ruff` is a fast linter (catches bugs, unused imports, style issues), `black` is an opinionated auto-formatter, `mypy` checks type hints statically, and `pre-commit` runs all of them automatically before every commit.

**Role in this project:** These are the first gate in your CI pipeline — nothing that fails linting or formatting should ever reach `main`. This also matters for a portfolio project: consistent, typed code reads as "production," not "notebook."

**Why these:** `flake8` + `isort` + `pylint` used to be the standard combo, but `ruff` reimplements almost all of them in one much faster Rust binary, so most teams have already migrated. `black` remains the default formatter because it removes all debate about style. `mypy` is the most mature static type checker for Python; `pyright` is a faster alternative but slightly less common in Python-first (non-VSCode-only) teams.

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks: [{id: ruff}, {id: ruff-format}]
```

### Git + GitHub
**What it is in general:** Distributed version control (Git) plus a hosting/collaboration platform (GitHub) with pull requests, code review, and CI integration.

**Role in this project:** The backbone everything else hangs off — DVC versions data *alongside* Git commits, GitHub Actions reads from the repo, ArgoCD (if you build it) syncs Kubernetes state *from* the repo. This is why "GitOps" works: Git becomes the single source of truth for code, config, *and* infrastructure state.

**Why this one:** GitLab and Bitbucket are equally valid technically; GitHub is used here because it has the largest ecosystem of pre-built Actions and is what you'll most likely encounter at a Berlin startup.

---

## 2. Data Layer

### DVC (Data Version Control)
**What it is in general:** Git for large files. It stores lightweight pointer files (`.dvc`) in Git while the actual data/model binaries live in a remote (S3, GCS, local disk, etc.).

**Role in this project:** Your raw dataset and trained model artifacts get versioned exactly like code. If a model regresses, you can `git checkout` an old commit and `dvc pull` the exact data it was trained on — this reproducibility is the whole point of MLOps.

**Alternatives & why DVC:** **Git LFS** solves storage but has no pipeline/versioning semantics for ML specifically. **lakeFS** is built for data-lake scale (petabytes, many teams) — overkill here. **Pachyderm** couples versioning with its own pipeline engine, which is more infrastructure than a solo project needs. DVC is the lightest-weight option that still integrates natively with Git, which is why job postings list it as a standalone skill.

```bash
dvc init
dvc remote add -d storage s3://churnops-dvc/store
dvc add data/raw/telco_churn.csv
git add data/raw/telco_churn.csv.dvc data/.gitignore
git commit -m "Track raw churn dataset with DVC"
dvc push
```

### Pandera / Great Expectations (data validation)
**What it is in general:** Libraries that let you declare a schema/contract for your data (types, ranges, allowed categories, nullability) and validate incoming data against it automatically.

**Role in this project:** Sits at the start of the pipeline — if new data doesn't match the expected schema (e.g. a negative `MonthlyCharges`, a new unseen category), the pipeline fails loudly *before* a bad model gets trained on garbage. This is the "data contract" pattern real teams use to stop silent data-quality regressions.

**Why Pandera over Great Expectations:** Great Expectations is more powerful and produces shareable HTML "data docs," but it's heavier — it wants its own project structure and config store. **Pandera** validates a pandas DataFrame with a plain Python schema object, no extra services required, which fits the "lightweight" constraint while teaching the identical concept. Use Great Expectations if you want the extra portfolio weight of a more enterprise-y tool.

```python
import pandera as pa

schema = pa.DataFrameSchema({
    "tenure": pa.Column(int, pa.Check.ge(0)),
    "MonthlyCharges": pa.Column(float, pa.Check.ge(0)),
    "Churn": pa.Column(str, pa.Check.isin(["Yes", "No"])),
})

schema.validate(df)  # raises pandera.errors.SchemaError on violation
```

---

## 3. Experimentation

### MLflow (experiment tracking + model registry)
**What it is in general:** An open-source platform with four components: Tracking (log params/metrics/artifacts per run), Projects (packaging), Models (a standard model format), and Model Registry (stage-based versioning: Staging/Production/Archived).

**Role in this project:** Every training run — baseline or Optuna trial — logs to MLflow, so you can compare 200 runs in a UI instead of scrolling through terminal output. The Model Registry is what your CI/CD promotion gate reads from to know "what is currently in Production."

**Alternatives & why MLflow:** **Weights & Biases (W&B)** has a nicer UI and is very popular, but it's cloud-hosted-first (self-hosting is more work) and adds a vendor dependency. **Neptune.ai** and **ClearML** are similar SaaS-first competitors. **Kubeflow** bundles tracking into a much larger Kubernetes-native platform — massive overkill for a single-model project. MLflow is chosen because it's fully self-hostable with `docker-compose`, free, and is the single most-requested tracking tool by name in Berlin job postings.

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("churn-prediction")

with mlflow.start_run():
    mlflow.log_params({"max_depth": 6, "n_estimators": 300})
    mlflow.log_metric("roc_auc", 0.874)
    mlflow.xgboost.log_model(model, artifact_path="model")
```

```python
# Promotion gate: only register if it's a real improvement
client = mlflow.MlflowClient()
result = mlflow.register_model("runs:/<run_id>/model", "churn-classifier")
client.transition_model_version_stage(
    name="churn-classifier", version=result.version, stage="Production"
)
```

### Optuna (hyperparameter tuning)
**What it is in general:** A hyperparameter optimization framework using Bayesian search (Tree-structured Parzen Estimator by default) instead of brute-force grid search, with built-in early pruning of bad trials.

**Role in this project:** Replaces manually guessing `max_depth` and `learning_rate` values. Each trial is logged as a nested MLflow run, so tuning history is fully auditable — you can later show *why* a particular config won.

**Alternatives & why Optuna:** **Grid/Random search** (scikit-learn's `GridSearchCV`) is simple but wastes compute exploring bad regions of the search space exhaustively. **Ray Tune** is more powerful for distributed tuning across many machines/GPUs, but that's solving a problem you don't have on a single CPU project — it adds a cluster-management layer for no benefit here. **Hyperopt** is Optuna's older, less actively maintained predecessor. Optuna is the right complexity level: smarter than grid search, without Ray's distributed-systems overhead.

```python
import optuna

def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
    }
    model = xgb.XGBClassifier(**params).fit(X_train, y_train)
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=50)
```

---

## 4. Orchestration

### Apache Airflow
**What it is in general:** A workflow orchestrator where you define pipelines as DAGs (Directed Acyclic Graphs) of tasks in Python. It handles scheduling, retries, dependency ordering, and provides a UI to see run history.

**Role in this project:** Chains together `validate_data → train → evaluate → conditionally_register` as one automated, schedulable pipeline, rather than you manually running scripts in order. This is also the trigger point for automated retraining later.

**Alternatives & why Airflow:** **Prefect** is more modern, more Pythonic (decorators instead of verbose Operator classes), and much lighter to run — a genuinely better fit for a *solo* project. **Dagster** adds strong data-asset-aware lineage tracking. **Kubeflow Pipelines** is Kubernetes-native and ties orchestration directly to K8s, which is elegant if you're already fully committed to that platform. Airflow is picked here specifically because it's the most-named orchestrator in Berlin job listings — the roadmap explicitly trades a bit of setup pain for CV relevance. If you're short on time, swapping in Prefect changes almost none of the surrounding architecture.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

with DAG("churn_retraining", schedule="@weekly", start_date=datetime(2026, 1, 1)) as dag:
    validate = PythonOperator(task_id="validate_data", python_callable=validate_data)
    train = PythonOperator(task_id="train_model", python_callable=train_model)
    evaluate = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model)

    validate >> train >> evaluate
```

---

## 5. Containerization & Serving

### Docker + Docker Compose
**What it is in general:** Docker packages an application with its exact runtime environment into a portable image; Compose lets you define and run multiple containers (services) together with one config file.

**Role in this project:** Every component — MLflow server, Airflow, the FastAPI serving app, Prometheus, Grafana — runs as its own container. Compose is your entire local "cloud" during development, and the same images later run in Kubernetes unchanged.

**Why this one:** There isn't really a mainstream alternative to Docker at this layer anymore (Podman is a compatible alternative runtime, but the workflow is identical). This is genuinely foundational, non-negotiable infrastructure.

```dockerfile
# docker/serve/Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen --no-dev
COPY src/ src/
CMD ["uv", "run", "uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
services:
  mlflow:
    build: ./docker/mlflow
    ports: ["5000:5000"]
  api:
    build:
      context: .
      dockerfile: docker/serve/Dockerfile
    ports: ["8000:8000"]
    depends_on: [mlflow]
```

### FastAPI (model serving)
**What it is in general:** A modern Python web framework for building APIs, with automatic request validation (via Pydantic) and auto-generated OpenAPI docs.

**Role in this project:** Wraps your trained model in a `/predict` HTTP endpoint that any downstream system could call — this is the "last mile" that turns a model artifact into an actual product.

**Alternatives & why FastAPI:** **Flask** is simpler but has no built-in request validation or async support, so you'd hand-roll input checking. **BentoML** is purpose-built for ML serving specifically (handles model packaging + serving in one tool, with adaptive batching) and is worth knowing about — but it's a higher-level abstraction that hides some of the HTTP-layer mechanics you learn by building it yourself in FastAPI. **TorchServe/TF-Serving** are framework-specific and don't fit an XGBoost model well. FastAPI is chosen because it's the most commonly named serving framework across ML job postings and forces you to understand the request/response contract explicitly.

```python
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc

app = FastAPI()
model = mlflow.pyfunc.load_model("models:/churn-classifier/Production")

class ChurnRequest(BaseModel):
    tenure: int
    MonthlyCharges: float
    Contract: str

@app.post("/predict")
def predict(req: ChurnRequest):
    proba = model.predict([req.model_dump()])[0]
    return {"churn_probability": float(proba)}

@app.get("/health")
def health():
    return {"status": "ok"}
```

---

## 6. CI/CD & Registry

### GitHub Actions
**What it is in general:** GitHub's built-in CI/CD system — YAML-defined workflows that run on events like pull requests or pushes (test, build, deploy).

**Role in this project:** Runs your lint/test suite on every PR, then on merge to `main` builds Docker images, tags them with the git SHA, pushes to the registry, and optionally kicks off a training-validation job that blocks the merge if model metrics regress.

**Why this one:** **GitLab CI** and **Jenkins** are equally capable — GitLab CI is arguably even more common in Germany's enterprise scene, and Jenkins is still everywhere in older infra. GitHub Actions is chosen here mainly because your code already lives on GitHub, so it's zero extra setup and has the largest library of ready-made Actions to reuse.

```yaml
# .github/workflows/ci.yml
name: CI
on: [pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install uv && uv sync
      - run: uv run pytest
      - run: uv run ruff check .
```

### GitHub Container Registry (GHCR)
**What it is in general:** A Docker image registry, i.e. a place to store and version-tag container images so they can be pulled by any deployment target.

**Role in this project:** After CI builds your `serve` and `train` images, they're pushed here tagged with the git commit SHA — this is what your Kubernetes manifests reference (`image: ghcr.io/you/churnops-serving:sha-abc123`).

**Alternatives & why GHCR:** **Docker Hub** is the original and most universal registry, but has rate limits on the free tier. **AWS ECR** is the natural choice if you're deploying to AWS (and the roadmap does use it for the Terraform-provisioned EC2/K8s path). GHCR is used for CI-built images specifically because it needs zero extra credentials — it authenticates automatically using your existing GitHub Actions token.

---

## 7. Infrastructure & Cloud

### Terraform
**What it is in general:** An Infrastructure-as-Code (IaC) tool — you declare the cloud resources you want in config files, and it computes and applies the diff against what currently exists.

**Role in this project:** Provisions your S3 bucket (DVC + MLflow artifact storage), ECR repository, and optionally an EC2 instance — all declaratively, so your entire cloud footprint can be destroyed and recreated identically from Git history.

**Alternatives & why Terraform:** **AWS CloudFormation/CDK** are AWS-native but lock you into that one cloud's tooling. **Pulumi** lets you write IaC in actual Python instead of HCL, which is genuinely appealing for a Python-heavy project — worth trying if you want a lighter learning curve. Terraform is chosen because it's cloud-agnostic and by far the most commonly required IaC tool across job postings, regardless of which cloud a company uses.

```hcl
resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = "churnops-mlflow-artifacts"
}

resource "aws_ecr_repository" "serving_app" {
  name = "churnops-serving"
}
```

### AWS (cloud provider)
**What it is in general:** A cloud platform offering storage (S3), compute (EC2, Fargate, Lambda), managed Kubernetes (EKS), and managed ML (SageMaker), billed on usage.

**Role in this project:** Hosts your DVC/MLflow artifact storage (S3) and container registry backup (ECR); optionally a small EC2 instance to run the Compose stack "for real," outside your laptop.

**Alternatives & why AWS:** **Azure** and **GCP** are equally valid and equally represented in Berlin job listings (Azure especially, given SAP/enterprise clients) — pick whichever you're more likely to interview for. AWS is the default recommendation here mainly because it has the most mature free tier and the largest volume of tutorials/Terraform modules if you get stuck.

### Kubernetes (via `kind`)
**What it is in general:** A container orchestration system — schedules containers across machines, handles scaling, self-healing (restarting crashed pods), and rolling updates. `kind` ("Kubernetes IN Docker") runs a real K8s cluster entirely inside Docker containers on your laptop, at zero cloud cost.

**Role in this project:** Deploys your serving app as a `Deployment` (multiple replicas, self-healing) behind a `Service`, with a `HorizontalPodAutoscaler` to scale replicas under load — the same manifests you'd use against a real cloud cluster (EKS/GKE/AKS), just running locally.

**Alternatives & why `kind`:** **Minikube** is the older, heavier alternative (runs a VM, not just containers). **k3d** (K3s in Docker) is another lightweight option, nearly equivalent to `kind`. A real managed cluster (EKS) teaches you cloud-specific IAM/networking quirks but costs money continuously just sitting idle — not worth it for a portfolio project. `kind` is chosen because it gives you the *identical* `kubectl`/Helm experience for free.

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: churn-api
spec:
  replicas: 2
  selector:
    matchLabels: {app: churn-api}
  template:
    metadata:
      labels: {app: churn-api}
    spec:
      containers:
        - name: api
          image: ghcr.io/you/churnops-serving:latest
          ports: [{containerPort: 8000}]
```

---

## 8. Monitoring

### Prometheus + Grafana
**What they are in general:** Prometheus is a time-series metrics database that "scrapes" (pulls) metrics from your services on an interval; Grafana is a dashboarding/alerting tool that queries Prometheus and visualizes it.

**Role in this project:** Track infra-level health of your serving API — request latency, throughput, error rate — so you can see load and failures over time, and set alerts (e.g. "error rate > 5% for 5 minutes").

**Alternatives & why this pair:** **Datadog** and **New Relic** are SaaS alternatives that do the same job with far less setup — but they're paid products and hide the mechanics you're trying to learn. Prometheus + Grafana is the open-source standard combination named repeatedly across MLOps tooling guides, and it's what you're most likely to encounter self-hosted at a startup.

```python
# One line adds a /metrics endpoint Prometheus can scrape
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
```

### Evidently AI
**What it is in general:** A Python library purpose-built for ML monitoring — generates reports comparing two datasets (e.g. training data vs. live production data) for statistical drift, and can compute classification-quality decay if ground truth becomes available.

**Role in this project:** Answers the ML-specific question Prometheus can't: not "is the service up," but "is the *model* still valid" — are incoming customer records starting to look statistically different from what it was trained on?

**Alternatives & why Evidently:** **NannyML** is a close competitor with a slightly different focus (estimating performance without ground truth). **WhyLabs** and **Arize** are SaaS observability platforms for ML specifically. Evidently is chosen because it's pure Python, free, self-hostable, and was named directly as the go-to lightweight monitoring tool in current MLOps tooling guides.

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=training_df, current_data=live_requests_df)
report.save_html("reports/drift_report.html")
```

---

## 9. Testing & Docs

### pytest
**What it is in general:** Python's de facto standard testing framework — simple function-based tests, fixtures for reusable setup, and a huge plugin ecosystem.

**Role in this project:** Three layers of tests matter here: unit tests (does `build_features()` return the right shape?), data tests (does the schema validation catch bad rows?), and a basic model-contract test (does the model output a probability between 0 and 1?).

**Why this one:** Python's built-in `unittest` is more verbose and class-based; `pytest` won essentially universal adoption because of its plain-function syntax and fixture system. There isn't a serious competitor at this layer anymore.

```python
def test_feature_engineering_output_shape():
    df = load_sample_data()
    features = build_features(df)
    assert features.shape[1] == 19

def test_model_output_is_valid_probability():
    proba = model.predict_proba(sample_input)[0][1]
    assert 0.0 <= proba <= 1.0
```

### MkDocs
**What it is in general:** A static site generator that turns Markdown files into a browsable documentation website.

**Role in this project:** Hosts your architecture diagram, setup instructions, and an incident "runbook" (e.g. "the drift alert fired, here's what to check") — the kind of documentation real infra teams are graded on maintaining.

**Alternatives & why MkDocs:** **Sphinx** is more powerful (better for auto-generating API reference docs from docstrings) but has a steeper learning curve and reST syntax quirks. **Docusaurus** is React-based and heavier to set up. MkDocs (especially with the `mkdocs-material` theme) gets you a genuinely good-looking docs site from plain Markdown in minutes — proportionate to the size of this project.

---

## 10. Stretch-Goal Tools

### Feast (feature store)
**What it is in general:** A feature store — a centralized place to define, compute, and serve ML features consistently for both training (batch) and serving (low-latency online lookups), preventing "training-serving skew."

**Role in this project (if added):** Even with one model, implementing Feast demonstrates you understand *point-in-time correctness* — the subtle bug where a feature computed for training accidentally uses information that wouldn't have been available at prediction time in production.

**Why worth learning despite being stretch:** It's explicitly called out in tooling guides as something to add "if you're sharing features across models" — with a single model, it's not strictly necessary, which is exactly why including it signals you understand *when* to reach for a tool, not just how to install it.

```python
from feast import FeatureView, Field, Entity
from feast.types import Int64, Float32

customer = Entity(name="customer_id")
churn_features = FeatureView(
    name="customer_features",
    entities=[customer],
    schema=[Field(name="tenure", dtype=Int64), Field(name="MonthlyCharges", dtype=Float32)],
    source=parquet_file_source,
)
```

### ArgoCD (GitOps)
**What it is in general:** A Kubernetes controller that continuously syncs your cluster's actual state to match what's declared in a Git repository — if you edit a manifest and push, ArgoCD applies the change automatically; if someone manually changes the live cluster, ArgoCD reverts it.

**Role in this project (if added):** Replaces a manual `kubectl apply` deployment step with a fully auditable, Git-driven one — every production change has a corresponding commit, which is the audit trail real compliance-conscious companies require.

**Alternatives & why ArgoCD:** **Flux** is a close, equally valid competitor with a similar philosophy. ArgoCD is picked mainly for its more approachable web UI, which makes the "desired state vs. live state" concept easier to see while you're still learning it.

---

## How It All Connects

```
 Git push
    │
    ▼
GitHub Actions (lint, test, build) ──► GHCR (image registry)
    │
    ▼
Airflow DAG: validate (Pandera) → train (XGBoost) → tune (Optuna)
    │                                                     │
    ▼                                                     ▼
DVC (data versions)                              MLflow (tracking + registry)
                                                          │
                                                          ▼
                                        FastAPI serving app ◄── loads "Production" model
                                                          │
                                        ┌─────────────────┼─────────────────┐
                                        ▼                                   ▼
                              Prometheus + Grafana                 Evidently (drift reports)
                              (is the service healthy?)             (is the model still valid?)
                                                          │
                                                          ▼
                                        Drift/schedule trigger → back to Airflow DAG
```

Everything upstream of "FastAPI serving app" is about getting a *trustworthy* model into production; everything downstream is about knowing when it stops being trustworthy and needs to be retrained. That loop — train, deploy, observe, retrain — is the actual definition of MLOps, and now every box in that diagram is a tool you'll have hands-on experience with.
