.PHONY: ingest preprocess train tune test lint mlflow-up mlflow-down

ingest:
	uv run python -m churnops.data.ingest

preprocess:
	uv run python -m churnops.data.preprocess

train:
	uv run python -m churnops.training.train $(ARGS)

tune:
	uv run python -m churnops.training.train -m $(ARGS)

test:
	uv run pytest

lint:
	uv run pre-commit run --all-files

mlflow-up:
	docker compose up -d mlflow

mlflow-down:
	docker compose down mlflow
