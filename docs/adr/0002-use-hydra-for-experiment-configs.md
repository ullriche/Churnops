# ADR-0002: Use Hydra for Experiment Config

* **Status:** Accepted
* **Date:** 2026-09-11
* **Deciders:** Eike Christoph Ullrich / ullriche

## Context and Problem Statement
During training and tuning, model-, data-, and training configurations need to be modifieable

## Considered Options
* **Option 1:** Argparse - Passing all hyperparameters as command line arguments
* **Option 2:** Own Implementation - Own implementation of configs using .yaml files
* **Option 3:** Hydra - Industry used config management system

## Decision Outcome
Chosen option: "[Option 1]", because not need to invent the wheel.

## Consequences
### Positive
* Used in industry
* Configs are overridable via cli-args
* No need to pass all hyperparameters as cli-args

### Negative / Trade-offs
* May be overkill for this project
