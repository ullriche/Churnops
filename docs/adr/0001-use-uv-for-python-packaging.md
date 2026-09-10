# ADR-0001: Use `uv` for python packaging

* **Status:** Accepted
* **Date:** 2026-09-10
* **Deciders:** ullriche

## Context and Problem Statement
Installing of dependencies. Reproducable environment (virtual, docker). Handling requirement trees of dependencies.

## Considered Options
* **Option 1:** uv - A packaging tool
* **Option 2:** pip - The default (and manual) way of installing packages
* **Option 3:** poetry - Another packaging tool

## Decision Outcome
Chosen option: "Option 1", because it is the indstry standard.

## Consequences
### Positive
* Industry Standard
* Fast
* Independent of the operation system
