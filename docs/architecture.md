# Architecture

## Goal

Provide a single tabular federated learning framework that can run separate
healthcare prediction experiments per dataset.

## Current implementation

- Dataset registry for `heart`, `diabetes`, `liver`, and `kidney`
- Shared tabular preprocessing contract returning the same `TabularSplit` shape
- Custom NumPy logistic regression model
- FedAvg and FedProx simulation across synthetic clients
- Centralized baseline for comparison
- JSON metrics, Markdown summary, and plot export

## Extension path

- Add privacy-preserving strategy variants such as differential privacy
- Add richer comparison dashboards across datasets and strategies
- Plug in alternative model families while keeping the same dataset registry
