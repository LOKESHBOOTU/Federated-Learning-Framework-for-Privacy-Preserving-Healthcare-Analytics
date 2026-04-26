# Federated Healthcare Benchmark

This project turns the healthcare CSV datasets in this workspace into a reusable
federated learning benchmark for tabular healthcare prediction. It now supports
`heart`, `diabetes`, `liver`, and `kidney` with a shared NumPy logistic
regression baseline, FedAvg, and FedProx.

## Current scope

- Reusable project structure for tabular healthcare FL experiments
- Working centralized baseline for all four datasets
- Working FedAvg and FedProx simulations for all four datasets
- Result export to `results/metrics/`
- Training curve plots to `results/plots/`
- Benchmark summary export across completed runs

## Datasets in this workspace

- `heart.csv`
- `diabetes.csv`
- `indian_liver_patient.csv`
- `kidney_disease.csv`

The datasets do not share the same schema, so the correct design is one shared
federated framework with disease-specific experiments rather than a single model
trained across all rows.

## Project layout

```text
src/
  core/
  data/
  fl/
  models/
  experiments/
data/
  raw/
  processed/
results/
  metrics/
  plots/
docs/
run_experiment.py
requirements.txt
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

You can leave the CSV files in the project root. The code automatically checks
both the root folder and `data/raw/` when resolving dataset paths.

## Run a baseline experiment

```bash
python run_experiment.py baseline --dataset heart
python run_experiment.py baseline --dataset diabetes
python run_experiment.py baseline --dataset liver
python run_experiment.py baseline --dataset kidney
```

## Run a FedAvg experiment

```bash
python run_experiment.py federated --dataset heart --num-clients 5 --rounds 25
python run_experiment.py federated --dataset diabetes --strategy fedavg --num-clients 5 --rounds 25
```

## Run a FedProx experiment

```bash
python run_experiment.py federated --dataset kidney --strategy fedprox --proximal-mu 0.01
```

## Example with non-IID partitioning

```bash
python run_experiment.py federated --dataset liver --strategy fedavg --partition dirichlet --alpha 0.5
```

## Build the benchmark summary

```bash
python run_experiment.py compare
```

## Next extensions

1. Add differential privacy and secure aggregation variants
2. Add hyperparameter sweep support for each dataset
3. Add a simple dashboard for result exploration
