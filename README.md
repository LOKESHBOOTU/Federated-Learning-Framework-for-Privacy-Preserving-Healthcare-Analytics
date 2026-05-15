# Federated Learning Framework for Privacy-Preserving Healthcare Analytics

This project implements a presenter-friendly federated learning framework for healthcare analytics. The main use case is diabetes risk prediction using the CDC Diabetes Health Indicators dataset.

## Recommended Major Project Title

**Privacy-Preserving Federated Learning Framework for Diabetes Risk Prediction using Differential Privacy**

## What It Includes

- CDC and original healthcare dataset loading and preprocessing
- Centralized baselines: Logistic Regression, Random Forest, and MLP
- Custom federated MLP classifier
- FedAvg aggregation across simulated hospitals
- IID and non-IID client partitioning
- Differential privacy simulation with update clipping and Gaussian noise
- Streamlit dashboard for presentation and explanation
- Single patient prediction demo
- Real-time triage, batch screening, hospital monitoring, and downloadable screening report

## Dataset

Place the CDC Diabetes Health Indicators CSV here:

```text
data/raw/cdc_diabetes_health_indicators.csv
```

The app also recognizes:

```text
data/raw/diabetes_binary_health_indicators_BRFSS2015.csv
data/raw/diabetes_012_health_indicators_BRFSS2015.csv
```

Original project CSVs such as `diabetes.csv`, `heart.csv`, `kidney_disease.csv`, and `liver.csv` are supported as fallbacks. If none of those exact names exist, the loader uses the first `.csv` file found in `data/raw/`.

## Saved CDC Results

The dashboard has two model modes:

```text
Saved CDC results
Simulate hospitals
```

`Saved CDC results` uses a fixed configuration:

```text
Hospitals: 5
Federated rounds: 15
Local epochs per hospital: 2
Partition: IID
Differential privacy: enabled
DP noise: 0.05
Learning rate: 0.03
```

The first time this mode is opened, the app trains the centralized baselines and federated model, then saves reusable artifacts under:

```text
results/saved_cdc_run/
```

After that, the dashboard reloads those saved results instead of retraining every time.

`Simulate hospitals` lets the user change hospital count, rounds, local epochs, IID/non-IID partitioning, differential privacy, noise, and learning rate. In this mode the model is trained for the selected simulation settings.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run The Dashboard

```bash
streamlit run app.py
```

## Run CLI Experiment

With the CDC dataset:

```bash
python run_experiment.py --clients 5 --rounds 20 --local-epochs 2 --partition iid
```

For non-IID:

```bash
python run_experiment.py --clients 5 --rounds 20 --partition non-iid
```

Without differential privacy:

```bash
python run_experiment.py --no-dp
```

## Project Structure

```text
.
|-- app.py
|-- dashboard/
|   `-- app.py
|-- run_experiment.py
|-- requirements.txt
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- clients/
|-- docs/
|   `-- architecture.md
|-- results/
|   |-- metrics/
|   |-- models/
|   `-- plots/
|-- src/
|   |-- baselines.py
|   |-- config.py
|   |-- data.py
|   |-- evaluation.py
|   |-- federated.py
|   `-- model.py
`-- tests/
    `-- test_smoke.py
```

## Best Model Choice

The final recommended model is:

```text
Federated MLP + FedAvg + Differential Privacy
```

This is strong enough for a final-year AIML project, but still understandable during evaluation.
