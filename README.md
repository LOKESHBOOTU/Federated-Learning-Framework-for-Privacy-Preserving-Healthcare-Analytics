# Privacy-Preserving Federated Learning Framework for Healthcare Analytics

## Description

This project predicts **diabetes risk** and **heart disease risk** using machine learning while preserving patient data privacy through **Federated Learning** and **Differential Privacy**. It combines a reproducible preprocessing and training pipeline, centralized baseline models, saved federated model artifacts, evaluation reports, training history files, and a Streamlit dashboard for interactive prediction.

Healthcare data is highly sensitive, and hospitals cannot freely share raw patient records. A normal centralized machine learning system requires all data to be collected in one place, which creates privacy and security risks. This project solves that problem by simulating multiple hospitals as clients. Each hospital trains locally, and only model updates are shared with the central server.

The final system supports two prediction modules:

```text
Diabetes Prediction
Heart Disease Prediction
```

The main model is:

```text
Federated MLP + FedAvg + Differential Privacy
```

## What Is Federated Learning

Federated Learning is a machine learning technique where data stays with local clients, such as hospitals, and the central server receives only model updates.

In simple terms:

```text
Hospital data stays inside the hospital.
Each hospital trains a local model.
The server combines only the learned updates.
Raw patient records are never directly shared.
```

This makes federated learning useful for healthcare, finance, and other privacy-sensitive domains.

## What Is Differential Privacy

Differential Privacy is a privacy technique that reduces the risk of leaking information from model updates. In this project, each local model update is:

```text
1. Clipped to limit very large updates
2. Noised using Gaussian noise
3. Sent to the server for FedAvg aggregation
```

This helps protect patient-level information while still allowing the global model to learn useful disease prediction patterns.

## How It Works

1. The user selects either Diabetes Prediction or Heart Disease Prediction in the dashboard.
2. The selected healthcare dataset is loaded and preprocessed.
3. Centralized baseline models are trained for comparison.
4. The training data is split into simulated hospital clients.
5. Each hospital trains a local MLP model on its own data.
6. Differential privacy is applied to local model updates.
7. The server aggregates updates using FedAvg.
8. The final global model is evaluated using classification metrics.
9. Saved model artifacts are loaded by the Streamlit dashboard.
10. The user enters patient details and receives a low, moderate, or high risk prediction.

## Objectives

- Predict diabetes and heart disease risk from healthcare features
- Build a privacy-preserving training framework using Federated Learning
- Add Differential Privacy through update clipping and Gaussian noise
- Compare the federated model with centralized machine learning baselines
- Save model artifacts so predictions can run without retraining
- Provide an interactive Streamlit dashboard for real-time prediction
- Visualize model comparison, dataset information, and federated training progress

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Plotly
- Streamlit
- Custom NumPy MLP
- Federated Averaging
- Differential Privacy simulation
- Pytest

## Dataset Information

### Diabetes Dataset

Dataset used:

```text
data/raw/diabetes_binary_5050split_health_indicators_BRFSS2015.csv
```

Target column:

```text
Diabetes_binary
```

Label meaning:

```text
0 = No diabetes
1 = Diabetes / Prediabetes
```

Main features include:

```text
HighBP
HighChol
BMI
GenHlth
PhysHlth
PhysActivity
DiffWalk
HeartDiseaseorAttack
Age
Sex
```

Some less important optional fields are hidden from the main prediction form and filled internally using dataset median values.

### Heart Disease Dataset

Dataset used:

```text
heart_cleveland_upload.csv
```

Target column:

```text
condition
```

Label meaning:

```text
0 = No heart disease
1 = Heart disease present
```

Main user-facing features include:

```text
age
sex
cp
trestbps
chol
thalach
exang
```

In the dashboard, these are shown as user-friendly fields:

```text
Age
Sex
Chest pain type
Resting blood pressure
Serum cholesterol
Maximum heart rate achieved
Exercise-induced angina
```

Advanced heart features include:

```text
Fasting blood sugar > 120 mg/dL
Resting ECG result
ST depression from exercise
Peak exercise ST slope
Major vessels colored by fluoroscopy
Thalassemia result
```

## Project Structure

```text
Federated Learning framework for privacy preserving system for healthcare Analytics/
|-- app.py
|-- dashboard/
|   `-- app.py
|-- data/
|   |-- README.md
|   `-- raw/
|       `-- diabetes_binary_5050split_health_indicators_BRFSS2015.csv
|-- docs/
|   |-- architecture.md
|   `-- presenter_notes.md
|-- results/
|   |-- saved_cdc_run/
|   |-- saved_diabetes_run/
|   `-- saved_heart_disease_run/
|-- src/
|   |-- artifacts.py
|   |-- baselines.py
|   |-- config.py
|   |-- data.py
|   |-- evaluation.py
|   |-- federated.py
|   `-- model.py
|-- tests/
|   `-- test_smoke.py
|-- heart_cleveland_upload.csv
|-- requirements.txt
|-- run_experiment.py
`-- README.md
```

## Requirements

- Python 3.10 or later
- pip
- Streamlit-compatible browser

## Installation And Setup

```bash
git clone https://github.com/LOKESHBOOTU/Federated-Learning-Framework-for-Privacy-Preserving-Healthcare-Analytics.git
cd Federated-Learning-Framework-for-Privacy-Preserving-Healthcare-Analytics
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```

If Windows or OneDrive locks files inside `.venv`, use the local package-folder fallback:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --target .python_packages -r requirements.txt
$env:PYTHONPATH = (Resolve-Path .python_packages).Path
python -m streamlit run app.py
```

## Methodology / Workflow

### Data Collection

The project uses two healthcare datasets:

- CDC Diabetes Health Indicators dataset
- Heart Disease Cleveland UCI dataset

### Data Preprocessing

The preprocessing pipeline:

- Loads the selected dataset
- Identifies the target column
- Converts target labels into binary values
- Handles missing values
- Encodes categorical columns
- Scales numeric features using StandardScaler
- Splits data into train and test sets

The current split is:

```text
Training data: 80%
Testing data: 20%
```

### Baseline Model Training

Centralized baseline models are trained first:

```text
Logistic Regression
Random Forest
Centralized MLP
```

These models provide comparison points for the federated model.

### Federated Client Simulation

The training data is split into multiple simulated hospitals. Each hospital behaves like a federated client.

The project supports:

```text
IID partitioning
Non-IID partitioning
```

### Local Model Training

Each hospital trains a local MLP model on its own local data. Raw patient records remain local and are not sent to the server.

### Differential Privacy

Before local updates are shared:

```text
Model update is clipped
Gaussian noise is added
Private update is sent to server
```

### FedAvg Aggregation

The central server aggregates hospital model updates using Federated Averaging:

```text
Global model = weighted average of local client models
```

### Model Evaluation

The model is evaluated using:

```text
Accuracy
Precision
Recall
F1-score
ROC-AUC
Training loss
```

### Prediction

The Streamlit app loads saved artifacts and predicts disease risk from user-entered patient details without retraining every time.

## Models Used

### Centralized Machine Learning Models

- Logistic Regression
- Random Forest
- Centralized MLP

### Main Federated Model

- Custom NumPy MLP
- FedAvg aggregation
- Differential Privacy enabled

## Results / Accuracy

### Diabetes Prediction

Latest saved evaluation metrics:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7458 | 0.7372 | 0.7639 | 0.7503 | 0.8232 |
| Random Forest | 0.7351 | 0.7166 | 0.7779 | 0.7460 | 0.8108 |
| Centralized MLP | 0.7325 | 0.7233 | 0.7531 | 0.7379 | 0.8048 |
| Federated MLP + FedAvg + DP | 0.7526 | 0.7291 | 0.8038 | 0.7646 | 0.8264 |

### Heart Disease Prediction

Latest saved evaluation metrics:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9167 | 1.0000 | 0.8214 | 0.9020 | 0.9531 |
| Random Forest | 0.8500 | 0.9524 | 0.7143 | 0.8163 | 0.9425 |
| Centralized MLP | 0.8500 | 0.9130 | 0.7500 | 0.8235 | 0.9587 |
| Federated MLP + FedAvg + DP | 0.9000 | 0.9583 | 0.8214 | 0.8846 | 0.9096 |

## Saved Model Artifacts

The project includes saved artifacts so the dashboard can load trained results directly.

```text
results/saved_diabetes_run/
results/saved_heart_disease_run/
```

Each saved run includes:

```text
baseline_metrics.json
client_summary.csv
federated_history.csv
federated_model.npz
metadata.json
```

## Screenshots / Output

The project includes a Streamlit interface for disease prediction, model comparison, dataset inspection, and federated training visualization.
<img width="1435" height="704" alt="image" src="https://github.com/user-attachments/assets/8e7a1625-60d5-4a4c-bba3-4180881d4ace" />


### Prediction Demo

Users can enter patient health details and receive a disease risk output.
<img width="1421" height="862" alt="image" src="https://github.com/user-attachments/assets/3cbdeec9-64d1-4498-930f-dc45ad93d5dd" />


### Model Comparison

The dashboard compares:

```text
Logistic Regression
Random Forest
Centralized MLP
Federated MLP + FedAvg + DP
```
<img width="1434" height="731" alt="image" src="https://github.com/user-attachments/assets/bcb3fee6-ebaa-4b23-9b3b-ece07663ab05" />

### Dataset View

The dashboard displays dataset preview, total records, feature count, and class balance.

### Federated Training Progress

The dashboard plots federated training history using:

```text
accuracy
f1
roc_auc
loss
```

## Features

- Diabetes prediction
- Heart disease prediction
- Privacy-preserving federated learning
- Differential privacy with clipping and Gaussian noise
- Centralized baseline model comparison
- IID and non-IID hospital simulation
- Saved model artifacts for quick prediction
- Streamlit dashboard
- User-friendly heart disease input labels
- Advanced optional details section
- Dataset preview and class balance charts
- Federated training progress visualization
- Smoke tests for key pipeline checks

## Applications

- Academic healthcare analytics projects
- Privacy-preserving machine learning demonstrations
- Federated learning research prototypes
- Disease prediction dashboards
- Comparative study of centralized and federated models
- Final-year AIML / CSE major project presentation

## Why This Project Is Useful

This project is useful because it demonstrates how disease prediction can be performed without directly collecting all patient data in one central location. It shows a practical privacy-aware machine learning workflow where hospitals can collaborate through model updates while keeping raw patient records private.

It also provides a dashboard, making the project easy to explain during academic evaluation or demonstration.

## Limitations

- The hospitals are simulated using dataset partitions, not real hospital servers
- Differential privacy is implemented as a simulation over model updates
- The heart disease dataset is small, so metrics can vary with configuration
- The system is not a medical diagnosis tool
- Predictions should not be used without clinical validation
- Secure aggregation and real distributed networking are not implemented

## Future Improvements

- Add real distributed federated clients
- Add secure aggregation
- Add explainable AI using SHAP or LIME
- Add more disease prediction modules
- Add authentication and role-based access for hospital users
- Deploy the dashboard as a live Streamlit cloud app
- Add automated experiment tracking
- Add downloadable prediction reports
- Add model drift monitoring for new healthcare data

## Author / Contributor

**Lokesh Bootu**

GitHub: [LOKESHBOOTU](https://github.com/LOKESHBOOTU)

## License

No license file has been added yet.

If this project is intended to be open-source, adding an MIT License would be a good next step.

## Deployment

This project can be run locally using the saved model artifacts. It is also structured for lightweight Streamlit deployment. The included model artifacts are small enough for simple demo hosting, while the datasets and saved results allow the app to run without retraining every time.
