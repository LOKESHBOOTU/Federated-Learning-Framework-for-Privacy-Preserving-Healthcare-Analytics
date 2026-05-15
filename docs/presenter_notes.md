# Presenter Notes

## One-Minute Explanation

This project solves a healthcare privacy problem. Instead of collecting patient data from every hospital into one central database, each hospital trains a local model on its own records. The server receives only model updates, aggregates them using FedAvg, and returns an improved global model. Differential privacy adds another protection layer by clipping and noising model updates.

## Why CDC Diabetes

The CDC Diabetes Health Indicators dataset is a strong major-project dataset because it is healthcare-focused, tabular, large enough for meaningful experiments, and includes lifestyle/demographic indicators that make privacy preservation relevant.

## Why Federated MLP

The final model is a small neural network with two hidden layers. It is more expressive than Logistic Regression, easier to explain than a very deep model, and works naturally with federated averaging because model weights from local clients can be averaged.

## Expected Questions

**Why not share data directly?**
Healthcare data is sensitive and often restricted by privacy laws and institutional policies.

**What does the server receive?**
Only model parameters or parameter updates, not patient rows.

**What is non-IID data?**
It means every hospital has a different patient distribution. This is realistic because hospitals serve different populations.

**What is the privacy-accuracy tradeoff?**
More differential privacy noise can improve privacy but may reduce accuracy.

**How can this be improved later?**
Add Flower, Opacus, secure aggregation, real hospital nodes, authentication, and model drift monitoring.

## Real-Time UI Features

The dashboard includes a practical workflow for presentation:

- Live patient triage with low, moderate, and high-risk categories
- Recommended next action for each predicted patient
- Batch screening simulation for multiple patients
- High-risk alert threshold control
- Hospital monitoring view showing diabetes-rate variation across clients
- Confusion matrix for operational evaluation
- Downloadable screening report
