# Architecture

## High-Level Flow

```text
Healthcare Dataset
        |
Preprocessing and Scaling
        |
Client Partitioning
        |
+-------------------+  +-------------------+  +-------------------+
| Hospital Client 1 |  | Hospital Client 2 |  | Hospital Client N |
| Local MLP Training|  | Local MLP Training|  | Local MLP Training|
+---------+---------+  +---------+---------+  +---------+---------+
          |                      |                      |
          | Model updates only   | Model updates only   |
          +----------------------+----------------------+
                                 |
                        Differential Privacy
                       Clip update + add noise
                                 |
                            FedAvg Server
                                 |
                           Global MLP Model
                                 |
                    Metrics, Explainability, Demo UI
```

## Why This Architecture Preserves Privacy

Raw patient records remain local to each simulated hospital. The central server only receives model parameters or parameter updates. Differential privacy further reduces leakage risk by clipping unusually large updates and adding Gaussian noise before aggregation.

## Experiments To Show

- Centralized ML vs federated ML
- IID vs non-IID hospital distributions
- With differential privacy vs without differential privacy
- Accuracy, precision, recall, F1 score, and ROC-AUC
- Client-wise record distribution and positive-class ratio
- Patient triage, batch screening, hospital monitoring, and screening report export
