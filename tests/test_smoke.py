from pathlib import Path

from src.data import prepare_diabetes_data, prepare_healthcare_data
from src.federated import FederatedConfig, run_federated_training


def test_federated_smoke_run():
    data = prepare_diabetes_data()
    config = FederatedConfig(num_clients=2, rounds=1, local_epochs=1, use_dp=False)
    _, history, clients = run_federated_training(
        data.X_train[:100], data.y_train[:100], data.X_test[:50], data.y_test[:50], config
    )
    assert len(history) == 1
    assert len(clients) == 2
    assert 0 <= history[-1]["accuracy"] <= 1


def test_cleveland_heart_dataset_loads():
    data = prepare_healthcare_data(
        path=Path("heart_cleveland_upload.csv"),
        target_name="condition",
        dataset_label="Heart Disease Cleveland UCI",
    )
    assert data.target_name == "condition"
    assert {"age", "cp", "trestbps", "chol", "thalach"}.issubset(data.feature_names)
    assert data.X_train.shape[1] == 13
