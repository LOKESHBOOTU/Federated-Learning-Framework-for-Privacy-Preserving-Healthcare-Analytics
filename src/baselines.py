from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from src.evaluation import classification_metrics


def train_baselines(X_train, y_train, X_test, y_test, random_state: int = 42):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=random_state),
        "Centralized MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=300,
            random_state=random_state,
        ),
    }

    results = {}
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_test)[:, 1]
        else:
            prob = model.predict(X_test)
        results[name] = classification_metrics(y_test, prob)
        fitted[name] = model
    return results, fitted
