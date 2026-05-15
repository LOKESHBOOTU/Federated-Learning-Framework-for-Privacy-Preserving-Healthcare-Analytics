from __future__ import annotations

import argparse
import json

import pandas as pd

from src.baselines import train_baselines
from src.config import METRICS_DIR
from src.data import prepare_diabetes_data
from src.federated import FederatedConfig, run_federated_training


def main():
    parser = argparse.ArgumentParser(description="Run federated diabetes analytics experiment.")
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--partition", choices=["iid", "non-iid"], default="iid")
    parser.add_argument("--no-dp", action="store_true")
    parser.add_argument("--noise", type=float, default=0.05)
    args = parser.parse_args()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    data = prepare_diabetes_data()
    baseline_results, _ = train_baselines(data.X_train, data.y_train, data.X_test, data.y_test)

    config = FederatedConfig(
        num_clients=args.clients,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        partition=args.partition,
        use_dp=not args.no_dp,
        noise_multiplier=args.noise,
    )
    _, history, _ = run_federated_training(data.X_train, data.y_train, data.X_test, data.y_test, config)

    history_df = pd.DataFrame(history)
    history_df.to_csv(METRICS_DIR / "federated_history.csv", index=False)
    with open(METRICS_DIR / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(baseline_results, f, indent=2)

    print(f"Dataset: {data.name} ({data.source_path})")
    print("Baseline metrics:")
    print(json.dumps(baseline_results, indent=2))
    print("Final federated metrics:")
    print(history_df.tail(1).to_string(index=False))


if __name__ == "__main__":
    main()
