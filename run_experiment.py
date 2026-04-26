from __future__ import annotations

import argparse

from src.data.registry import SUPPORTED_DATASETS
from src.experiments.baseline import run_baseline_experiment
from src.experiments.compare import run_benchmark_comparison
from src.experiments.federated import run_federated_experiment
from src.fl.strategies import FederatedConfig, SUPPORTED_STRATEGIES


def _validate_dataset(name: str) -> str:
    if name not in SUPPORTED_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_DATASETS))
        raise SystemExit(
            f"Dataset '{name}' is not implemented yet. Supported datasets: {supported}"
        )
    return name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run centralized or federated healthcare experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline_parser = subparsers.add_parser("baseline", help="Run centralized training.")
    baseline_parser.add_argument("--dataset", default="heart")
    baseline_parser.add_argument("--epochs", type=int, default=80)
    baseline_parser.add_argument("--batch-size", type=int, default=32)
    baseline_parser.add_argument("--learning-rate", type=float, default=0.05)
    baseline_parser.add_argument("--l2-reg", type=float, default=0.001)
    baseline_parser.add_argument("--seed", type=int, default=42)

    federated_parser = subparsers.add_parser(
        "federated", help="Run a FedAvg or FedProx federated simulation."
    )
    federated_parser.add_argument("--dataset", default="heart")
    federated_parser.add_argument("--rounds", type=int, default=25)
    federated_parser.add_argument("--local-epochs", type=int, default=5)
    federated_parser.add_argument("--batch-size", type=int, default=32)
    federated_parser.add_argument("--learning-rate", type=float, default=0.05)
    federated_parser.add_argument("--l2-reg", type=float, default=0.001)
    federated_parser.add_argument("--num-clients", type=int, default=5)
    federated_parser.add_argument("--fraction-fit", type=float, default=1.0)
    federated_parser.add_argument("--partition", choices=("iid", "dirichlet"), default="iid")
    federated_parser.add_argument("--alpha", type=float, default=0.5)
    federated_parser.add_argument("--strategy", choices=tuple(sorted(SUPPORTED_STRATEGIES)), default="fedavg")
    federated_parser.add_argument("--proximal-mu", type=float, default=0.0)
    federated_parser.add_argument("--seed", type=int, default=42)

    subparsers.add_parser(
        "compare", help="Summarize existing baseline and federated experiment outputs."
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "compare":
        payload = run_benchmark_comparison()
        print(f"Comparison summary complete. Included {payload['run_count']} runs.")
        return

    dataset = _validate_dataset(args.dataset)

    if args.command == "baseline":
        payload = run_baseline_experiment(
            dataset_name=dataset,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            l2_reg=args.l2_reg,
        )
        print("Baseline experiment complete.")
        print(f"Dataset: {payload['dataset']}")
        print(f"Test accuracy: {payload['test_metrics']['accuracy']:.4f}")
        print(f"Test F1: {payload['test_metrics']['f1']:.4f}")
        return

    if args.command == "federated":
        config = FederatedConfig(
            num_rounds=args.rounds,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            l2_reg=args.l2_reg,
            num_clients=args.num_clients,
            fraction_fit=args.fraction_fit,
            partition_mode=args.partition,
            dirichlet_alpha=args.alpha,
            strategy=args.strategy,
            proximal_mu=args.proximal_mu,
            seed=args.seed,
        )
        payload = run_federated_experiment(dataset_name=dataset, config=config)
        print("Federated experiment complete.")
        print(f"Dataset: {payload['dataset']}")
        print(f"Strategy: {payload['strategy']}")
        print(f"Final test accuracy: {payload['final_test_metrics']['accuracy']:.4f}")
        print(f"Final test F1: {payload['final_test_metrics']['f1']:.4f}")
        return


if __name__ == "__main__":
    main()
