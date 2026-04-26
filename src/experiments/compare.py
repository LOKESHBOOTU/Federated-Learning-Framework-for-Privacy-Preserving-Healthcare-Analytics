from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from src.core.artifacts import plot_bar_chart, write_json, write_text
from src.core.paths import METRICS_DIR, PLOTS_DIR, ensure_project_dirs


METRIC_PATTERNS = (
    "*_baseline_metrics.json",
    "*_fedavg_metrics.json",
    "*_fedprox_metrics.json",
)
STRATEGY_ORDER = {
    "baseline": 0,
    "fedavg": 1,
    "fedprox": 2,
}


def _discover_metric_files() -> list[Path]:
    files: set[Path] = set()
    for pattern in METRIC_PATTERNS:
        files.update(METRICS_DIR.glob(pattern))
    return sorted(files)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_summary_row(payload: dict, file_name: str) -> dict:
    strategy = payload.get("strategy", "baseline")
    dataset = payload["dataset"]
    mode = payload.get("mode", "baseline")

    if mode == "baseline":
        metrics = payload["test_metrics"]
        client_count = None
        partition_mode = None
        seed = payload["hyperparameters"]["seed"]
    else:
        metrics = payload["final_test_metrics"]
        config = payload["config"]
        client_count = config["num_clients"]
        partition_mode = config["partition_mode"]
        seed = config["seed"]

    return {
        "dataset": dataset,
        "mode": mode,
        "strategy": strategy,
        "artifact": file_name,
        "test_accuracy": metrics["accuracy"],
        "test_f1": metrics["f1"],
        "test_roc_auc": metrics["roc_auc"],
        "test_loss": metrics["loss"],
        "client_count": client_count,
        "partition_mode": partition_mode,
        "seed": seed,
    }


def _render_markdown(rows: list[dict]) -> str:
    header = (
        "| Dataset | Mode | Strategy | Test Accuracy | Test F1 | ROC-AUC | Loss | "
        "Clients | Partition | Seed |\n"
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |"
    )
    lines = [header]

    for row in rows:
        client_count = "" if row["client_count"] is None else str(row["client_count"])
        partition_mode = "" if row["partition_mode"] is None else str(row["partition_mode"])
        lines.append(
            "| "
            + f"{row['dataset']} | {row['mode']} | {row['strategy']} | "
            + f"{row['test_accuracy']:.4f} | {row['test_f1']:.4f} | "
            + f"{row['test_roc_auc']:.4f} | {row['test_loss']:.4f} | "
            + f"{client_count} | {partition_mode} | {row['seed']} |"
        )

    return "\n".join(lines) + "\n"


def run_benchmark_comparison() -> dict:
    ensure_project_dirs()
    metric_files = _discover_metric_files()
    if not metric_files:
        raise FileNotFoundError("No baseline, fedavg, or fedprox metrics files were found.")

    rows = [
        _build_summary_row(_load_json(metric_file), metric_file.name)
        for metric_file in metric_files
    ]
    rows.sort(key=lambda row: (row["dataset"], STRATEGY_ORDER.get(row["strategy"], 99)))

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_count": len(rows),
        "runs": rows,
    }
    write_json(METRICS_DIR / "benchmark_summary.json", payload)
    write_text(METRICS_DIR / "benchmark_summary.md", _render_markdown(rows))

    plot_bar_chart(
        labels=[f"{row['dataset']}-{row['strategy']}" for row in rows],
        values=[row["test_accuracy"] for row in rows],
        title="Final Test Accuracy Across Available Runs",
        ylabel="Accuracy",
        output_path=PLOTS_DIR / "benchmark_final_test_accuracy.png",
    )
    return payload

