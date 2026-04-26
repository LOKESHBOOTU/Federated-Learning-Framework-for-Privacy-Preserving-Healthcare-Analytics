from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.core.paths import ensure_project_dirs


def write_json(path: Path, payload: dict) -> None:
    ensure_project_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_text(path: Path, content: str) -> None:
    ensure_project_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(content)


def plot_history(
    rounds: Iterable[int],
    values: Iterable[float],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    ensure_project_dirs()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4.5))
    plt.plot(list(rounds), list(values), marker="o")
    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_epoch_history(
    values: Iterable[float],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    materialized_values = list(values)
    rounds = range(1, len(materialized_values) + 1)
    plot_history(rounds, materialized_values, title, ylabel, output_path)


def plot_bar_chart(
    labels: Iterable[str],
    values: Iterable[float],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    ensure_project_dirs()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    materialized_labels = list(labels)
    materialized_values = list(values)

    plt.figure(figsize=(max(8, len(materialized_labels) * 1.6), 4.8))
    plt.bar(materialized_labels, materialized_values)
    plt.title(title)
    plt.xlabel("Run")
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
