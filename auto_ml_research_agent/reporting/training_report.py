"""
Post-training report and visualization utilities.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _safe_loss(metric_name: str, score: float) -> float:
    if metric_name in {"accuracy", "f1"}:
        return max(0.0, 1.0 - score)
    if metric_name == "neg_rmse":
        return abs(score)
    return abs(score)


def _plot_experiment_curves(history: List[Dict[str, Any]], output_dir: Path) -> None:
    if not history:
        return
    iterations = [int(h.get("iteration", i + 1)) for i, h in enumerate(history)]
    scores = [float(h.get("score", 0.0)) for h in history]
    metric_names = [str(h.get("metric_name", "")) for h in history]
    losses = [_safe_loss(m, s) for m, s in zip(metric_names, scores)]
    best_model_names = [h.get("config", {}).get("model_name", "unknown") for h in history]

    best_so_far: List[float] = []
    current_best = None
    for s in scores:
        current_best = s if current_best is None else max(current_best, s)
        best_so_far.append(current_best)

    ymin = min(scores + best_so_far)
    ymax = max(scores + best_so_far)
    if abs(ymax - ymin) < 1e-9:
        pad = max(1e-3, abs(ymax) * 0.01)
    else:
        pad = (ymax - ymin) * 0.12

    plt.figure(figsize=(9, 4))
    plt.plot(iterations, scores, marker="o", linewidth=1.8, label="Per-iteration best")
    plt.plot(iterations, best_so_far, linestyle="--", linewidth=2.0, label="Best-so-far")
    plt.title("Score by Iteration")
    plt.xlabel("Epoch/Iteration")
    plt.ylabel("Score")
    plt.ylim(ymin - pad, ymax + pad)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "score_by_epoch.png")
    plt.close()

    lmin = min(losses)
    lmax = max(losses)
    if abs(lmax - lmin) < 1e-9:
        lpad = max(1e-3, abs(lmax) * 0.01)
    else:
        lpad = (lmax - lmin) * 0.12

    plt.figure(figsize=(9, 4))
    plt.plot(iterations, losses, marker="o", color="tab:red", linewidth=1.8)
    plt.title("Loss by Iteration")
    plt.xlabel("Epoch/Iteration")
    plt.ylabel("Loss")
    plt.ylim(lmin - lpad, lmax + lpad)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_by_epoch.png")
    plt.close()

    model_counts: Dict[str, int] = {}
    for h in history:
        candidate_models = h.get("candidate_models")
        if isinstance(candidate_models, list) and candidate_models:
            for m in candidate_models:
                model_counts[str(m)] = model_counts.get(str(m), 0) + 1
        else:
            m = h.get("config", {}).get("model_name", "unknown")
            model_counts[str(m)] = model_counts.get(str(m), 0) + 1
    # Keep deterministic ordering: most-used first, then name
    ordered = sorted(model_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    labels = [k for k, _ in ordered]
    values = [v for _, v in ordered]
    plt.figure(figsize=(10, 4.5))
    bars = plt.bar(labels, values)
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), str(v), ha="center", va="bottom", fontsize=8)
    plt.title("Experiments by Model Type (All Tried Variants)")
    plt.xlabel("Model")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "experiments_by_model.png")
    plt.close()

    # Optional: show best-model wins per iteration in report json context if needed later.
    _ = best_model_names


def generate_training_report(
    pipeline,
    df: pd.DataFrame,
    target_column: str,
    task: str,
    experiment_history: List[Dict[str, Any]],
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    _plot_experiment_curves(experiment_history, output)

    X = df.drop(columns=[target_column])
    y = df[target_column]
    stratify = y if task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    report: Dict[str, Any] = {
        "task": task,
        "target_column": target_column,
        "n_samples": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    if task == "classification":
        acc = float(accuracy_score(y_test, y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred)
        report.update(
            {
                "accuracy": acc,
                "precision_weighted": float(precision),
                "recall_weighted": float(recall),
                "f1_weighted": float(f1),
                "classification_report": classification_report(y_test, y_pred, zero_division=0),
            }
        )

        classes = [str(c) for c in np.unique(pd.concat([pd.Series(y_test), pd.Series(y_pred)]))]
        plt.figure(figsize=(7, 6))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, ha="right")
        plt.yticks(tick_marks, classes)
        thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black"
                )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(output / "confusion_matrix.png")
        plt.close()
    else:
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        report.update({"rmse": rmse, "mae": mae, "r2": r2})

    with open(output / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(output / "training_report.txt", "w", encoding="utf-8") as f:
        for k, v in report.items():
            f.write(f"{k}: {v}\n")

    return report
