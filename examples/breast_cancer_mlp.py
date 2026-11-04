#!/usr/bin/env python3
"""Use AHCO-Rust to optimize a small MLP on scikit-learn's breast-cancer data."""

from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ahco_rust import run_ahco3


HIDDEN_UNITS = 3
INPUT_SEED = 2025
OBJECTIVE_NAMES = ("binary_cross_entropy", "brier_score", "roc_auc")


def build_parameter_bounds(input_units: int) -> tuple[np.ndarray, np.ndarray]:
    """Set Xavier bounds for weights and narrower bounds for biases."""
    hidden_weight_limit = np.sqrt(6.0 / (input_units + HIDDEN_UNITS))
    output_weight_limit = np.sqrt(6.0 / (HIDDEN_UNITS + 1))

    lower = np.concatenate(
        (
            np.full(input_units * HIDDEN_UNITS, -hidden_weight_limit),
            np.full(HIDDEN_UNITS, -0.5),
            np.full(HIDDEN_UNITS, -output_weight_limit),
            np.array([-0.5]),
        )
    )
    upper = -lower
    return lower, upper


def predict_logits(
    normalized_point: list[float],
    features: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Decode AHCO's unit-cube point and evaluate the 30-3-1 MLP."""
    parameters = lower + np.asarray(normalized_point) * (upper - lower)
    input_units = features.shape[1]
    split_1 = input_units * HIDDEN_UNITS
    split_2 = split_1 + HIDDEN_UNITS
    split_3 = split_2 + HIDDEN_UNITS

    hidden_weights = parameters[:split_1].reshape(input_units, HIDDEN_UNITS)
    hidden_bias = parameters[split_1:split_2]
    output_weights = parameters[split_2:split_3]
    output_bias = parameters[split_3]
    hidden = np.tanh(features @ hidden_weights + hidden_bias)
    return hidden @ output_weights + output_bias


def binary_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))


def brier_score(logits: np.ndarray, labels: np.ndarray) -> float:
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    return float(np.mean((probabilities - labels) ** 2))


def make_objectives(
    features: np.ndarray,
    labels: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, Callable[[list[float]], float]]:
    def logits_for(point: list[float]) -> np.ndarray:
        return predict_logits(point, features, lower, upper)

    return {
        "binary_cross_entropy": lambda point: -binary_cross_entropy(
            logits_for(point), labels
        ),
        "brier_score": lambda point: -brier_score(logits_for(point), labels),
        "roc_auc": lambda point: float(
            roc_auc_score(labels, logits_for(point))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=200, help="calls per objective")
    parser.add_argument("--seed", type=int, default=7, help="base AHCO seed")
    args = parser.parse_args()
    if args.budget < 1:
        parser.error("--budget must be at least 1")

    dataset = load_breast_cancer()
    train_x, test_x, train_y, test_y = train_test_split(
        dataset.data,
        dataset.target.astype(np.float64),
        test_size=0.25,
        random_state=INPUT_SEED,
        stratify=dataset.target,
    )
    scaler = StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    lower, upper = build_parameter_bounds(train_x.shape[1])
    dimension = len(lower)
    objectives = make_objectives(train_x, train_y, lower, upper)

    print(
        f"Dataset: Wisconsin Diagnostic Breast Cancer "
        f"({len(dataset.data)} rows, {train_x.shape[1]} features)"
    )
    print(
        f"MLP: {train_x.shape[1]}-{HIDDEN_UNITS}-1, "
        f"{dimension} jointly optimized parameters"
    )
    print(
        f"Split: {len(train_y)} train / {len(test_y)} test; "
        f"scaler fitted on training data only"
    )
    print(f"AHCO calls: {args.budget} per objective\n")
    print(
        "objective                 seed  calls  train objective  "
        "test BCE  test Brier  test ROC-AUC  test balanced accuracy"
    )

    for objective_index, name in enumerate(OBJECTIVE_NAMES):
        objective = objectives[name]
        result = run_ahco3(
            objective,
            d=dimension,
            budget=args.budget,
            seed=args.seed + objective_index,
        )

        test_logits = predict_logits(result["best_x"], test_x, lower, upper)
        test_probabilities = 1.0 / (
            1.0 + np.exp(-np.clip(test_logits, -40.0, 40.0))
        )
        test_predictions = (test_probabilities >= 0.5).astype(np.int64)
        test_metrics = (
            binary_cross_entropy(test_logits, test_y),
            brier_score(test_logits, test_y),
            roc_auc_score(test_y, test_logits),
            balanced_accuracy_score(test_y, test_predictions),
        )
        print(
            f"{name:<25} {args.seed + objective_index:>4} "
            f"{result['evaluations']:>6} {result['best_score']:>16.6f} "
            f"{test_metrics[0]:>9.4f} {test_metrics[1]:>11.4f} "
            f"{test_metrics[2]:>13.4f} {test_metrics[3]:>23.4f}"
        )


if __name__ == "__main__":
    main()
