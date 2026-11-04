#!/usr/bin/env python3
"""Reproducible, equal-budget AHCO benchmark suite."""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import nevergrad as ng
import numpy as np
import optuna
import scipy
import sklearn
from optuna.samplers import TPESampler
from scipy.stats import mannwhitneyu
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import ahco_rust
from ahco_rust import ahco as ahco_extension


ALGORITHMS = ("AHCO", "Random", "Optuna-TPE", "Nevergrad-OnePlusOne")
TASKS = ("sphere", "sparse-sphere", "ellipsoid", "rosenbrock", "diabetes-mlp")
DIABETES_SPLIT_SEED = 2025
BOOTSTRAP_SEED = 20261104


def build_task(name: str, dimension: int) -> tuple[Callable[[np.ndarray], float], dict]:
    if name == "sphere":
        shift = np.full(dimension, 0.7)

        def objective(point: np.ndarray) -> float:
            return -float(np.mean((point - shift) ** 2))

        return objective, {
            "optimum": 0.0,
            "active_dimensions": dimension,
            "formulation": "negative mean squared distance from x_j=0.7",
        }

    if name == "sparse-sphere":
        active = np.arange(0, dimension, max(1, dimension // 10))[:10]
        shift = np.full(len(active), 0.7)

        def objective(point: np.ndarray) -> float:
            return -float(np.mean((point[active] - shift) ** 2))

        return objective, {
            "optimum": 0.0,
            "active_dimensions": len(active),
            "active_indices": active.tolist(),
            "formulation": "negative mean squared error on active coordinates; target 0.7",
        }

    if name == "ellipsoid":
        shift = np.full(dimension, 0.3)
        weights = np.logspace(0.0, 4.0, dimension)
        weights /= weights.mean()

        def objective(point: np.ndarray) -> float:
            return -float(np.mean(weights * (point - shift) ** 2))

        return objective, {
            "optimum": 0.0,
            "active_dimensions": dimension,
            "formulation": (
                "negative mean weighted squared error from x_j=0.3; "
                "weights are mean-normalized logspace(1, 1e4)"
            ),
        }

    if name == "rosenbrock":
        def objective(point: np.ndarray) -> float:
            x = 4.0 * point - 2.0
            loss = 100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2
            return -float(np.mean(loss))

        return objective, {
            "optimum": 0.0,
            "active_dimensions": dimension,
            "formulation": (
                "negative mean Rosenbrock loss after mapping x in [0,1] "
                "to [-2,2]; global minimizer x_j=0.75"
            ),
        }

    if name == "diabetes-mlp":
        data = load_diabetes()
        train_indices, test_indices = train_test_split(
            np.arange(len(data.target)),
            test_size=0.25,
            random_state=DIABETES_SPLIT_SEED,
        )
        train_x, test_x = data.data[train_indices], data.data[test_indices]
        train_y, test_y = data.target[train_indices], data.target[test_indices]
        scaler_x = StandardScaler().fit(train_x)
        scaler_y = StandardScaler().fit(train_y.reshape(-1, 1))
        train_x = scaler_x.transform(train_x)
        test_x = scaler_x.transform(test_x)
        train_y = scaler_y.transform(train_y.reshape(-1, 1)).ravel()
        test_y = scaler_y.transform(test_y.reshape(-1, 1)).ravel()
        input_dim, hidden_dim = train_x.shape[1], 8
        expected_dimension = input_dim * hidden_dim + hidden_dim + hidden_dim + 1
        if dimension != expected_dimension:
            raise ValueError(
                f"diabetes-mlp requires dimension={expected_dimension}, got {dimension}"
            )

        def predict(point: np.ndarray, features: np.ndarray) -> np.ndarray:
            weights = 6.0 * point - 3.0
            offset = 0
            w1 = weights[offset : offset + input_dim * hidden_dim].reshape(
                input_dim, hidden_dim
            )
            offset += input_dim * hidden_dim
            b1 = weights[offset : offset + hidden_dim]
            offset += hidden_dim
            w2 = weights[offset : offset + hidden_dim]
            b2 = weights[-1]
            hidden = np.tanh(features @ w1 + b1)
            return hidden @ w2 + b2

        def objective(point: np.ndarray) -> float:
            predictions = predict(point, train_x)
            return -float(np.mean((predictions - train_y) ** 2))

        def heldout_loss(point: np.ndarray) -> float:
            predictions = predict(point, test_x)
            return float(np.mean((predictions - test_y) ** 2))

        return objective, {
            "optimum": None,
            "active_dimensions": dimension,
            "dataset": "scikit-learn diabetes (442 rows, 10 features)",
            "split": f"75/25 shuffled; random_state={DIABETES_SPLIT_SEED}",
            "dataset_sha256": hashlib.sha256(
                data.data.tobytes() + data.target.tobytes()
            ).hexdigest(),
            "train_indices": train_indices.tolist(),
            "test_indices": test_indices.tolist(),
            "model": "10-input, 8-hidden-unit tanh MLP, scalar linear output",
            "parameter_bounds": "each of 97 weights/biases is mapped from [0,1] to [-3,3]",
            "objective": "negative training MSE after train-only feature and target scaling",
            "heldout_metric": "test MSE after training-only target scaling",
            "heldout_loss": heldout_loss,
        }

    raise ValueError(f"unknown task: {name}")


def run_algorithm(
    algorithm: str,
    objective: Callable[[np.ndarray], float],
    dimension: int,
    budget: int,
    seed: int,
) -> tuple[float, list[float], int, np.ndarray, list[float], list[np.ndarray]]:
    scores: list[float] = []
    points: list[np.ndarray] = []
    started_ns = time.perf_counter_ns()

    def evaluate(point: np.ndarray) -> float:
        score = float(objective(np.asarray(point, dtype=np.float64)))
        if not np.isfinite(score):
            raise ValueError(f"{algorithm} produced a non-finite score")
        scores.append(score)
        points.append(np.asarray(point, dtype=np.float64).copy())
        return score

    def completed(
        best: float, history: list[float], best_point: np.ndarray
    ) -> tuple[float, list[float], int, np.ndarray, list[float], list[np.ndarray]]:
        if len(scores) != budget or len(history) != budget:
            raise RuntimeError(
                f"{algorithm} used {len(scores)} objective calls; expected {budget}"
            )
        return (
            best,
            history,
            time.perf_counter_ns() - started_ns,
            np.asarray(best_point, dtype=np.float64),
            scores.copy(),
            points.copy(),
        )

    if algorithm == "AHCO":
        result = ahco_rust.run_ahco3(
            lambda point: evaluate(np.asarray(point)),
            dimension,
            budget,
            seed,
        )
        if int(result["evaluations"]) != budget or len(result["history"]) != budget:
            raise RuntimeError("AHCO did not consume the exact requested budget")
        best_history = [float(value) for value in result["history"]]
        if not np.all(np.isfinite(best_history)):
            raise RuntimeError("AHCO returned a non-finite best-so-far history")
        return completed(
            float(result["best_score"]),
            best_history,
            np.asarray(result["best_x"], dtype=np.float64),
        )

    if algorithm == "Random":
        rng = np.random.default_rng(seed)
        best = -np.inf
        best_point = np.empty(dimension)
        best_history = []
        for _ in range(budget):
            point = rng.random(dimension)
            score = evaluate(point)
            if score > best:
                best, best_point = score, point.copy()
            best_history.append(best)
        return completed(best, best_history, best_point)

    if algorithm == "Optuna-TPE":
        sampler = TPESampler(seed=seed, n_startup_trials=min(10, budget))
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def optuna_objective(trial: optuna.Trial) -> float:
            point = np.asarray(
                [
                    trial.suggest_float(f"x_{index}", 0.0, 1.0)
                    for index in range(dimension)
                ]
            )
            score = evaluate(point)
            trial.set_user_attr("point", point.tolist())
            return score

        study.optimize(
            optuna_objective,
            n_trials=budget,
            n_jobs=1,
            show_progress_bar=False,
        )
        best = -np.inf
        best_history = []
        for trial in study.trials:
            if trial.value is None:
                raise RuntimeError("Optuna produced a trial without a score")
            best = max(best, float(trial.value))
            best_history.append(best)
        best_trial = study.best_trial
        return completed(
            best,
            best_history,
            np.asarray(best_trial.user_attrs["point"], dtype=np.float64),
        )

    if algorithm == "Nevergrad-OnePlusOne":
        parametrization = (
            ng.p.Array(shape=(dimension,))
            .set_mutation(sigma=0.25)
            .set_bounds(lower=0.0, upper=1.0)
        )
        parametrization.random_state.seed(seed)
        optimizer = ng.optimizers.OnePlusOne(
            parametrization=parametrization, budget=budget, num_workers=1
        )
        best = -np.inf
        best_point = np.empty(dimension)
        best_history = []
        for _ in range(budget):
            candidate = optimizer.ask()
            score = evaluate(np.asarray(candidate.value))
            optimizer.tell(candidate, -score)
            if score > best:
                best, best_point = score, np.asarray(candidate.value).copy()
            best_history.append(best)
        return completed(best, best_history, best_point)

    raise ValueError(f"unknown algorithm: {algorithm}")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_median_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(values), size=(10_000, len(values)))
    medians = np.median(values[sample_indices], axis=1)
    return tuple(float(value) for value in np.quantile(medians, [0.025, 0.975]))


def bootstrap_median_difference_ci(
    values_a: np.ndarray, values_b: np.ndarray, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sample_indices_a = rng.integers(
        0, len(values_a), size=(10_000, len(values_a))
    )
    sample_indices_b = rng.integers(
        0, len(values_b), size=(10_000, len(values_b))
    )
    differences = (
        np.median(values_a[sample_indices_a], axis=1)
        - np.median(values_b[sample_indices_b], axis=1)
    )
    return tuple(float(value) for value in np.quantile(differences, [0.025, 0.975]))


def summarize(
    results: list[dict], repetitions: int, tasks: list[str]
) -> tuple[list[dict], list[dict]]:
    summary_rows = []
    comparison_rows = []
    for task in tasks:
        task_results = [row for row in results if row["task"] == task]
        for algorithm in ALGORITHMS:
            method_rows = sorted(
                [row for row in task_results if row["algorithm"] == algorithm],
                key=lambda row: row["seed"],
            )
            values = np.asarray([row["best_score"] for row in method_rows])
            if len(values) != repetitions:
                raise RuntimeError(f"incomplete runs for {task}/{algorithm}")
            ci_low, ci_high = bootstrap_median_ci(
                values, BOOTSTRAP_SEED + tasks.index(task) * len(ALGORITHMS)
                + ALGORITHMS.index(algorithm)
            )
            row = {
                "task": task,
                "algorithm": algorithm,
                "n": len(values),
                "mean_best_score": float(np.mean(values)),
                "sd_best_score": float(np.std(values, ddof=1)),
                "median_best_score": float(np.median(values)),
                "q1_best_score": float(np.quantile(values, 0.25)),
                "q3_best_score": float(np.quantile(values, 0.75)),
                "median_95ci_low": ci_low,
                "median_95ci_high": ci_high,
            }
            heldout_values = [
                float(item["heldout_loss"])
                for item in method_rows
                if item["heldout_loss"] != ""
            ]
            if heldout_values:
                heldout = np.asarray(heldout_values)
                row.update(
                    heldout_median_mse=float(np.median(heldout)),
                    heldout_q1_mse=float(np.quantile(heldout, 0.25)),
                    heldout_q3_mse=float(np.quantile(heldout, 0.75)),
                    heldout_median_95ci_low=bootstrap_median_ci(
                        heldout, BOOTSTRAP_SEED + 1000
                    )[0],
                    heldout_median_95ci_high=bootstrap_median_ci(
                        heldout, BOOTSTRAP_SEED + 1000
                    )[1],
                )
            else:
                row.update(
                    heldout_median_mse="",
                    heldout_q1_mse="",
                    heldout_q3_mse="",
                    heldout_median_95ci_low="",
                    heldout_median_95ci_high="",
                )
            summary_rows.append(row)

        def append_comparison(
            method_a: str,
            method_b: str,
            metric: str,
            values_a: np.ndarray,
            values_b: np.ndarray,
            higher_is_better: bool,
        ) -> None:
            if len(values_a) != repetitions or len(values_b) != repetitions:
                raise RuntimeError(
                    f"unpaired runs for {task}/{method_a}/{method_b}/{metric}"
                )
            p_value = (
                1.0
                if np.array_equal(np.sort(values_a), np.sort(values_b))
                else float(
                    mannwhitneyu(
                        values_a,
                        values_b,
                        alternative="two-sided",
                        method="auto",
                    ).pvalue
                )
            )
            direction = "higher" if higher_is_better else "lower"
            ci_low, ci_high = bootstrap_median_difference_ci(
                values_a,
                values_b,
                BOOTSTRAP_SEED
                + len(comparison_rows)
                + 100 * tasks.index(task),
            )
            comparison_rows.append(
                {
                    "task": task,
                    "method_a": method_a,
                    "method_b": method_b,
                    "metric": metric,
                    "comparison": f"{method_a} - {method_b} ({direction} is better)",
                    "n_a": len(values_a),
                    "n_b": len(values_b),
                    "median_difference": float(
                        np.median(values_a) - np.median(values_b)
                    ),
                    "median_difference_95ci_low": ci_low,
                    "median_difference_95ci_high": ci_high,
                    "mannwhitney_p_unadjusted": p_value,
                }
            )

        rows_by_method = {
            algorithm: {
                int(item["seed"]): item
                for item in task_results
                if item["algorithm"] == algorithm
            }
            for algorithm in ALGORITHMS
        }
        values_by_method = {
            algorithm: np.asarray(
                [item["best_score"] for item in rows_by_method[algorithm].values()]
            )
            for algorithm in ALGORITHMS
        }
        heldout_values_by_method = (
            {
                algorithm: np.asarray(
                    [
                        float(item["heldout_loss"])
                        for item in rows_by_method[algorithm].values()
                    ]
                )
                for algorithm in ALGORITHMS
            }
            if task == "diabetes-mlp"
            else {}
        )
        for index, method_a in enumerate(ALGORITHMS):
            for method_b in ALGORITHMS[index + 1 :]:
                append_comparison(
                    method_a,
                    method_b,
                    "best_objective",
                    values_by_method[method_a],
                    values_by_method[method_b],
                    higher_is_better=True,
                )
                if task == "diabetes-mlp":
                    append_comparison(
                        method_a,
                        method_b,
                        "heldout_mse",
                        heldout_values_by_method[method_a],
                        heldout_values_by_method[method_b],
                        higher_is_better=False,
                    )

    ordered_p = sorted(
        enumerate(comparison_rows),
        key=lambda item: item[1]["mannwhitney_p_unadjusted"],
    )
    previous_adjusted = 0.0
    total_tests = len(ordered_p)
    adjusted = [1.0] * total_tests
    for rank, (original_index, row) in enumerate(ordered_p):
        candidate = min(
            1.0,
            row["mannwhitney_p_unadjusted"] * (total_tests - rank),
        )
        previous_adjusted = max(previous_adjusted, candidate)
        adjusted[original_index] = previous_adjusted
    for index, row in enumerate(comparison_rows):
        row["mannwhitney_p_holm"] = adjusted[index]
    return summary_rows, comparison_rows


def main() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=97)
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASKS,
        default=list(TASKS),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks") / "results",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.dimension < 2 or args.budget < 2 or args.repetitions < 2:
        parser.error("dimension, budget, and repetitions must each be at least 2")
    if "diabetes-mlp" in args.tasks and args.dimension != 97:
        parser.error("use --dimension 97 when including diabetes-mlp")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    output_files = [
        output / "runs.csv",
        output / "curves.csv",
        output / "points.csv",
        output / "summary.csv",
        output / "comparisons.csv",
        output / "metadata.json",
    ]
    if not args.overwrite and any(path.exists() for path in output_files):
        parser.error(
            f"results already exist in {output}; choose another --output or use --overwrite"
        )

    run_rows: list[dict] = []
    curve_rows: list[dict] = []
    point_rows: list[dict] = []
    task_metadata: dict[str, dict] = {}
    seeds = list(range(args.first_seed, args.first_seed + args.repetitions))

    for task in args.tasks:
        objective, details = build_task(task, args.dimension)
        task_metadata[task] = {
            key: value
            for key, value in details.items()
            if key != "heldout_loss"
        }
        heldout_loss = details.get("heldout_loss")
        for seed in seeds:
            for algorithm in ALGORITHMS:
                (
                    best_score,
                    history,
                    elapsed_ns,
                    best_point,
                    objective_scores,
                    evaluation_points,
                ) = run_algorithm(
                    algorithm, objective, args.dimension, args.budget, seed
                )
                if (
                    len(history) != args.budget
                    or len(objective_scores) != args.budget
                    or len(evaluation_points) != args.budget
                    or len(best_point) != args.dimension
                ):
                    raise RuntimeError(
                        f"{task}/{algorithm}/seed={seed} returned invalid run data"
                    )
                run_rows.append(
                    {
                        "task": task,
                        "dimension": args.dimension,
                        "budget": args.budget,
                        "algorithm": algorithm,
                        "seed": seed,
                        "best_score": best_score,
                        "heldout_loss": (
                            float(heldout_loss(best_point))
                            if heldout_loss is not None
                            else ""
                        ),
                        "evaluations": len(history),
                        "runtime_seconds": elapsed_ns / 1_000_000_000,
                    }
                )
                curve_rows.extend(
                    {
                        "task": task,
                        "dimension": args.dimension,
                        "budget": args.budget,
                        "algorithm": algorithm,
                        "seed": seed,
                        "evaluation": evaluation,
                        "objective_score": objective_scores[evaluation - 1],
                        "best_score": score,
                    }
                    for evaluation, score in enumerate(history, start=1)
                )
                point_rows.extend(
                    {
                        "task": task,
                        "dimension": args.dimension,
                        "budget": args.budget,
                        "algorithm": algorithm,
                        "seed": seed,
                        "evaluation": evaluation,
                        "point": json.dumps(
                            point.tolist(), separators=(",", ":"), allow_nan=False
                        ),
                    }
                    for evaluation, point in enumerate(evaluation_points, start=1)
                )
                print(
                    f"{task:16} {algorithm:18} seed={seed:3} "
                    f"best={best_score: .8g} elapsed={elapsed_ns / 1e9:.3f}s",
                    flush=True,
                )

    summary_rows, comparison_rows = summarize(
        run_rows, args.repetitions, args.tasks
    )
    write_csv(
        output_files[0],
        run_rows,
        [
            "task",
            "dimension",
            "budget",
            "algorithm",
            "seed",
            "best_score",
            "heldout_loss",
            "evaluations",
            "runtime_seconds",
        ],
    )
    write_csv(
        output_files[1],
        curve_rows,
        [
            "task",
            "dimension",
            "budget",
            "algorithm",
            "seed",
            "evaluation",
            "objective_score",
            "best_score",
        ],
    )
    write_csv(
        output_files[2],
        point_rows,
        [
            "task",
            "dimension",
            "budget",
            "algorithm",
            "seed",
            "evaluation",
            "point",
        ],
    )
    write_csv(
        output_files[3],
        summary_rows,
        [
            "task",
            "algorithm",
            "n",
            "mean_best_score",
            "sd_best_score",
            "median_best_score",
            "q1_best_score",
            "q3_best_score",
            "median_95ci_low",
            "median_95ci_high",
            "heldout_median_mse",
            "heldout_q1_mse",
            "heldout_q3_mse",
            "heldout_median_95ci_low",
            "heldout_median_95ci_high",
        ],
    )
    write_csv(
        output_files[4],
        comparison_rows,
        [
            "task",
            "method_a",
            "method_b",
            "metric",
            "comparison",
            "n_a",
            "n_b",
            "median_difference",
            "median_difference_95ci_low",
            "median_difference_95ci_high",
            "mannwhitney_p_unadjusted",
            "mannwhitney_p_holm",
        ],
    )
    root = Path(__file__).resolve().parents[1]
    rustc_path = shutil.which("rustc")
    rustc_version = (
        subprocess.run(
            [rustc_path, "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if rustc_path
        else None
    )
    metadata = {
        "command": sys.argv,
        "dimension": args.dimension,
        "budget": args.budget,
        "repetitions": args.repetitions,
        "seeds": seeds,
        "tasks": task_metadata,
        "algorithms": list(ALGORITHMS),
        "optimizer_settings": {
            "AHCO": {"entry_point": "run_ahco3", "kmax": 14, "batch_size": 1},
            "Random": {"generator": "numpy.default_rng", "bounds": [0.0, 1.0]},
            "Optuna-TPE": {
                "sampler": "TPESampler",
                "n_startup_trials": min(10, args.budget),
                "n_jobs": 1,
            },
            "Nevergrad-OnePlusOne": {
                "optimizer": "OnePlusOne",
                "num_workers": 1,
                "bounds": [0.0, 1.0],
                "mutation_sigma": 0.25,
                "seeded_parameterization": True,
            },
        },
        "bootstrap": {
            "statistic": "median",
            "confidence_level": 0.95,
            "resamples": 10_000,
            "seed": BOOTSTRAP_SEED,
        },
        "pairwise_tests": "unpaired two-sided Mann-Whitney U tests; bootstrap confidence intervals for independent median differences; Holm-adjusted across task/method-pair/metric tests",
        "environment": {
            "python": sys.version,
            "rustc": rustc_version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "optuna": optuna.__version__,
            "nevergrad": ng.__version__,
            "ahco_rust": version("ahco-rust"),
            "ahco_rust_module": Path(ahco_rust.__file__).name,
            "ahco_extension_module": Path(ahco_extension.__file__).name,
        },
        "sha256": {
            "ahco.rs": source_hash(root / "ahco.rs"),
            "benchmark_script": source_hash(Path(__file__).resolve()),
            "Cargo.toml": source_hash(root / "Cargo.toml"),
            "pyproject.toml": source_hash(root / "pyproject.toml"),
            "benchmark_requirements": source_hash(
                Path(__file__).resolve().with_name("requirements.txt")
            ),
        },
    }
    output_files[5].write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote benchmark data to {output}")


if __name__ == "__main__":
    main()
