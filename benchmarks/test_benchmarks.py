import numpy as np
import pytest

from benchmarks.run_benchmarks import (
    ALGORITHMS,
    build_task,
    bootstrap_median_difference_ci,
    run_algorithm,
)


@pytest.mark.parametrize(
    ("task", "dimension", "optimum_point"),
    [
        ("sphere", 20, 0.7),
        ("sparse-sphere", 20, 0.7),
        ("ellipsoid", 20, 0.3),
        ("rosenbrock", 20, 0.75),
    ],
)
def test_analytic_objectives_have_documented_optima(task, dimension, optimum_point):
    objective, details = build_task(task, dimension)
    assert objective(np.full(dimension, optimum_point)) == pytest.approx(0.0)
    assert details["optimum"] == 0.0


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_seeded_runs_are_reproducible_and_use_exact_budget(algorithm):
    objective, _ = build_task("sphere", 12)
    first = run_algorithm(algorithm, objective, 12, 16, 123)
    second = run_algorithm(algorithm, objective, 12, 16, 123)

    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[3], second[3])
    np.testing.assert_array_equal(first[4], second[4])
    np.testing.assert_array_equal(first[5], second[5])
    assert len(first[1]) == 16
    assert np.all(np.diff(first[1]) >= 0.0)
    assert np.all((0.0 <= first[3]) & (first[3] <= 1.0))
    for point, score in zip(first[5], first[4]):
        assert objective(point) == pytest.approx(score)


def test_one_plus_one_uses_distinct_optimizer_seeds():
    objective, _ = build_task("sphere", 12)
    first = run_algorithm("Nevergrad-OnePlusOne", objective, 12, 16, 1)
    second = run_algorithm("Nevergrad-OnePlusOne", objective, 12, 16, 2)

    assert first[0] != second[0]


def test_independent_bootstrap_interval_reflects_median_difference():
    values_a = np.full(20, 2.0)
    values_b = np.full(20, 1.0)

    assert bootstrap_median_difference_ci(values_a, values_b, 7) == (1.0, 1.0)


def test_diabetes_task_uses_fixed_split_and_reports_heldout_loss():
    objective, details = build_task("diabetes-mlp", 97)
    point = np.full(97, 0.5)

    assert details["split"] == "75/25 shuffled; random_state=2025"
    assert set(details["train_indices"]).isdisjoint(details["test_indices"])
    assert sorted(details["train_indices"] + details["test_indices"]) == list(
        range(442)
    )
    assert objective(point) == pytest.approx(objective(point))
    assert np.isfinite(details["heldout_loss"](point))
