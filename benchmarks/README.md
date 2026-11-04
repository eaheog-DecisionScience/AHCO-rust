# Reproducible benchmarks

This suite compares the current Rust-backed AHCO implementation with uniform
random search, Optuna TPE, and Nevergrad's OnePlusOne evolutionary strategy
under the same sequential
objective-call budget. The default experiment uses 97 dimensions, 40 calls
per run, and seeds 0 through 19. It covers dense Sphere, 10-active-coordinate
sparse Sphere, ill-conditioned Ellipsoid, Rosenbrock, and a 97-parameter
neural network on scikit-learn's diabetes dataset.

The four analytic objectives are deterministic, bounded, and have known
optimum score 0. The neural-network task has a fixed, seeded 75/25 split.
Feature and target standardization is fit only on the training partition.
Optimizers maximize negative training MSE; held-out MSE is measured once for
the selected best training point and is never used to guide search.

## Run

Use Python 3.11 and a Rust toolchain compatible with the crate. From the
repository root:

```powershell
python -m pip install -r benchmarks\requirements.txt
python -m pip install -e .
python benchmarks\run_benchmarks.py --dimension 97 --budget 40 --repetitions 20 --output benchmarks\results\main-97d-40eval-20seeds-final
```

Dimension 97 is needed to include the neural-network task; the synthetic
tasks in this run are then also 97-dimensional. To run all analytic tasks at
100 dimensions, omit the diabetes task:

```powershell
python benchmarks\run_benchmarks.py --dimension 100 --budget 40 --repetitions 20 --tasks sphere sparse-sphere ellipsoid rosenbrock --output benchmarks\results\synthetic-100d-40eval
```

For a quick smoke test, reduce `--repetitions` to 2 and use a separate output
directory. Existing output files are not overwritten unless `--overwrite` is
passed.

## Outputs and analysis

- `runs.csv`: one row per task, method, and seed, including best objective,
  exact call count, elapsed time, and (for the diabetes task) held-out MSE.
- `curves.csv`: raw objective score and best-so-far score after every
  evaluation.
- `points.csv`: every evaluated point in JSON-array form, keyed by task,
  method, seed, and evaluation number.
- `summary.csv`: mean, standard deviation, median, interquartile range, and
  bootstrap 95% confidence interval for the median, plus held-out MSE
  summaries when applicable.
- `comparisons.csv`: unpaired Mann–Whitney U comparisons between every method
  pair for training objective and held-out MSE where applicable, with
  independent median-difference confidence intervals and Holm correction.
- `metadata.json`: command, seed list, task settings, package and platform
  versions (including Rust), statistical settings, and SHA-256 hashes of the
  benchmark and optimizer source.

All methods receive the same number of sequential calls. Random search uses
NumPy's `default_rng(seed)`, TPE uses a seeded Optuna `TPESampler`, and
OnePlusOne's parameterization uses mutation sigma 0.25 and is seeded before
optimizer construction. Timings are descriptive only; score comparisons are
based on equal objective-call budgets. The seed values are reused across
methods, but each method's random stream is independent; statistical tests
therefore treat the samples as unpaired. The output files are raw evidence:
do not report only the best seed.

This is a fixed benchmark configuration, not proof that AHCO is universally
superior. The results apply to these tasks, objective definitions, budgets,
and package versions. The diabetes experiment is a small real-data
black-box weight-search example, not a clinical claim or a substitute for
larger external validation.
