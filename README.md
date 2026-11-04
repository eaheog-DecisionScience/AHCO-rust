# AHCO

AHCO is a Rust implementation of a derivative-free strategy for maximizing
bounded black-box objectives. The Rust API supports real and categorical
parameters and feasibility constraints. Evaluations are sequential: each
objective result informs the next proposal. The optional Python extension
supports categorical parameters, but constraint reporting is currently
available through the Rust API.

## Rust

Add the crate to `Cargo.toml`:

```toml
[dependencies]
ahco = "0.1"
```

Every coordinate is represented in `[0, 1]`. Real coordinates retain
continuous values. For a `Categorical(n)` coordinate, AHCO passes the centre of
one of `n` equal-width cells; decode the category as
`min((x * n) as usize, n - 1)`.

Example with one real parameter, one categorical parameter, and a known
feasibility constraint:

```rust
use ahco::{optimize_eval, AhcoConfig, Evaluation, ParamKind};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = AhcoConfig::new(2, 100, 7)
        .with_kinds(vec![ParamKind::Real, ParamKind::Categorical(3)])
        .with_known_constraint(|x| x[0] - 0.8);

    let result = optimize_eval(config, |x| {
        let category = ((x[1] * 3.0) as usize).min(2);
        let score = -(x[0] - 0.6).powi(2) + if category == 1 { 1.0 } else { 0.0 };
        Ok(Evaluation::from(score))
    })?;

    println!("best score: {}", result.best_score);
    println!("best point: {:?}", result.best_x);
    println!("feasible: {}", result.feasible);
    Ok(())
}
```

Every objective receives a point in `[0, 1]^dimension` and returns a score to
maximize. Return `Err(AhcoError::Objective(...))` to stop on an objective
failure. Non-finite scores are treated as very poor scores. `history` contains
the best feasible score so far after each evaluation. Known constraints return
a non-positive value when satisfied and a positive violation otherwise; they
are checked during candidate generation and used to rank evaluated points.
If the search cannot find a feasible point, `feasible` is false and the result
is the least-violating point found. For constraints that are expensive to
evaluate, use `optimize_eval` and return `Evaluation::new(score, violation)`;
select `ConstraintMode` to choose feasibility-first ranking or a
scale-dependent penalty. The default execution model is strictly sequential.

The crate is dual-licensed under MIT or Apache-2.0.

## Python

Install the extension from this source directory with `pip install .` (a Rust
toolchain is required to build it):

```python
from ahco_rust import run_ahco3

result = run_ahco3(
    lambda x: -sum((value - 0.7) ** 2 for value in x),
    d=4,
    budget=100,
    seed=7,
)

print(result["best_x"], result["best_score"])
```

The Python function accepts the same `f`, `d`, `budget`, `seed`, and optional
`kmax` arguments as `ahco.py`'s `run_ahco3`; it also accepts `levels`, a list
with one category count per dimension (`0` or `1` means real-valued). Its result has the same keys:
`method`, `best_score`, `best_x`, `evaluations`, `history`, and `seconds`.
The Python objective is called once per evaluation; exceptions raised by it
are propagated to the caller.

