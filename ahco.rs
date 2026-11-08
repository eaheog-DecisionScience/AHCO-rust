use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Gamma, Normal};
use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

const BAD_SCORE: f64 = -1.0e100;

// ----------------------------------------------------------------------------------------------
// Public configuration / result types
// ----------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum ParamKind {
    Real,
    /// Unordered categorical with `n >= 2` levels.
    Categorical(usize),
}

/// How a violation reported by the objective is turned into a ranking.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConstraintMode {
    /// Any feasible point beats any infeasible one; infeasible points are ordered by violation.
    FeasibilityFirst,
    /// score = value - weight * violation. Scale dependent, but kept the best result in the
    /// stress test for observed-only constraints (see report).
    Penalty { weight: f64 },
}

/// Side-effect-free constraint. Return <= 0.0 when satisfied, the amount of violation
/// otherwise. NaN is treated as infinitely violated. If a candidate pool contains no feasible
/// point, the least-violating candidate can still be evaluated and is ranked as infeasible.
#[derive(Clone)]
pub struct Constraint(Arc<dyn Fn(&[f64]) -> f64>);

impl Constraint {
    pub fn new<F>(function: F) -> Self
    where
        F: Fn(&[f64]) -> f64 + 'static,
    {
        Self(Arc::new(function))
    }

    fn violation(&self, x: &[f64]) -> f64 {
        let value = (self.0)(x);
        if value.is_nan() {
            f64::INFINITY
        } else {
            value.max(0.0)
        }
    }
}

impl fmt::Debug for Constraint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Constraint(..)")
    }
}

/// Result of one objective call. `violation <= 0` means feasible.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Evaluation {
    pub value: f64,
    pub violation: f64,
}

impl Evaluation {
    pub fn new(value: f64, violation: f64) -> Self {
        Self { value, violation }
    }
}

impl From<f64> for Evaluation {
    fn from(value: f64) -> Self {
        Self {
            value,
            violation: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AhcoConfig {
    pub dimension: usize,
    pub budget: usize,
    pub seed: u64,
    pub max_active_dimensions: usize,
    /// Empty = every dimension is Real. Otherwise must have exactly `dimension` entries.
    pub kinds: Vec<ParamKind>,
    pub constraint_mode: ConstraintMode,
    pub known_constraint: Option<Constraint>,
}

impl AhcoConfig {
    pub fn new(dimension: usize, budget: usize, seed: u64) -> Self {
        Self {
            dimension,
            budget,
            seed,
            max_active_dimensions: 14,
            kinds: Vec::new(),
            constraint_mode: ConstraintMode::Penalty { weight: 5.0 },
            known_constraint: None,
        }
    }

    pub fn with_max_active_dimensions(mut self, max_active_dimensions: usize) -> Self {
        self.max_active_dimensions = max_active_dimensions;
        self
    }

    pub fn with_kinds(mut self, kinds: Vec<ParamKind>) -> Self {
        self.kinds = kinds;
        self
    }

    pub fn with_constraint_mode(mut self, mode: ConstraintMode) -> Self {
        self.constraint_mode = mode;
        self
    }

    pub fn with_known_constraint<F>(mut self, function: F) -> Self
    where
        F: Fn(&[f64]) -> f64 + 'static,
    {
        self.known_constraint = Some(Constraint::new(function));
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OptimizationResult {
    pub method: &'static str,
    /// Raw objective value of `best_x`.
    pub best_score: f64,
    pub best_x: Vec<f64>,
    /// False only if no feasible point was ever found (then `best_x` is the least-bad point).
    pub feasible: bool,
    pub infeasible_evaluations: usize,
    pub evaluations: usize,
    /// Running best *feasible* value (BAD_SCORE until the first feasible point).
    pub history: Vec<f64>,
    pub seconds: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AhcoError {
    InvalidDimension,
    InvalidBudget,
    InvalidMaxActiveDimensions,
    InvalidKinds,
    Objective(String),
}

impl fmt::Display for AhcoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimension => write!(formatter, "dimension must be greater than zero"),
            Self::InvalidBudget => write!(formatter, "budget must be greater than zero"),
            Self::InvalidMaxActiveDimensions => {
                write!(formatter, "max_active_dimensions must be greater than zero")
            }
            Self::InvalidKinds => write!(
                formatter,
                "kinds must be empty or have one entry per dimension, categoricals need >= 2 levels"
            ),
            Self::Objective(message) => write!(formatter, "objective failed: {message}"),
        }
    }
}

impl Error for AhcoError {}

// ----------------------------------------------------------------------------------------------
// Search space helpers
// ----------------------------------------------------------------------------------------------

struct Space {
    /// 0 = real, n >= 2 = categorical with n levels.
    levels: Vec<usize>,
}

impl Space {
    fn new(config: &AhcoConfig) -> Self {
        let levels = if config.kinds.is_empty() {
            vec![0; config.dimension]
        } else {
            config
                .kinds
                .iter()
                .map(|kind| match kind {
                    ParamKind::Real => 0,
                    ParamKind::Categorical(n) => *n,
                })
                .collect()
        };
        Self { levels }
    }

    fn is_cat(&self, feature: usize) -> bool {
        self.levels[feature] > 0
    }

    fn level(&self, feature: usize, value: f64) -> usize {
        let n = self.levels[feature];
        ((value * n as f64) as usize).min(n - 1)
    }

    fn width(&self, feature: usize) -> usize {
        self.levels[feature].max(1)
    }

    fn snap(&self, point: &mut [f64]) {
        for (feature, &n) in self.levels.iter().enumerate() {
            if n > 0 {
                let level = ((point[feature].clamp(0.0, 1.0) * n as f64) as usize).min(n - 1);
                point[feature] = (level as f64 + 0.5) / n as f64;
            }
        }
    }

    fn random_level(&self, feature: usize, rng: &mut ChaCha8Rng) -> f64 {
        let n = self.levels[feature];
        (rng.gen_range(0..n) as f64 + 0.5) / n as f64
    }

    /// Real dims pass through, categorical dims become one-hot columns.
    fn encode(&self, point: &[f64], columns: &[usize]) -> Vec<f64> {
        let mut out = Vec::with_capacity(columns.len());
        for &feature in columns {
            let n = self.levels[feature];
            if n > 0 {
                let level = self.level(feature, point[feature]);
                for candidate in 0..n {
                    out.push(if candidate == level { 1.0 } else { 0.0 });
                }
            } else {
                out.push(point[feature]);
            }
        }
        out
    }
}

struct Ctx<'a> {
    space: &'a Space,
    constraint: Option<&'a Constraint>,
}

// ----------------------------------------------------------------------------------------------
// Archive of observations
// ----------------------------------------------------------------------------------------------

struct Archive {
    points: Vec<Vec<f64>>,
    values: Vec<f64>,
    violations: Vec<f64>,
    /// Effective (constraint-aware) scores; only their ORDER is meaningful.
    scores: Vec<f64>,
    history: Vec<f64>,
    mode: ConstraintMode,
    known_constraint: bool,
}

impl Archive {
    fn new(budget: usize, mode: ConstraintMode, known_constraint: bool) -> Self {
        Self {
            points: Vec::with_capacity(budget),
            values: Vec::with_capacity(budget),
            violations: Vec::with_capacity(budget),
            scores: Vec::with_capacity(budget),
            history: Vec::with_capacity(budget),
            mode,
            known_constraint,
        }
    }

    fn push(&mut self, point: Vec<f64>, evaluation: Evaluation, known_violation: f64) {
        let value = if evaluation.value.is_finite() {
            evaluation.value
        } else {
            BAD_SCORE
        };
        let reported_violation = if evaluation.violation.is_nan() {
            1.0e6
        } else {
            evaluation.violation.max(0.0)
        };
        let violation = reported_violation.max(known_violation);
        let previous = self.history.last().copied().unwrap_or(BAD_SCORE);
        let feasible_value = if violation <= 0.0 { value } else { BAD_SCORE };
        self.history.push(previous.max(feasible_value));
        self.points.push(point);
        self.values.push(value);
        self.violations.push(violation);
        let mode = if self.known_constraint {
            ConstraintMode::FeasibilityFirst
        } else {
            self.mode
        };
        self.scores = effective_scores(&self.values, &self.violations, mode);
    }
}

fn effective_scores(values: &[f64], violations: &[f64], mode: ConstraintMode) -> Vec<f64> {
    match mode {
        ConstraintMode::Penalty { weight } => values
            .iter()
            .zip(violations)
            .map(|(value, violation)| value - weight * violation)
            .collect(),
        ConstraintMode::FeasibilityFirst => {
            let floor = values
                .iter()
                .zip(violations)
                .filter(|(_, violation)| **violation <= 0.0)
                .map(|(value, _)| *value)
                .fold(f64::INFINITY, f64::min);
            let floor = if floor.is_finite() { floor } else { 0.0 };
            values
                .iter()
                .zip(violations)
                .map(|(value, violation)| {
                    if *violation <= 0.0 {
                        *value
                    } else {
                        floor - 1.0 - violation
                    }
                })
                .collect()
        }
    }
}

// ----------------------------------------------------------------------------------------------
// Surrogate (unchanged random-threshold forest)
// ----------------------------------------------------------------------------------------------

#[derive(Clone)]
enum Tree {
    Leaf(f64),
    Split {
        feature: usize,
        threshold: f64,
        left: Box<Tree>,
        right: Box<Tree>,
    },
}

impl Tree {
    fn predict(&self, row: &[f64]) -> f64 {
        match self {
            Self::Leaf(value) => *value,
            Self::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                if row[*feature] <= *threshold {
                    left.predict(row)
                } else {
                    right.predict(row)
                }
            }
        }
    }
}

struct Forest {
    trees: Vec<Tree>,
}

impl Forest {
    fn fit(x: &[Vec<f64>], y: &[f64], rng: &mut ChaCha8Rng, tree_count: usize) -> (Self, Vec<f64>) {
        let dimensions = x[0].len();
        let mut importance = vec![0.0; dimensions];
        let mut trees = Vec::with_capacity(tree_count);
        for _ in 0..tree_count {
            let rows = (0..x.len()).collect::<Vec<_>>();
            trees.push(build_tree(x, y, rows, 0, rng, &mut importance));
        }
        let total: f64 = importance.iter().sum();
        if total > 0.0 {
            importance.iter_mut().for_each(|value| *value /= total);
        } else if dimensions > 0 {
            importance.fill(1.0 / dimensions as f64);
        }
        (Self { trees }, importance)
    }

    fn mean_std(&self, row: &[f64]) -> (f64, f64) {
        let predictions: Vec<f64> = self.trees.iter().map(|tree| tree.predict(row)).collect();
        let mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
        let variance = predictions
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / predictions.len() as f64;
        (mean, variance.sqrt())
    }
}

fn build_tree(
    x: &[Vec<f64>],
    y: &[f64],
    rows: Vec<usize>,
    depth: usize,
    rng: &mut ChaCha8Rng,
    importance: &mut [f64],
) -> Tree {
    let average = rows.iter().map(|&row| y[row]).sum::<f64>() / rows.len() as f64;
    if rows.len() < 5 || depth >= 8 {
        return Tree::Leaf(average);
    }

    let dimensions = x[0].len();
    let feature_count = ((dimensions as f64).sqrt().ceil() as usize).max(1);
    let mut features: Vec<usize> = (0..dimensions).collect();
    features.shuffle(rng);
    features.truncate(feature_count);

    let parent_error = squared_error(y, &rows);
    let mut best: Option<(usize, f64, f64)> = None;
    for feature in features {
        let low = rows
            .iter()
            .map(|&row| x[row][feature])
            .fold(f64::INFINITY, f64::min);
        let high = rows
            .iter()
            .map(|&row| x[row][feature])
            .fold(f64::NEG_INFINITY, f64::max);
        if high - low <= 1.0e-12 {
            continue;
        }
        for _ in 0..6 {
            let threshold = rng.gen_range(low..high);
            let (left, right): (Vec<_>, Vec<_>) = rows
                .iter()
                .copied()
                .partition(|&row| x[row][feature] <= threshold);
            if left.len() < 2 || right.len() < 2 {
                continue;
            }
            let gain = parent_error - squared_error(y, &left) - squared_error(y, &right);
            if best.map_or(true, |(_, _, best_gain)| gain > best_gain) {
                best = Some((feature, threshold, gain));
            }
        }
    }

    let Some((feature, threshold, gain)) = best.filter(|(_, _, gain)| *gain > 1.0e-12) else {
        return Tree::Leaf(average);
    };
    importance[feature] += gain;
    let (left_rows, right_rows): (Vec<_>, Vec<_>) = rows
        .into_iter()
        .partition(|&row| x[row][feature] <= threshold);
    Tree::Split {
        feature,
        threshold,
        left: Box::new(build_tree(x, y, left_rows, depth + 1, rng, importance)),
        right: Box::new(build_tree(x, y, right_rows, depth + 1, rng, importance)),
    }
}

fn squared_error(y: &[f64], rows: &[usize]) -> f64 {
    if rows.is_empty() {
        return 0.0;
    }
    let mean = rows.iter().map(|&row| y[row]).sum::<f64>() / rows.len() as f64;
    rows.iter().map(|&row| (y[row] - mean).powi(2)).sum()
}

// ----------------------------------------------------------------------------------------------
// Entry points
// ----------------------------------------------------------------------------------------------

/// Backwards-compatible sequential entry point (objective returns a plain score).
pub fn optimize<F>(config: AhcoConfig, mut objective: F) -> Result<OptimizationResult, AhcoError>
where
    F: FnMut(&[f64]) -> Result<f64, AhcoError>,
{
    optimize_core(config, |point| objective(point).map(Evaluation::from))
}

/// Sequential evaluation, objective may report a constraint violation.
pub fn optimize_eval<F>(
    config: AhcoConfig,
    mut objective: F,
) -> Result<OptimizationResult, AhcoError>
where
    F: FnMut(&[f64]) -> Result<Evaluation, AhcoError>,
{
    optimize_core(config, |point| objective(point))
}

// ----------------------------------------------------------------------------------------------
// Core loop
// ----------------------------------------------------------------------------------------------

fn evaluate_one<E>(
    archive: &mut Archive,
    point: &[f64],
    known_constraint: Option<&Constraint>,
    objective: &mut E,
) -> Result<(), AhcoError>
where
    E: FnMut(&[f64]) -> Result<Evaluation, AhcoError>,
{
    let evaluation = objective(point)?;
    let known_violation = known_constraint.map_or(0.0, |constraint| constraint.violation(point));
    archive.push(point.to_vec(), evaluation, known_violation);
    Ok(())
}

fn optimize_core<E>(config: AhcoConfig, mut objective: E) -> Result<OptimizationResult, AhcoError>
where
    E: FnMut(&[f64]) -> Result<Evaluation, AhcoError>,
{
    validate_config(&config)?;
    let started = Instant::now();
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let space = Space::new(&config);
    let constraint = config.known_constraint.clone();
    let constraint_ref = constraint.as_ref();
    let dimension = config.dimension;
    let budget = config.budget;
    let mut archive = Archive::new(budget, config.constraint_mode, constraint_ref.is_some());

    // A small grouped bootstrap is needed before score-based hypotheses are useful.
    let bootstrap = budget.min(6);
    let design = grouped_design(bootstrap, dimension, &space, constraint_ref, &mut rng);
    for point in design {
        evaluate_one(&mut archive, &point, constraint_ref, &mut objective)?;
    }

    let min_active = dimension.min(3);
    let max_active = dimension.min(config.max_active_dimensions);
    let (mut active, mut maxima) = choose_active(
        &archive.points,
        &archive.scores,
        &space,
        min_active,
        max_active,
        None,
        &mut rng,
    );

    let strategies = [
        Strategy::Ucb,
        Strategy::Block,
        Strategy::Direction,
        Strategy::Inactive,
        Strategy::ProxyStar,
        Strategy::GroupHypothesis,
    ];
    let mut wins = [2.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let mut losses = [1.0; 6];
    let mut group_trials = vec![0usize; dimension];
    let mut group_successes = vec![0usize; dimension];
    let mut trust_region = 0.45;
    let mut successes = 0usize;
    let mut failures = 0usize;
    let mut center_index: Option<usize> = None;
    let mut momentum = vec![0.0; active.len()];
    let mut model: Option<Forest> = None;
    let mut last_fit = 0usize;

    while archive.points.len() < budget {
        let count = archive.points.len();
        let inactive: Vec<usize> = (0..dimension)
            .filter(|feature| !active.contains(feature))
            .collect();
        let hypotheses =
            proxy_star_hypotheses(&archive.points, &archive.scores, &active, &space, &mut rng);
        let blocks = relationship_blocks(&active, &hypotheses);
        let best_index = argmax(&archive.scores);
        let best = archive.points[best_index].clone();
        let center = center_index
            .map(|index| archive.points[index].clone())
            .unwrap_or_else(|| best.clone());
        let scores_now = archive.scores.clone();
        let transformed_y = rank_scores(&scores_now);

        if model.is_none() || count.saturating_sub(last_fit) >= 2 {
            let active_x: Vec<Vec<f64>> = archive
                .points
                .iter()
                .map(|point| space.encode(point, &active))
                .collect();
            model = Some(Forest::fit(&active_x, &transformed_y, &mut rng, 45).0);
            last_fit = count;
        }
        let fitted = model.as_ref().expect("surrogate is fit before proposing");
        let beta = 1.4;
        let perturb_probability = (4.0 / active.len().max(1) as f64).clamp(0.25, 1.0);
        let available: Vec<usize> = strategies
            .iter()
            .enumerate()
            .filter(|(_, strategy)| {
                !(**strategy == Strategy::Block && blocks.len() < 2
                    || **strategy == Strategy::Inactive && inactive.is_empty()
                    || **strategy == Strategy::ProxyStar && hypotheses.is_empty())
            })
            .map(|(index, _)| index)
            .collect();

        let selected_strategy = sample_strategy(&available, &wins, &losses, &mut rng);
        let ctx = Ctx {
            space: &space,
            constraint: constraint_ref,
        };
        let candidate = match strategies[selected_strategy] {
            Strategy::Ucb => propose_ucb(
                &center,
                &active,
                fitted,
                trust_region,
                perturb_probability,
                beta,
                &ctx,
                &mut rng,
            ),
            Strategy::Block => propose_block(
                &archive.points,
                &scores_now,
                &center,
                &blocks,
                &active,
                fitted,
                trust_region,
                beta,
                &ctx,
                &mut rng,
            ),
            Strategy::Direction => propose_direction(
                &archive.points,
                &transformed_y,
                &best,
                &active,
                fitted,
                trust_region,
                &mut momentum,
                &ctx,
                &mut rng,
            ),
            Strategy::Inactive => propose_inactive(&best, &inactive, &maxima, &ctx, &mut rng),
            Strategy::ProxyStar => propose_proxy_star(
                &best,
                &hypotheses,
                &active,
                fitted,
                trust_region,
                &ctx,
                &mut rng,
            ),
            Strategy::GroupHypothesis => propose_group_hypothesis(
                &archive.points,
                &transformed_y,
                &best,
                &space,
                &group_trials,
                &group_successes,
                &ctx,
                &mut rng,
            ),
        };
        let group = if strategies[selected_strategy] == Strategy::GroupHypothesis {
            Some(
                (0..dimension)
                    .filter(|&feature| (candidate[feature] - best[feature]).abs() > 1.0e-12)
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };

        let best_before = argmax(&archive.scores);
        evaluate_one(&mut archive, &candidate, constraint_ref, &mut objective)?;
        let new_index = archive.points.len() - 1;
        let improved = archive.scores[new_index] > archive.scores[best_before] + 1.0e-12;
        if let Some(group) = group {
            for feature in group {
                group_trials[feature] += 1;
                if improved {
                    group_successes[feature] += 1;
                }
            }
        }

        for index in 0..wins.len() {
            wins[index] = 1.0 + 0.96 * (wins[index] - 1.0);
            losses[index] = 1.0 + 0.96 * (losses[index] - 1.0);
        }
        if improved {
            wins[selected_strategy] += 1.2;
            successes += 1;
            failures = 0;
            center_index = None;
            let selection = choose_active(
                &archive.points,
                &archive.scores,
                &space,
                min_active,
                max_active,
                Some(&active),
                &mut rng,
            );
            if selection.0 != active {
                momentum = vec![0.0; selection.0.len()];
                model = None;
            }
            active = selection.0;
            maxima = selection.1;
            if strategies[selected_strategy] == Strategy::Direction {
                momentum = active
                    .iter()
                    .map(|&feature| archive.points[new_index][feature] - best[feature])
                    .collect();
                normalize(&mut momentum);
            }
            if successes >= 2 {
                trust_region = (trust_region * 1.5).min(0.85);
                successes = 0;
            }
        } else {
            losses[selected_strategy] += 1.0;
            failures += 1;
            successes = 0;
            if failures >= 2.max(active.len().div_ceil(4)) {
                trust_region = (trust_region / 1.8).max(1.0e-3);
                failures = 0;
                if trust_region <= 0.005 {
                    trust_region = 0.30;
                    let mut top: Vec<usize> = (0..archive.scores.len()).collect();
                    top.sort_by(|&a, &b| archive.scores[b].total_cmp(&archive.scores[a]));
                    top.truncate(6);
                    center_index = top.choose(&mut rng).copied();
                    momentum.fill(0.0);
                    let selection = choose_active(
                        &archive.points,
                        &archive.scores,
                        &space,
                        min_active,
                        max_active,
                        Some(&active),
                        &mut rng,
                    );
                    active = selection.0;
                    maxima = selection.1;
                    model = None;
                }
            }
        }
    }

    let best_index = argmax(&archive.scores);
    let infeasible_evaluations = archive
        .violations
        .iter()
        .filter(|violation| **violation > 0.0)
        .count();
    Ok(OptimizationResult {
        method: "AHCO-AdaptiveGroupedHypothesis-research",
        best_score: archive.values[best_index],
        best_x: archive.points[best_index].clone(),
        feasible: archive.violations[best_index] <= 0.0,
        infeasible_evaluations,
        evaluations: archive.points.len(),
        history: archive.history,
        seconds: started.elapsed().as_secs_f64(),
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Strategy {
    Ucb,
    Block,
    Direction,
    Inactive,
    ProxyStar,
    GroupHypothesis,
}

fn validate_config(config: &AhcoConfig) -> Result<(), AhcoError> {
    if config.dimension == 0 {
        return Err(AhcoError::InvalidDimension);
    }
    if config.budget == 0 {
        return Err(AhcoError::InvalidBudget);
    }
    if config.max_active_dimensions == 0 {
        return Err(AhcoError::InvalidMaxActiveDimensions);
    }
    if !config.kinds.is_empty() {
        if config.kinds.len() != config.dimension {
            return Err(AhcoError::InvalidKinds);
        }
        if config
            .kinds
            .iter()
            .any(|kind| matches!(kind, ParamKind::Categorical(n) if *n < 2))
        {
            return Err(AhcoError::InvalidKinds);
        }
    }
    Ok(())
}

// ----------------------------------------------------------------------------------------------
// Designs
// ----------------------------------------------------------------------------------------------

fn grouped_point(dimension: usize, space: &Space, rng: &mut ChaCha8Rng) -> Vec<f64> {
    let normal = Normal::new(0.0, 0.04).expect("positive normal deviation");
    let groups =
        ((dimension as f64).powf(rng.gen_range(0.4..0.9)).round() as usize).clamp(1, dimension);
    let centers: Vec<f64> = (0..groups).map(|_| rng.gen_range(0.05..0.95)).collect();
    let labels: Vec<usize> = (0..dimension).map(|_| rng.gen_range(0..groups)).collect();
    (0..dimension)
        .map(|feature| {
            if space.is_cat(feature) {
                // grouping is meaningless for unordered levels: independent uniform level
                rng.gen::<f64>()
            } else {
                (centers[labels[feature]] + normal.sample(rng)).clamp(0.0, 1.0)
            }
        })
        .collect()
}

fn grouped_design(
    count: usize,
    dimension: usize,
    space: &Space,
    constraint: Option<&Constraint>,
    rng: &mut ChaCha8Rng,
) -> Vec<Vec<f64>> {
    let mut design = Vec::with_capacity(count);
    for _ in 0..count {
        let mut best: Option<Vec<f64>> = None;
        let mut best_violation = f64::INFINITY;
        for _ in 0..40 {
            let mut point = grouped_point(dimension, space, rng);
            space.snap(&mut point);
            let violation = constraint.map_or(0.0, |c| c.violation(&point));
            if best.is_none() || violation < best_violation {
                best_violation = violation;
                best = Some(point);
            }
            if violation <= 0.0 {
                break;
            }
        }
        design.push(best.expect("at least one design draw"));
    }
    design
}

// ----------------------------------------------------------------------------------------------
// Screening / grouping
// ----------------------------------------------------------------------------------------------

fn rank_scores(scores: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
    let mut ranks = vec![0.0; scores.len()];
    if scores.len() == 1 {
        return ranks;
    }
    let denominator = (scores.len() - 1) as f64;
    for (rank, index) in order.into_iter().enumerate() {
        ranks[index] = 2.0 * rank as f64 / denominator - 1.0;
    }
    ranks
}

/// ANOVA effect size sqrt(SS_between / SS_total) of `y` grouped by category level.
fn eta(levels: &[usize], y: &[f64], n: usize) -> f64 {
    let grand = mean(y);
    let total: f64 = y.iter().map(|value| (value - grand).powi(2)).sum();
    if total < 1.0e-12 {
        return 0.0;
    }
    let mut between = 0.0;
    for level in 0..n {
        let mut sum = 0.0;
        let mut count = 0usize;
        for (index, &candidate) in levels.iter().enumerate() {
            if candidate == level {
                sum += y[index];
                count += 1;
            }
        }
        if count > 0 {
            let group_mean = sum / count as f64;
            between += count as f64 * (group_mean - grand).powi(2);
        }
    }
    (between / total).sqrt()
}

fn choose_active(
    points: &[Vec<f64>],
    scores: &[f64],
    space: &Space,
    min_active: usize,
    max_active: usize,
    keep: Option<&[usize]>,
    rng: &mut ChaCha8Rng,
) -> (Vec<usize>, Vec<f64>) {
    let dimension = space.levels.len();
    let target = rank_scores(scores);
    let mut effect = vec![0.0; dimension];
    let mut maxima = vec![f64::NAN; dimension];
    for feature in 0..dimension {
        if space.is_cat(feature) {
            let n = space.levels[feature];
            let levels: Vec<usize> = points
                .iter()
                .map(|point| space.level(feature, point[feature]))
                .collect();
            effect[feature] = eta(&levels, &target, n);
            let mut best_level: Option<(usize, f64)> = None;
            for level in 0..n {
                let mut sum = 0.0;
                let mut count = 0usize;
                for (index, &candidate) in levels.iter().enumerate() {
                    if candidate == level {
                        sum += target[index];
                        count += 1;
                    }
                }
                if count > 0 {
                    let level_mean = sum / count as f64;
                    if best_level.map_or(true, |(_, best_mean)| level_mean > best_mean) {
                        best_level = Some((level, level_mean));
                    }
                }
            }
            if let Some((level, _)) = best_level {
                maxima[feature] = (level as f64 + 0.5) / n as f64;
            }
            continue;
        }
        let linear: Vec<f64> = points
            .iter()
            .map(|point| 2.0 * point[feature] - 1.0)
            .collect();
        let quadratic: Vec<f64> = linear
            .iter()
            .map(|value| value.powi(2) - 1.0 / 3.0)
            .collect();
        let linear_score = correlation(&linear, &target).abs();
        let quadratic_score = correlation(&quadratic, &target).abs();
        effect[feature] = linear_score.max(quadratic_score);
        let (a, b) = quadratic_fit(&linear, &target);
        if a.abs() + b.abs() > 1.0e-10 {
            let optimum = if b.abs() > 1.0e-10 {
                (-a / (2.0 * b)).clamp(-1.0, 1.0)
            } else if a > 0.0 {
                1.0
            } else {
                -1.0
            };
            maxima[feature] = ((optimum + 1.0) / 2.0).clamp(0.0, 1.0);
        }
    }

    let bootstraps = 24usize;
    let sample_size = points.len().div_ceil(2).max(6).min(points.len());
    let mut selected_count = vec![0usize; dimension];
    for _ in 0..bootstraps {
        let mut indices: Vec<usize> = (0..points.len()).collect();
        indices.shuffle(rng);
        indices.truncate(sample_size);
        let y: Vec<f64> = indices.iter().map(|&i| target[i]).collect();
        for feature in 0..dimension {
            let strength = if space.is_cat(feature) {
                let levels: Vec<usize> = indices
                    .iter()
                    .map(|&i| space.level(feature, points[i][feature]))
                    .collect();
                eta(&levels, &y, space.levels[feature])
            } else {
                let x: Vec<f64> = indices.iter().map(|&i| points[i][feature]).collect();
                correlation(&x, &y).abs()
            };
            if strength >= 0.22 {
                selected_count[feature] += 1;
            }
        }
    }
    let mut score: Vec<f64> = effect
        .iter()
        .zip(selected_count)
        .map(|(&magnitude, frequency)| magnitude + frequency as f64 / bootstraps as f64)
        .collect();

    let all_features: Vec<usize> = (0..dimension).collect();
    let encoded: Vec<Vec<f64>> = points
        .iter()
        .map(|point| space.encode(point, &all_features))
        .collect();
    let importance = Forest::fit(&encoded, &target, rng, 35).1;
    let mut position = 0usize;
    for feature in 0..dimension {
        let width = space.width(feature);
        score[feature] += importance[position..position + width].iter().sum::<f64>();
        position += width;
    }
    if let Some(kept) = keep {
        for &feature in kept {
            score[feature] *= 1.15;
        }
    }

    let mut active: Vec<usize> = (0..dimension).collect();
    active.sort_by(|&a, &b| score[b].total_cmp(&score[a]));
    active.truncate(max_active.min(dimension));
    if active.len() < min_active {
        active = (0..dimension).collect();
        active.sort_by(|&a, &b| score[b].total_cmp(&score[a]));
        active.truncate(min_active.min(dimension));
    }
    active.sort_unstable();
    (active, maxima)
}

fn quadratic_fit(x: &[f64], y: &[f64]) -> (f64, f64) {
    let linear: Vec<f64> = x.to_vec();
    let quadratic: Vec<f64> = x.iter().map(|value| value.powi(2) - 1.0 / 3.0).collect();
    let ll = covariance(&linear, &linear);
    let qq = covariance(&quadratic, &quadratic);
    let lq = covariance(&linear, &quadratic);
    let ly = covariance(&linear, y);
    let qy = covariance(&quadratic, y);
    let determinant = ll * qq - lq * lq;
    if determinant.abs() < 1.0e-12 {
        (ly, 0.0)
    } else {
        (
            (ly * qq - qy * lq) / determinant,
            (qy * ll - ly * lq) / determinant,
        )
    }
}

#[derive(Clone, Debug)]
struct ProxyHypothesis {
    left: usize,
    right: usize,
    direction: i8,
    confidence: f64,
    mediator: Option<usize>,
}

fn proxy_star_hypotheses(
    points: &[Vec<f64>],
    scores: &[f64],
    active: &[usize],
    space: &Space,
    rng: &mut ChaCha8Rng,
) -> Vec<ProxyHypothesis> {
    if active.len() < 2 || points.len() < 8 {
        return Vec::new();
    }
    let target = rank_scores(scores);
    let real: Vec<usize> = active
        .iter()
        .copied()
        .filter(|&feature| !space.is_cat(feature))
        .collect();
    let mut direct = Vec::new();
    for left in 0..real.len() {
        for right in left + 1..real.len() {
            let product: Vec<f64> = points
                .iter()
                .map(|point| (2.0 * point[real[left]] - 1.0) * (2.0 * point[real[right]] - 1.0))
                .collect();
            let effect = correlation(&product, &target);
            if effect.abs() < 0.15 {
                continue;
            }

            let sample_size = points.len().div_ceil(2).max(4).min(points.len());
            let mut consistent = 0usize;
            for _ in 0..24 {
                let mut indices: Vec<usize> = (0..points.len()).collect();
                indices.shuffle(rng);
                indices.truncate(sample_size);
                let sampled_product: Vec<f64> =
                    indices.iter().map(|&index| product[index]).collect();
                let sampled_target: Vec<f64> = indices.iter().map(|&index| target[index]).collect();
                let sampled_effect = correlation(&sampled_product, &sampled_target);
                if sampled_effect * effect > 0.0 {
                    consistent += 1;
                }
            }
            let stability = consistent as f64 / 24.0;
            if stability >= 0.70 {
                direct.push(ProxyHypothesis {
                    left: real[left],
                    right: real[right],
                    direction: if effect >= 0.0 { 1 } else { -1 },
                    confidence: stability * (effect.abs() / 0.4).min(1.0),
                    mediator: None,
                });
            }
        }
    }

    let mut hypotheses = direct.clone();
    for (first_index, first) in direct.iter().enumerate() {
        for second in direct.iter().skip(first_index + 1) {
            let (mediator, left, right) = if first.left == second.left {
                (first.left, first.right, second.right)
            } else if first.left == second.right {
                (first.left, first.right, second.left)
            } else if first.right == second.left {
                (first.right, first.left, second.right)
            } else if first.right == second.right {
                (first.right, first.left, second.left)
            } else {
                continue;
            };
            if left == right {
                continue;
            }
            let confidence = first.confidence * second.confidence;
            if confidence < 0.20 {
                continue;
            }
            let inferred = ProxyHypothesis {
                left,
                right,
                direction: first.direction * second.direction,
                confidence,
                mediator: Some(mediator),
            };
            let pair = ordered_pair(inferred.left, inferred.right);
            if let Some(existing) = hypotheses
                .iter_mut()
                .find(|hypothesis| ordered_pair(hypothesis.left, hypothesis.right) == pair)
            {
                if inferred.confidence > existing.confidence {
                    *existing = inferred;
                }
            } else {
                hypotheses.push(inferred);
            }
        }
    }
    hypotheses
}

fn ordered_pair(left: usize, right: usize) -> (usize, usize) {
    if left < right {
        (left, right)
    } else {
        (right, left)
    }
}

fn relationship_blocks(active: &[usize], hypotheses: &[ProxyHypothesis]) -> Vec<Vec<usize>> {
    let mut parent: Vec<usize> = (0..active.len()).collect();
    for hypothesis in hypotheses {
        let Some(left) = active
            .iter()
            .position(|&feature| feature == hypothesis.left)
        else {
            continue;
        };
        let Some(right) = active
            .iter()
            .position(|&feature| feature == hypothesis.right)
        else {
            continue;
        };
        union(&mut parent, left, right);
    }
    let mut blocks: Vec<Vec<usize>> = Vec::new();
    let mut roots: Vec<usize> = Vec::new();
    for (index, &feature) in active.iter().enumerate() {
        let root = find(&mut parent, index);
        if let Some(position) = roots.iter().position(|&candidate| candidate == root) {
            blocks[position].push(feature);
        } else {
            roots.push(root);
            blocks.push(vec![feature]);
        }
    }
    blocks
}

fn group_dimension_weights(
    points: &[Vec<f64>],
    target: &[f64],
    space: &Space,
    trials: &[usize],
    successes: &[usize],
) -> Vec<f64> {
    let dimension = space.levels.len();
    let mut evidence = vec![0.0; dimension];
    for feature in 0..dimension {
        evidence[feature] = if space.is_cat(feature) {
            let levels: Vec<usize> = points
                .iter()
                .map(|point| space.level(feature, point[feature]))
                .collect();
            eta(&levels, target, space.levels[feature])
        } else {
            let values: Vec<f64> = points
                .iter()
                .map(|point| 2.0 * point[feature] - 1.0)
                .collect();
            let quadratic: Vec<f64> = values
                .iter()
                .map(|value| value.powi(2) - 1.0 / 3.0)
                .collect();
            correlation(&values, target)
                .abs()
                .max(correlation(&quadratic, target).abs())
        };
    }

    let evidence_total = evidence.iter().sum::<f64>();
    let uncertainty: Vec<f64> = trials
        .iter()
        .map(|&count| 1.0 / (1.0 + count as f64).sqrt())
        .collect();
    let uncertainty_total = uncertainty.iter().sum::<f64>();
    (0..dimension)
        .map(|feature| {
            0.50 / dimension as f64
                + 0.25 * evidence[feature] / evidence_total.max(1.0e-12)
                + 0.25 * uncertainty[feature] / uncertainty_total.max(1.0e-12)
                + 0.10 * successes[feature] as f64
                    / (2.0 + trials[feature] as f64)
                    / dimension as f64
        })
        .collect()
}

fn sample_group_size(dimension: usize, rng: &mut ChaCha8Rng) -> usize {
    let maximum = dimension.min(256);
    let mut sizes = vec![1usize];
    while *sizes.last().expect("group size list is non-empty") < maximum {
        sizes.push((sizes.last().copied().unwrap() * 2).min(maximum));
    }
    for size in [dimension / 2, dimension] {
        if size > 0 && !sizes.contains(&size) {
            sizes.push(size);
        }
    }
    sizes[rng.gen_range(0..sizes.len())]
}

fn propose_group_hypothesis(
    points: &[Vec<f64>],
    target: &[f64],
    best: &[f64],
    space: &Space,
    group_trials: &[usize],
    group_successes: &[usize],
    ctx: &Ctx,
    rng: &mut ChaCha8Rng,
) -> Vec<f64> {
    let dimension = space.levels.len();
    let weights = group_dimension_weights(points, target, space, group_trials, group_successes);
    let group_size = sample_group_size(dimension, rng);
    let mut remaining: Vec<usize> = (0..dimension).collect();
    let mut group = Vec::with_capacity(group_size);
    let mut remaining_weight: f64 = weights.iter().sum();
    for _ in 0..group_size {
        let mut draw = rng.gen::<f64>() * remaining_weight;
        let mut selected = remaining.len() - 1;
        for (position, &feature) in remaining.iter().enumerate() {
            draw -= weights[feature];
            if draw <= 0.0 {
                selected = position;
                break;
            }
        }
        let feature = remaining.swap_remove(selected);
        remaining_weight -= weights[feature];
        group.push(feature);
    }

    let group_x: Vec<Vec<f64>> = points
        .iter()
        .map(|point| space.encode(point, &group))
        .collect();
    let (model, _) = Forest::fit(&group_x, target, rng, 30);
    let mut candidates = Vec::with_capacity(192);
    for _ in 0..192 {
        let mut candidate = best.to_vec();
        let pivot = group[rng.gen_range(0..group.len())];
        let pivot_value = [0.0, 0.5, 1.0][rng.gen_range(0..3)];
        for &feature in &group {
            if feature != pivot && rng.gen_bool(0.25) {
                continue;
            }
            if space.is_cat(feature) {
                let levels = space.levels[feature];
                let current = space.level(feature, candidate[feature]);
                let level = if rng.gen_bool(0.65) {
                    let draw = rng.gen_range(0..levels - 1);
                    if draw >= current {
                        draw + 1
                    } else {
                        draw
                    }
                } else {
                    space.level(feature, candidate[feature])
                };
                candidate[feature] = (level as f64 + 0.5) / levels as f64;
            } else if feature == pivot || rng.gen_bool(0.45) {
                candidate[feature] = pivot_value;
            } else {
                candidate[feature] = [0.0, 0.5, 1.0][rng.gen_range(0..3)];
            }
        }
        candidates.push(candidate);
    }
    pick_best(candidates, &model, &group, 1.4, ctx)
}

fn find(parent: &mut [usize], item: usize) -> usize {
    if parent[item] != item {
        let root = find(parent, parent[item]);
        parent[item] = root;
    }
    parent[item]
}

fn union(parent: &mut [usize], left: usize, right: usize) {
    let left_root = find(parent, left);
    let right_root = find(parent, right);
    if left_root != right_root {
        parent[right_root] = left_root;
    }
}

// ----------------------------------------------------------------------------------------------
// Proposals
// ----------------------------------------------------------------------------------------------

/// Keep only constraint-satisfying candidates; if none satisfy, keep the least-violating ones.
fn filter_feasible(candidates: Vec<Vec<f64>>, constraint: Option<&Constraint>) -> Vec<Vec<f64>> {
    let Some(constraint) = constraint else {
        return candidates;
    };
    let violations: Vec<f64> = candidates
        .iter()
        .map(|candidate| constraint.violation(candidate))
        .collect();
    let least = violations.iter().copied().fold(f64::INFINITY, f64::min);
    candidates
        .into_iter()
        .zip(violations)
        .filter(|(_, violation)| *violation <= least)
        .map(|(candidate, _)| candidate)
        .collect()
}

fn acquisition(model: &Forest, point: &[f64], active: &[usize], beta: f64, ctx: &Ctx) -> f64 {
    let (mean, deviation) = model.mean_std(&ctx.space.encode(point, active));
    mean + beta * deviation
}

fn pick_best(
    mut candidates: Vec<Vec<f64>>,
    model: &Forest,
    active: &[usize],
    beta: f64,
    ctx: &Ctx,
) -> Vec<f64> {
    candidates
        .iter_mut()
        .for_each(|candidate| ctx.space.snap(candidate));
    let candidates = filter_feasible(candidates, ctx.constraint);
    candidates
        .into_iter()
        .map(|candidate| (acquisition(model, &candidate, active, beta, ctx), candidate))
        .max_by(|left, right| left.0.total_cmp(&right.0))
        .map(|(_, candidate)| candidate)
        .expect("candidate pool is non-empty")
}

fn propose_ucb(
    center: &[f64],
    active: &[usize],
    model: &Forest,
    trust_region: f64,
    perturb_probability: f64,
    beta: f64,
    ctx: &Ctx,
    rng: &mut ChaCha8Rng,
) -> Vec<f64> {
    let space = ctx.space;
    let mut candidates = Vec::with_capacity(600);
    for _ in 0..600 {
        let mut candidate = center.to_vec();
        let forced = rng.gen_range(0..active.len());
        for (index, &feature) in active.iter().enumerate() {
            if index == forced || rng.gen_bool(perturb_probability.clamp(0.0, 1.0)) {
                if space.is_cat(feature) {
                    candidate[feature] = space.random_level(feature, rng);
                } else {
                    let low = (center[feature] - trust_region / 2.0).clamp(0.0, 1.0);
                    let high = (center[feature] + trust_region / 2.0).clamp(0.0, 1.0);
                    candidate[feature] = if high > low {
                        rng.gen_range(low..high)
                    } else {
                        low
                    };
                }
            }
        }
        candidates.push(candidate);
    }
    pick_best(candidates, model, active, beta, ctx)
}

fn propose_block(
    points: &[Vec<f64>],
    scores: &[f64],
    center: &[f64],
    blocks: &[Vec<usize>],
    active: &[usize],
    model: &Forest,
    trust_region: f64,
    beta: f64,
    ctx: &Ctx,
    rng: &mut ChaCha8Rng,
) -> Vec<f64> {
    let space = ctx.space;
    let block = blocks.choose(rng).expect("at least one block");
    let mut candidates = Vec::with_capacity(250);
    let mut elite: Vec<usize> = (0..scores.len()).collect();
    elite.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    elite.truncate(scores.len().min(12));
    let scale = (2.0 * trust_region).min(1.0);
    let normal = Normal::new(0.0, 0.20).expect("positive deviation");
    for _ in 0..250 {
        let mut candidate = center.to_vec();
        let source = &points[*elite.choose(rng).expect("non-empty observations")];
        for &feature in block {
            if space.is_cat(feature) {
                candidate[feature] = if rng.gen_bool(0.5) {
                    source[feature]
                } else {
                    space.random_level(feature, rng)
                };
            } else {
                candidate[feature] = (source[feature] + normal.sample(rng) * scale).clamp(0.0, 1.0);
            }
        }
        candidates.push(candidate);
    }
    pick_best(candidates, model, active, beta, ctx)
}

fn propose_direction(
    points: &[Vec<f64>],
    transformed_y: &[f64],
    best: &[f64],
    active: &[usize],
    model: &Forest,
    trust_region: f64,
    momentum: &mut [f64],
    ctx: &Ctx,
    rng: &mut ChaCha8Rng,
) -> Vec<f64> {
    let space = ctx.space;
    let mut neighbors: Vec<(f64, usize)> = points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let distance = active
                .iter()
                .map(|&feature| (point[feature] - best[feature]).powi(2))
                .sum::<f64>();
            (distance, index)
        })
        .collect();
    neighbors.sort_by(|left, right| left.0.total_cmp(&right.0));
    neighbors.truncate(points.len().min(4 * active.len() + 8));
    let mut gradient = vec![0.0; active.len()];
    for (position, &feature) in active.iter().enumerate() {
        if space.is_cat(feature) {
            continue; // no gradient along an unordered dimension
        }
        let x: Vec<f64> = neighbors
            .iter()
            .map(|&(_, index)| points[index][feature] - best[feature])
            .collect();
        let y: Vec<f64> = neighbors
            .iter()
            .map(|&(_, index)| transformed_y[index])
            .collect();
        gradient[position] = covariance(&x, &y) / (variance(&x) + 1.0e-4);
    }
    if gradient.iter().all(|value| value.abs() < 1.0e-9) {
        for (position, &feature) in active.iter().enumerate() {
            if !space.is_cat(feature) {
                gradient[position] = rng.gen_range(-1.0..1.0);
            }
        }
    }
    normalize(&mut gradient);
    for (value, &prior) in gradient.iter_mut().zip(momentum.iter()) {
        *value = 0.7 * *value + 0.3 * prior;
    }
    normalize(&mut gradient);
    let mut candidates = Vec::with_capacity(6);
    for step in [0.1, 0.25, 0.5, 0.8, 1.2, 1.8] {
        let mut candidate = best.to_vec();
        for (index, &feature) in active.iter().enumerate() {
            if !space.is_cat(feature) {
                candidate[feature] =
                    (best[feature] + step * trust_region * gradient[index]).clamp(0.0, 1.0);
            }
        }
        space.snap(&mut candidate);
        candidates.push(candidate);
    }
    let candidates = filter_feasible(candidates, ctx.constraint);
    let mut best_candidate = best.to_vec();
    let mut best_prediction = f64::NEG_INFINITY;
    for candidate in candidates {
        let prediction = model.mean_std(&space.encode(&candidate, active)).0;
        if prediction > best_prediction {
            best_prediction = prediction;
            best_candidate = candidate;
        }
    }
    best_candidate
}

fn propose_inactive(
    best: &[f64],
    inactive: &[usize],
    maxima: &[f64],
    ctx: &Ctx,
    rng: &mut ChaCha8Rng,
) -> Vec<f64> {
    let space = ctx.space;
    let group_size = ((inactive.len() as f64 * 0.3).round() as usize)
        .max(1)
        .min(inactive.len());
    let mut group = inactive.to_vec();
    group.shuffle(rng);
    group.truncate(group_size);
    let mut use_model =
        group.iter().any(|&feature| maxima[feature].is_finite()) && rng.gen_bool(0.6);
    let mut candidate = best.to_vec();
    for _ in 0..10 {
        candidate = best.to_vec();
        for &feature in &group {
            candidate[feature] = if use_model && maxima[feature].is_finite() {
                maxima[feature]
            } else if space.is_cat(feature) {
                space.random_level(feature, rng)
            } else {
                rng.gen_range(0.05..0.95)
            };
        }
        space.snap(&mut candidate);
        let violation = ctx.constraint.map_or(0.0, |c| c.violation(&candidate));
        if violation <= 0.0 {
            break;
        }
        use_model = false; // retry with random values if the model-guided one is infeasible
    }
    candidate
}

fn propose_proxy_star(
    best: &[f64],
    hypotheses: &[ProxyHypothesis],
    active: &[usize],
    model: &Forest,
    trust_region: f64,
    ctx: &Ctx,
    rng: &mut ChaCha8Rng,
) -> Vec<f64> {
    let total_confidence: f64 = hypotheses
        .iter()
        .map(|hypothesis| hypothesis.confidence)
        .sum();
    let mut draw = rng.gen::<f64>() * total_confidence;
    let mut selected = &hypotheses[0];
    for hypothesis in hypotheses {
        draw -= hypothesis.confidence;
        if draw <= 0.0 {
            selected = hypothesis;
            break;
        }
    }

    let mut candidates = Vec::with_capacity(160);
    for _ in 0..160 {
        let mut candidate = best.to_vec();
        let sign = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
        let distance = rng.gen_range(0.1..=1.0) * trust_region;
        let left_distance = distance * rng.gen_range(0.5..1.5);
        let right_distance = distance * rng.gen_range(0.5..1.5);
        candidate[selected.left] = (best[selected.left] + sign * left_distance).clamp(0.0, 1.0);
        candidate[selected.right] = (best[selected.right]
            + sign * selected.direction as f64 * right_distance)
            .clamp(0.0, 1.0);
        if let Some(mediator) = selected.mediator {
            let mediator_distance = distance * rng.gen_range(0.5..1.5);
            candidate[mediator] =
                (best[mediator] + rng.gen_range(-1.0..1.0) * mediator_distance).clamp(0.0, 1.0);
        }
        candidates.push(candidate);
    }
    pick_best(candidates, model, active, 1.4, ctx)
}

fn sample_strategy(
    available: &[usize],
    wins: &[f64; 6],
    losses: &[f64; 6],
    rng: &mut ChaCha8Rng,
) -> usize {
    let mut best = available[0];
    let mut best_sample = f64::NEG_INFINITY;
    for &index in available {
        let win = Gamma::new(wins[index].max(1.0e-6), 1.0)
            .expect("positive beta shape")
            .sample(rng);
        let loss = Gamma::new(losses[index].max(1.0e-6), 1.0)
            .expect("positive beta shape")
            .sample(rng);
        let sample = win / (win + loss);
        if sample > best_sample {
            best = index;
            best_sample = sample;
        }
    }
    best
}

fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .expect("optimization always has an observation")
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn covariance(left: &[f64], right: &[f64]) -> f64 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }
    let left_mean = mean(left);
    let right_mean = mean(right);
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - left_mean) * (b - right_mean))
        .sum::<f64>()
        / left.len() as f64
}

fn variance(values: &[f64]) -> f64 {
    covariance(values, values)
}

fn correlation(left: &[f64], right: &[f64]) -> f64 {
    let denominator = (variance(left) * variance(right)).sqrt();
    if denominator <= 1.0e-12 {
        0.0
    } else {
        (covariance(left, right) / denominator).clamp(-1.0, 1.0)
    }
}

fn normalize(values: &mut [f64]) {
    let norm = values.iter().map(|value| value.powi(2)).sum::<f64>().sqrt();
    if norm > 1.0e-12 {
        values.iter_mut().for_each(|value| *value /= norm);
    }
}

// ----------------------------------------------------------------------------------------------
// Python bindings
// ----------------------------------------------------------------------------------------------

#[cfg(feature = "python")]
mod python {
    use super::{optimize, AhcoConfig, AhcoError, ParamKind};
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyList};

    /// `levels`: per-dimension category count, 0 (or 1) = real-valued. Categorical dims arrive
    /// in the objective as cell centres; decode with min(int(x * n), n - 1).
    #[pyfunction]
    #[pyo3(signature = (f, d, budget, seed, kmax=14, levels=None))]
    fn run_ahco3(
        py: Python<'_>,
        f: Py<PyAny>,
        d: usize,
        budget: usize,
        seed: u64,
        kmax: usize,
        levels: Option<Vec<usize>>,
    ) -> PyResult<Py<PyDict>> {
        let mut config = AhcoConfig::new(d, budget, seed).with_max_active_dimensions(kmax);
        if let Some(levels) = levels {
            config = config.with_kinds(
                levels
                    .iter()
                    .map(|&n| {
                        if n >= 2 {
                            ParamKind::Categorical(n)
                        } else {
                            ParamKind::Real
                        }
                    })
                    .collect(),
            );
        }
        let mut python_error = None;
        let result = optimize(config, |point| {
            Python::with_gil(|py| {
                let values = PyList::new(py, point)
                    .map_err(|error| AhcoError::Objective(error.to_string()))?;
                let value = f.call1(py, (values,)).map_err(|error| {
                    python_error = Some(error);
                    AhcoError::Objective("Python objective raised an exception".into())
                })?;
                value.extract::<f64>(py).map_err(|error| {
                    python_error = Some(error);
                    AhcoError::Objective("Python objective must return a number".into())
                })
            })
        });
        if let Some(error) = python_error {
            return Err(error);
        }
        let result =
            result.map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        let output = PyDict::new(py);
        output.set_item("method", result.method)?;
        output.set_item("best_score", result.best_score)?;
        output.set_item("best_x", result.best_x)?;
        output.set_item("evaluations", result.evaluations)?;
        output.set_item("history", result.history)?;
        output.set_item("seconds", result.seconds)?;
        Ok(output.into())
    }

    #[pymodule]
    fn ahco(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_function(wrap_pyfunction!(run_ahco3, module)?)?;
        Ok(())
    }
}

// ----------------------------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{
        group_dimension_weights, optimize, optimize_eval, proxy_star_hypotheses, sample_group_size,
        AhcoConfig, AhcoError, ConstraintMode, Evaluation, ParamKind, Space,
    };
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn evaluates_exact_budget_and_returns_bounded_best_point() {
        let result = optimize(AhcoConfig::new(3, 25, 41), |point| {
            Ok(-point.iter().map(|value| (value - 0.7).powi(2)).sum::<f64>())
        })
        .unwrap();
        assert_eq!(result.evaluations, 25);
        assert_eq!(result.method, "AHCO-AdaptiveGroupedHypothesis-research");
        assert_eq!(result.history.len(), 25);
        assert!(result
            .best_x
            .iter()
            .all(|value| (0.0..=1.0).contains(value)));
        assert!(result.history.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn grouped_search_weights_influential_and_uncertain_dimensions() {
        let mut rng = ChaCha8Rng::seed_from_u64(2027);
        let points: Vec<Vec<f64>> = (0..100)
            .map(|_| vec![rng.gen(), rng.gen(), rng.gen()])
            .collect();
        let scores: Vec<f64> = points.iter().map(|point| point[0]).collect();
        let space = Space { levels: vec![0; 3] };
        let weights = group_dimension_weights(
            &points,
            &super::rank_scores(&scores),
            &space,
            &[0, 0, 4],
            &[0, 0, 0],
        );

        assert!(weights[0] > weights[1]);
        assert!(weights[1] > weights[2]);
    }

    #[test]
    fn group_hypothesis_sizes_cover_multiple_scales() {
        let mut rng = ChaCha8Rng::seed_from_u64(2028);
        let sizes: Vec<usize> = (0..100)
            .map(|_| sample_group_size(1_000, &mut rng))
            .collect();

        assert!(sizes.iter().all(|&size| (1..=1_000).contains(&size)));
        assert!(sizes.iter().any(|&size| size <= 4));
        assert!(sizes.iter().any(|&size| size >= 128));
        assert!(sizes.contains(&500));
        assert!(sizes.contains(&1_000));
    }

    #[test]
    fn proxy_star_composes_opposite_pair_directions() {
        let mut rng = ChaCha8Rng::seed_from_u64(2026);
        let points: Vec<Vec<f64>> = (0..400)
            .map(|_| vec![rng.gen(), rng.gen(), rng.gen()])
            .collect();
        let scores: Vec<f64> = points
            .iter()
            .map(|point| {
                let a = 2.0 * point[0] - 1.0;
                let b = 2.0 * point[1] - 1.0;
                let c = 2.0 * point[2] - 1.0;
                a * b - b * c
            })
            .collect();
        let space = Space { levels: vec![0; 3] };
        let hypotheses = proxy_star_hypotheses(&points, &scores, &[0, 1, 2], &space, &mut rng);

        assert!(hypotheses.iter().any(|hypothesis| {
            hypothesis.mediator == Some(1)
                && (hypothesis.left, hypothesis.right) == (0, 2)
                && hypothesis.direction == -1
                && hypothesis.confidence >= 0.20
        }));
    }

    #[test]
    fn proxy_star_requires_sufficient_observations() {
        let space = Space { levels: vec![0; 2] };
        let hypotheses = proxy_star_hypotheses(
            &vec![vec![0.1, 0.9]; 7],
            &[0.0; 7],
            &[0, 1],
            &space,
            &mut ChaCha8Rng::seed_from_u64(1),
        );
        assert!(hypotheses.is_empty());
    }

    #[test]
    fn repeated_seed_produces_same_observations() {
        let objective = |point: &[f64]| Ok(point[0] * point[0] - point[1]);
        let first = optimize(AhcoConfig::new(2, 18, 123), objective).unwrap();
        let second = optimize(AhcoConfig::new(2, 18, 123), objective).unwrap();
        assert_eq!(first.best_score, second.best_score);
        assert_eq!(first.best_x, second.best_x);
        assert_eq!(first.history, second.history);
    }

    #[test]
    fn non_finite_scores_are_penalized() {
        let result = optimize(AhcoConfig::new(1, 3, 3), |_| Ok(f64::NAN)).unwrap();
        assert_eq!(result.best_score, -1.0e100);
    }

    #[test]
    fn invalid_config_is_rejected_before_objective_call() {
        let result = optimize(AhcoConfig::new(0, 10, 0), |_| Ok(0.0));
        assert_eq!(result.unwrap_err(), AhcoError::InvalidDimension);
        let result = optimize(
            AhcoConfig::new(2, 10, 0).with_kinds(vec![ParamKind::Real]),
            |_| Ok(0.0),
        );
        assert_eq!(result.unwrap_err(), AhcoError::InvalidKinds);
    }

    #[test]
    fn categorical_points_are_snapped_to_cell_centres() {
        let seen = std::cell::RefCell::new(Vec::new());
        let config = AhcoConfig::new(3, 30, 5).with_kinds(vec![
            ParamKind::Real,
            ParamKind::Categorical(4),
            ParamKind::Categorical(3),
        ]);
        optimize(config, |point| {
            seen.borrow_mut().push(point.to_vec());
            Ok(point[0])
        })
        .unwrap();
        for point in seen.borrow().iter() {
            let a = point[1] * 4.0 - 0.5;
            let b = point[2] * 3.0 - 0.5;
            assert!((a - a.round()).abs() < 1.0e-9, "dim 1 not on a cell centre");
            assert!((b - b.round()).abs() < 1.0e-9, "dim 2 not on a cell centre");
        }
    }

    #[test]
    fn known_constraint_keeps_the_best_point_feasible() {
        let config = AhcoConfig::new(2, 40, 11).with_known_constraint(|x| x[0] + x[1] - 1.0);
        let result = optimize_eval(config, |point| {
            Ok(Evaluation::from(point[0] + 0.5 * point[1]))
        })
        .unwrap();
        assert!(result.feasible);
        assert!(result.best_x[0] + result.best_x[1] <= 1.0 + 1.0e-12);
    }

    #[test]
    fn observed_violation_is_tracked() {
        let config =
            AhcoConfig::new(2, 30, 2).with_constraint_mode(ConstraintMode::FeasibilityFirst);
        let result = optimize_eval(config, |point| {
            Ok(Evaluation::new(point[0], (point[0] - 0.5).max(0.0)))
        })
        .unwrap();
        assert!(result.infeasible_evaluations <= result.evaluations);
        if result.feasible {
            assert!(result.best_x[0] <= 0.5);
        }
    }
}
