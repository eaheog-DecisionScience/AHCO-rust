#[path = "../ahco.rs"]
mod ahco_v4;

use ahco_v4::{
    optimize as optimize_v4, optimize_eval, AhcoConfig as ConfigV4, ConstraintMode, Evaluation,
    ParamKind,
};

fn objective(point: &[f64]) -> f64 {
    let sphere = point
        .iter()
        .map(|value| (value - 0.37).powi(2))
        .sum::<f64>();
    let interaction = (point[0] * point[1] - 0.16).powi(2);
    -(sphere + 0.5 * interaction)
}

fn assert_valid_result(result: &ahco_v4::OptimizationResult, budget: usize, dimension: usize) {
    assert_eq!(result.evaluations, budget);
    assert_eq!(result.history.len(), budget);
    assert_eq!(result.best_x.len(), dimension);
    assert!(result
        .best_x
        .iter()
        .all(|value| (0.0..=1.0).contains(value)));
    assert!(result.history.windows(2).all(|pair| pair[0] <= pair[1]));
}

#[test]
fn v4_sequential_real_search_respects_budget_over_fixed_seeds() {
    let budget = 72;
    let dimension = 8;
    let mut scores_v4 = Vec::new();

    for seed in 0..8 {
        let sequential = optimize_v4(
            ConfigV4::new(dimension, budget, seed).with_max_active_dimensions(dimension),
            |point| Ok(objective(point)),
        )
        .unwrap();

        assert_valid_result(&sequential, budget, dimension);
        scores_v4.push(sequential.best_score);
    }

    let mean = |scores: &[f64]| scores.iter().sum::<f64>() / scores.len() as f64;
    println!(
        "v4 sequential objective mean over 8 seeds (higher is better): {:.6}",
        mean(&scores_v4)
    );
}

#[test]
fn categorical_dimensions_stay_on_centres_and_explore_each_level() {
    let seen = std::cell::RefCell::new(Vec::new());
    let config = ConfigV4::new(4, 100, 17).with_kinds(vec![
        ParamKind::Real,
        ParamKind::Categorical(5),
        ParamKind::Categorical(3),
        ParamKind::Categorical(4),
    ]);
    let result = optimize_v4(config, |point| {
        seen.borrow_mut().push(point.to_vec());
        let category_1 = (point[1] * 5.0) as usize;
        let category_2 = (point[2] * 3.0) as usize;
        let category_3 = (point[3] * 4.0) as usize;
        let score = -(point[0] - 0.73).powi(2)
            - if category_1 == 4 { 0.0 } else { 1.0 }
            - if category_2 == 1 { 0.0 } else { 0.4 }
            - if category_3 == 2 { 0.0 } else { 0.6 };
        Ok(score)
    })
    .unwrap();

    assert_valid_result(&result, 100, 4);
    for point in seen.borrow().iter() {
        for (coordinate, levels) in [(1, 5), (2, 3), (3, 4)] {
            let scaled = point[coordinate] * levels as f64;
            assert!((scaled.fract() - 0.5).abs() < 1e-9);
        }
    }
    for (coordinate, levels) in [(1, 5), (2, 3), (3, 4)] {
        let observed_levels: std::collections::HashSet<usize> = seen
            .borrow()
            .iter()
            .map(|point| (point[coordinate] * levels as f64) as usize)
            .collect();
        assert_eq!(observed_levels.len(), levels);
    }
}

#[test]
fn known_constraint_returns_a_feasible_best_point() {
    let config = ConfigV4::new(2, 80, 31).with_known_constraint(|point| point[0] + point[1] - 1.0);
    let result = optimize_eval(config, |point| {
        let score = point[0] + 0.5 * point[1];
        let infeasible_bonus = if point[0] + point[1] > 1.0 {
            1_000.0
        } else {
            0.0
        };
        Ok(Evaluation::from(score + infeasible_bonus))
    })
    .unwrap();

    assert!(result.feasible, "best result: {result:?}");
    assert!(result.best_x[0] + result.best_x[1] <= 1.0 + 1e-12);
    assert!(result.infeasible_evaluations <= result.evaluations);
}

#[test]
fn infeasible_known_constraint_is_reported_without_claiming_feasibility() {
    let config = ConfigV4::new(2, 24, 13).with_known_constraint(|_| f64::NAN);
    let result = optimize_eval(config, |point| Ok(Evaluation::from(point[0] + point[1]))).unwrap();

    assert!(!result.feasible);
    assert_eq!(result.infeasible_evaluations, result.evaluations);
    assert_eq!(result.evaluations, 24);
}

#[test]
fn feasibility_first_prefers_feasible_results_over_better_infeasible_scores() {
    let config = ConfigV4::new(2, 50, 42).with_constraint_mode(ConstraintMode::FeasibilityFirst);
    let result = optimize_eval(config, |point| {
        let violation = (point[0] - 0.4).max(0.0);
        Ok(Evaluation::new(point[0] + point[1], violation))
    })
    .unwrap();

    assert!(result.feasible, "best result: {result:?}");
    assert!(result.best_x[0] <= 0.4 + 1e-12);
    assert!(result.history.windows(2).all(|pair| pair[0] <= pair[1]));
}

#[test]
fn objective_error_stops_before_the_next_sequential_evaluation() {
    let calls = std::cell::Cell::new(0);
    let result = optimize_v4(ConfigV4::new(2, 20, 7), |_| {
        calls.set(calls.get() + 1);
        Err(ahco_v4::AhcoError::Objective("test failure".into()))
    });

    assert!(matches!(result, Err(ahco_v4::AhcoError::Objective(_))));
    assert_eq!(calls.get(), 1);
}
