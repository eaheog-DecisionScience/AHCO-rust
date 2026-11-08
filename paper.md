---
title: "AHCO: Adaptive Hypothesis Coordinate Optimizer for Mixed Continuous and Categorical --  Black-Box Optimization"
authors:
  - name: Deepak Paney
    affiliation: "1"
affiliations:
  - index: 1
    name: Independent Researcher
date: 3 November 2026
bibliography: paper.bib
---

# Summary

The Adaptive Hypothesis Coordinate Optimizer (AHCO) is a Heuristic Optimizer which maximizes a scalar objective without need for derivatives under a given finite/ fixed evaluation budget.
The Mechanism : (The implicit-requisite) It searches bounded real parameters and unordered categorical parameters along with honouring known constraints and also violations of constraints/threshods which are reported by the objective.
Core loop: Its surrogate is an ensemble of randomized regression trees, and it chooses adaptively among four proposal strategies then it returns the best observed point, whether that point is feasible, and the best-so-far history.

AHCO (ahco_rust) is a Rust library with an optional Python extension. 
The disagreement among its trees is an exploration heuristic which works on prior guided screeniing done between different feature groups, and it does not show or measure/ calibrated probability, and a result is not a proof of a global optimum.

# Statement of need

Simulation and tuning problems often have no derivatives, costly evaluations, and parameters that mix continuous values with categories. Bayesian optimization is a widely used framework for such problems [@Shahriari2016], and mature libraries exist for hyperparameter search [@Akiba2019]. AHCO solves for a problem in work settings where need is for  a small, seeded, dependency-light Rust implementation (with Python bindings) for mixed bounded parameters under a fixed budget, with explicit constraint handling and deterministic batch evaluation.

Internally AHCO works on the unit cube $[0,1]^d$. A continuous physical parameter with bounds $l_j \leq x_j \leq u_j$ is mapped from $z_j\in[0,1]$ by $x_j=l_j+z_j(u_j-l_j)$. A categorical parameter with $r\geq2$ unordered levels is decoded as $\min(\lfloor r z_j\rfloor, r-1)$. Integer and log-scale parameters are not first-class types; users map them from the unit cube themselves. Explicit noise models are not supported.

```rust
use ahco::{optimize_parallel, AhcoConfig, Evaluation, ParamKind};

let config = AhcoConfig::new(3, 120, 7)
    .with_kinds(vec![ParamKind::Real, ParamKind::Real, ParamKind::Categorical(4)])
    .with_batch_size(4)
    .with_known_constraint(|x| x[0] + x[1] - 1.0); // <= 0 means feasible
let result = optimize_parallel(config, |x| {
    let level = ((x[2] * 4.0) as usize).min(3);
    let value = -(x[0] - 0.3).powi(2) - (x[1] - 0.6).powi(2) - 0.1 * level as f64;
    Ok(Evaluation::from(value))
})
.expect("optimization run");
```

# State of the field

Each main ingredient of AHCO has precedent.
 SMAC uses random-forest surrogates for mixed and categorical configured spaces [@Hutter2011]. 
 TuRBO adapts a trust region according to successes and failures [@Eriksson2019]. 
 Dropout Bayesian optimization optimizes a subset of coordinates at a time while filling in the rest [@Li2017]. Group screening tests coordinates together to find the important ones with few runs [@Watson1961]. 
 Nevergrad packages many derivative-free optimizers behind one interface [@Rapin2018], and bandit-style selection among proposal mechanisms is a common idea [@Thompson1933].

AHCO combines randomized trees with -
 (i) a grouped screening design in which random groups of coordinates share a value, 
 (ii) interaction blocks detected from product correlations and used as a dedicated proposal move, and 
 (iii) bandit selection among four strategies. 
 In AHCO "hypothesis" refers to grouped-coordinate design with proposed moves that help navigate and implicitly segment spaces into blocks.

Broader frameworks may suit users who need many algorithms, integer or log-scale parameter types, or asynchronous trials.

## Empirical evaluation

To evaluate AHCO in its intended high-dimensional, low-budget setting, we ran four deterministic, bounded maximization problems at dimension $d=97$ with a budget of $B=40$ objective calls per run ($B<d$). The suite includes dense Sphere (97 active coordinates), sparse Sphere (10 active coordinates), an ill-conditioned Ellipsoid (97 active coordinates), and Rosenbrock (97 active coordinates). Each analytic objective has known optimum score 0; higher scores are better. The formulations and fixed active-coordinate indices are in [`benchmarks/run_benchmarks.py`](./benchmarks/run_benchmarks.py).

These benchmark results describe the earlier AHCO-v4 policy; they are not an evaluation of the proxy-star or adaptive grouped-hypothesis research policies described below. Each research policy needs its own repeated-seed benchmark and ablation before performance claims can be made.

We also ran a 97-parameter, $10\to8\to1$ tanh MLP weight-search task on scikit-learn's diabetes dataset (442 rows, 10 features). For this task only, the 97 decision variables are all network weights and biases, decoded from $[0,1]$ to $[-3,3]$. A fixed shuffled 75/25 split (random state 2025) was used; feature and target scalers were fit on the training partition only. Optimizers maximized negative training MSE. Test MSE was computed once for the best training-score point and was never used to select proposals. Test MSE is in training-standardized target units.

Each method received the same 40 sequential calls on each of 20 seeds (0--19): AHCO-v4 (`kmax=14`, batch size 1), uniform random search, Optuna TPE (10 startup trials), and Nevergrad OnePlusOne (mutation sigma 0.25). The latter settings were fixed rather than tuned per task. The reported entries are medians over seeds with 95% percentile-bootstrap confidence intervals for the median (10,000 resamples). For the analytic tasks the metric is best objective score; for diabetes the table gives both best training score and held-out MSE. Parentheses are the bootstrap confidence intervals. The run used Windows 10, Python 3.11.0, Rust 1.89.0, AHCO 0.1.1, NumPy 1.26.4, SciPy 1.14.1, scikit-learn 1.6.0, Optuna 5.0.0, and Nevergrad 1.0.12.

| Task | AHCO | Random | Optuna TPE | Nevergrad OnePlusOne |
|---|---:|---:|---:|---:|
| Sphere (97 active) | $-0.0526$ ($[-0.0594,-0.0463]$) | $-0.0941$ ($[-0.0974,-0.0897]$) | $-0.0774$ ($[-0.0816,-0.0724]$) | $-0.0952$ ($[-0.1006,-0.0896]$) |
| Sparse Sphere (10/97 active) | $-0.0196$ ($[-0.0283,-0.0160]$) | $-0.0401$ ($[-0.0474,-0.0348]$) | $-0.0216$ ($[-0.0282,-0.0150]$) | $-0.0296$ ($[-0.0354,-0.0206]$) |
| Ellipsoid (97 active) | $-0.0404$ ($[-0.0489,-0.0350]$) | $-0.0662$ ($[-0.0704,-0.0625]$) | $-0.0439$ ($[-0.0496,-0.0381]$) | $-0.0393$ ($[-0.0452,-0.0374]$) |
| Rosenbrock (97 active) | $-148.39$ ($[-162.64,-113.64]$) | $-323.05$ ($[-332.84,-311.19]$) | $-236.69$ ($[-248.61,-222.69]$) | $-305.83$ ($[-346.07,-292.48]$) |
| Diabetes MLP: training score | $-3.93$ ($[-5.78,-2.99]$) | $-7.70$ ($[-9.37,-6.69]$) | $-4.89$ ($[-6.20,-4.08]$) | $-7.45$ ($[-8.38,-6.10]$) |
| Diabetes MLP: held-out MSE | $4.07$ ($[3.68,5.79]$) | $7.60$ ($[6.23,9.56]$) | $5.46$ ($[4.33,6.99]$) | $6.96$ ($[4.84,8.61]$) |

Unpaired two-sided Mann-Whitney U tests compared every method pair, with Holm correction across all 36 task/metric comparisons; 95% bootstrap intervals for differences of medians were calculated from independent samples. AHCO's score distribution was higher than random search on all four analytic tasks after correction. Against TPE, the difference favored AHCO on Sphere and Rosenbrock, but was not significant on sparse Sphere or Ellipsoid. Against OnePlusOne, the difference favored AHCO on Sphere and Rosenbrock, but not on sparse Sphere or Ellipsoid; OnePlusOne's Ellipsoid median was slightly better. On the fixed diabetes split, AHCO's training and test medians were better than random search, but corrected pairwise tests did not distinguish AHCO from either TPE or OnePlusOne on either metric. These results support task-specific promise under this configuration, not general superiority.

The complete experiment is reproducible with [`benchmarks/README.md`](./benchmarks/README.md) and [`benchmarks/requirements.txt`](./benchmarks/requirements.txt). The run preserves per-seed outcomes, best-so-far curves, every evaluated point, confidence intervals, pairwise tests, environment versions, data indices and hashes in [`benchmarks/results/main-97d-40eval-20seeds-final/`](./benchmarks/results/main-97d-40eval-20seeds-final/). All 400 runs used exactly 40 calls; an independent audit re-evaluated all 16,000 saved points and reproduced their recorded objective scores.

The diabetes task is one small fixed-split real-data demonstration, not broad evidence of generalization. The earlier notebook comparisons remain exploratory: their one-off outputs, including hand-written control simulators and the California Housing runs, are not used as evidence in this evaluation. We do not report wall-clock superiority, and these results do not establish performance on physical systems or across arbitrary black-box objectives.

### Adaptive grouped-hypothesis prototype

We separately evaluated the grouped-hypothesis policy with the same 40-call budget and seeds 0--19. At 97 dimensions it was compared with saved runs from fixed-allocation AHCO-v4, the proxy-star-only research prototype, Random, TPE, and Nevergrad. The analytic tasks and diabetes split are identical to the earlier experiment.

| 97D task | Fixed AHCO-v4 | Proxy-star | Adaptive grouped hypothesis |
|---|---:|---:|---:|
| Sphere | -0.0526 | -0.0492 | **-0.0486** |
| Sparse Sphere | **-0.0196** | -0.0212 | -0.0243 |
| Ellipsoid | -0.0404 | **-0.0337** | -0.0360 |
| Rosenbrock | -148.39 | **-93.12** | -109.33 |
| Diabetes MLP: training score | -3.93 | **-3.39** | -4.39 |

These are medians over 20 seeds; higher scores are better. No grouped-versus-fixed or grouped-versus-proxy difference remained significant after Holm correction across the 25 comparisons. The grouped policy therefore did not establish a general improvement over the previous AHCO variants at 97D.

To probe the intended high-dimensional use case, we also ran 1,000D Sphere and 10-active-coordinate Sparse Sphere, again with 40 calls and 20 seeds. The table compares medians against methods run on the same tasks and seeds.

| 1,000D task | Adaptive grouped hypothesis | Random | Optuna TPE | Nevergrad OnePlusOne |
|---|---:|---:|---:|---:|
| Sphere | **-0.0794** | -0.1146 | -0.1011 | -0.1172 |
| Sparse Sphere | -0.0308 | -0.0375 | **-0.0260** | -0.0272 |

On Sphere, grouped search scored better than all three baselines after Holm correction across the six comparisons. On sparse Sphere, it beat Random but was not distinguishable from TPE or Nevergrad. Thus the initial 1,000D results are promising for dense objectives but do not yet show that grouped hypotheses reliably identify sparse relevant dimensions. Median AHCO runtime was 0.48 seconds per Sphere run and 0.26 seconds per sparse run on the recorded Windows system; TPE was substantially slower here, while Random and Nevergrad were faster.

The runs, points, trajectories, metadata, source hashes, and summaries are available in [`benchmarks/results/adaptive-grouped-97d-40eval-20seeds/`](./benchmarks/results/adaptive-grouped-97d-40eval-20seeds/) and [`benchmarks/results/adaptive-grouped-1000d-40eval-20seeds/`](./benchmarks/results/adaptive-grouped-1000d-40eval-20seeds/). All 10,400 saved evaluations were audited against the objective definitions and stayed within bounds. These are prototype results, not evidence that the full global space is understood; group-level shared credit can conflate correlated dimensions, and broader task suites and ablations are needed.

# Software design

## Overview

A run proceeds from a small bootstrap into an evidence-driven sequential loop:

1. **Grouped bootstrap**: up to six points where randomized groups of real coordinates move together.
2. **Evidence-driven loop**: estimate coordinate influence and uncertainty over the full space, infer and prune signed pair hypotheses, derive proxy-star hypotheses from two-edge paths, then let a bandit select among six proposal strategies. One strategy samples overlapping groups at several scales and uses a group-specific surrogate to rank star, shared-value, and mixed-value patterns. Refresh active coordinates when an evaluation improves the best score or a trust-region restart occurs.

The total evaluation budget remains a hard cap, but there is no fixed exploration/integration allocation or budget-percentage refresh schedule. The bootstrap only supplies initial observations; all later evaluations are selected sequentially from the evidence available so far. Group memberships are resampled, and per-coordinate evidence balances uniform coverage, observed influence, uncertainty, and prior group successes. A group improvement is shared credit for its changed dimensions, so it is a useful prioritization signal but not a causal attribution. Hypothesis confidence is a heuristic consistency score, not a calibrated probability or proof of a global optimum.

## Scores, constraints and categorical coordinates

Let $f$ be the objective and $B$ the budget. A run makes exactly $B$ objective calls. A non-finite value becomes $-10^{100}$ (assuming ordinary scores exceed it), a NaN violation becomes a large finite violation, and an objective error stops the run. Proposals lie in $[0,1]^d$.

A categorical coordinate with $r$ levels is snapped to the cell centre $(k+1/2)/r$. Grouped bootstrap draws categorical levels uniformly and snaps all designs; every proposal is snapped before use.

A constraint reports a violation $v(x)$, with $v(x)\leq0$ feasible. Two sources are supported and may be combined:

- A **known constraint** is a cheap function of $x$. It never consumes objective budget: bootstrap designs make up to 40 draws per point to find a feasible one, and every candidate pool is filtered to feasible candidates (or, if none is feasible, the least-violating ones). The inactive-coordinate strategy retries up to ten times and may still return an infeasible point. Known violations are used only for filtering and are not stored with observations.
- An **objective-reported violation** is returned with the value (`Evaluation`) and stored with the observation. It affects ranking after the fact.

Ranking uses an effective score. With `FeasibilityFirst`, all feasible observations rank above infeasible ones, and infeasible ones are ordered by violation. With `Penalty { weight }` (the default, weight 5), the effective score is $f(x)-\mathrm{weight}\cdot\max(v(x),0)$; the weight is scale-dependent and should be set for the objective at hand. The returned point is the best by effective score. The `feasible` flag says whether it is feasible, and the history records the best feasible value (the sentinel $-10^{100}$ until one is found).

## Initial design and screening

For each bootstrap point AHCO draws a group count $g=\mathrm{clip}(\mathrm{round}(d^{U}),1,d)$ with $U\sim\mathrm{Uniform}[0.4,0.9)$, draws one centre per group uniformly on $[0.05,0.95]$, assigns each real coordinate to a random group, adds $\mathcal N(0,0.04^2)$ noise and clips to $[0,1]$. Categorical coordinates are drawn independently and uniformly instead, since grouping unordered levels has no meaning. The bootstrap size is $\min(B,6)$; it does not reserve a separate integration budget.

Scores are converted to ranks in $[-1,1]$ (ties follow observation order). For a real coordinate with $z_j=2x_j-1$, the effect score is the larger of the absolute correlations of $z_j$ and of $z_j^2-1/3$ with the ranks. For a categorical coordinate it is $\eta=\sqrt{SS_{\mathrm{between}}/SS_{\mathrm{total}}}$ over the rank scores grouped by level. The coordinate score adds (a) the fraction of 24 random half-samples in which the *linear* correlation magnitude (or $\eta$ for categoricals) is at least $0.22$, and (b) normalized split-gain importance from a 35-tree forest on the encoded data, summed over one-hot columns per coordinate. Active coordinates from an earlier selection get a factor $1.15$ at re-selection. The active set has size $m=\min(d,\max(k_{\max},\min(d,3)))$ with $k_{\max}=14$ by default. This heuristic is not an error-controlled stability-selection procedure.

For coordinates left inactive, AHCO also records a guess at a good value: for a real coordinate, the peak of a quadratic regression of the ranks on $z_j$ and $z_j^2-1/3$; for a categorical coordinate, the level with the highest mean rank. The active set is re-estimated after a best-score improvement and at a trust-region restart, retaining a mild preference for previously active coordinates. The group strategy is not limited to this active set: it scores all dimensions and fits a temporary surrogate only on each sampled group.

## Interaction blocks

For each real active pair, AHCO correlates $z_jz_k$ with the rank scores. A pair is retained when the absolute correlation is at least $0.15$ and its sign agrees in at least 70% of 24 random half-samples. Its heuristic confidence is sign stability multiplied by $\min(|r|/0.4,1)$. Two retained relationships sharing a mediator create a proxy-star hypothesis between their other endpoints; its direction is the product of the two signs and its confidence is the product of their confidences. Proxy hypotheses below confidence $0.20$ are pruned, and the strongest hypothesis is retained for each endpoint pair. Connected components of retained direct and proxy hypotheses form proposal blocks. Categorical active coordinates are singleton blocks; categorical interactions are not inferred. These are associations in the observed design, not causal or functional dependence, and the chained direction does not imply equal magnitudes or equal mediator influence.

## Surrogate and acquisition

The surrogate is an ensemble of 45 randomized regression trees fitted to all observations, using the active coordinates (categoricals one-hot encoded) and the rank scores. Each split considers $\lceil\sqrt{c}\rceil$ random columns, where $c$ is the number of encoded columns, samples six uniform thresholds per column within the observed range, and keeps the split with the largest squared-error reduction provided both children have at least two observations. Nodes are leaves with fewer than five observations, at depth eight, or without a positive-gain split. This relates to randomized tree ensembles [@Geurts2006] without claiming equivalence to any named implementation. The surrogate is refitted about every two evaluations and whenever the active set changes.

With $T_t(x)$ the prediction of tree $t$, AHCO computes the mean $\hat\mu(x)$ and population standard deviation $\hat\sigma(x)$ across the 45 trees, and ranks candidates by $\hat\mu(x)+1.4\hat\sigma(x)$. This is an exploration heuristic based on tree disagreement, not a posterior standard deviation or confidence bound; its coefficient is not scheduled by elapsed budget.

## Proposal strategies

Let $\mathrm{TR}$ be the trust-region width, $x_c$ the centre (the best point unless a restart has chosen another), and $x^*$ the best observed point. Every candidate is snapped and constraint-filtered before scoring. The five strategies are:

1. **UCB.** 600 perturbations of $x_c$ (each active coordinate changed with probability $\mathrm{clip}(4/m,0.25,1)$, at least one forced; real coordinates drawn uniformly in $[x_c-\mathrm{TR}/2,x_c+\mathrm{TR}/2]$ clipped to $[0,1]$, categorical coordinates redrawn uniformly). The highest-scoring candidate is chosen.
2. **Block.** One random block is chosen. 250 candidates copy $x_c$ and set the block's coordinates from a point drawn from the best $\min(n,12)$ observations plus $\mathcal N(0,(0.20\min(1,2\mathrm{TR}))^2)$ noise (categorical coordinates take that point's level with probability $0.5$, otherwise a random level). The highest-scoring candidate is chosen.
3. **Direction.** For each real active coordinate a slope $g_j=\mathrm{cov}(x_j-x^*_j,\tilde y)/(\mathrm{var}(x_j-x^*_j)+10^{-4})$ is estimated from the $4m+8$ observations nearest to $x^*$ (distance over active coordinates), where $\tilde y$ are the rank scores. The normalized direction is blended $0.7/0.3$ with the previous successful direction, and the step in $\{0.1,0.25,0.5,0.8,1.2,1.8\}\mathrm{TR}$ with the best surrogate mean is chosen. Categorical coordinates stay at the values of $x^*$.
4. **Inactive.** Starting from $x^*$, change $\min(q',\max(1,\mathrm{round}(0.3q')))$ of the $q'$ inactive coordinates, using the recorded good values with probability $0.6$ (if any selected coordinate has one) and otherwise random values (uniform on $[0.05,0.95]$, or a random level).
5. **Proxy-star.** Sample a retained direct or inferred relationship in proportion to its heuristic confidence. Generate 160 candidates that move the endpoints in the inferred same/opposite directions; inferred hypotheses also perturb their mediator. Endpoint step magnitudes are sampled independently so a chained direction does not force equal influence, and the highest UCB-scoring candidate is evaluated.

Block is available only when at least two blocks exist, Inactive only when some coordinate is inactive, and Proxy-star only when at least one relationship hypothesis remains after pruning.
6. **Adaptive group hypothesis.** Choose a group size uniformly from powers of two up to $\min(d,256)$, plus half-space and full-space groups. Sample dimensions without replacement using a mixture of uniform coverage, full-space univariate evidence, inverse-square-root under-sampling, and a small bonus for past group improvements. Fit a 30-tree surrogate on current observations encoded only on this group. Generate 192 candidates around $x^*$: choose a pivot coordinate and set it to 0, 0.5, or 1; change most other group members either to the pivot value or independently to one of these three values, while holding some members and all out-of-group dimensions fixed. Categorical members may switch to a different level. The group surrogate's UCB selects one candidate. A successful group trial increments success counts for every group member changed from $x^*$; this deliberately cheap shared-credit update does not isolate individual causal effects.

## Adaptation

A strategy is chosen by a Thompson-sampling-style rule: for each available strategy draw $G_w\sim\mathrm{Gamma}(w,1)$ and $G_l\sim\mathrm{Gamma}(l,1)$ and pick the largest $G_w/(G_w+G_l)$. Initially $w=(2,1,1,1,1)$ and $l=(1,1,1,1,1)$. After every evaluation both counts are pulled toward one by a factor $0.96$. An evaluation is an improvement if its effective score exceeds the previous best by more than $10^{-12}$; then the chosen strategy's $w$ increases by $1.2$, otherwise its $l$ increases by $1$.

The trust region starts at $0.45$, is multiplied by $1.5$ (capped at $0.85$) after two consecutive improvements, and divided by $1.8$ (floored at $10^{-3}$) after $\max(2,\lceil m/4\rceil)$ consecutive failures. When it falls to $0.005$ or below, it restarts at $0.30$ around a random one of the six best observations, clears direction memory, and re-estimates active coordinates. Active coordinates are also re-estimated after an improvement. Relationship hypotheses and blocks are reconstructed from the current observations for every proposal; no refresh is tied to a fraction of the remaining budget.

| Quantity | Default |
|---|---|
| Grouped bootstrap points | $\min(B,6)$ |
| Group sizes | powers of two through $\min(d,256)$, plus half-space and full-space |
| Direct relationship filter | $|r|\ge0.15$ and sign stability $\ge0.70$ across 24 half-samples |
| Proxy-star pruning | confidence $\ge0.20$; chained directions compose by sign multiplication |
| Candidates per proposal | 600 for UCB; 250 for Block; 192 for grouped hypothesis; 160 for Proxy-star; 6 steps for Direction |

## Batch and parallel evaluation

With `batch_size` $q>1$, each round proposes $q$ points one after another from the same fitted surrogate and the same $x^*$ and $x_c$. To keep a batch diverse, a candidate's score is reduced for being close to points already chosen in the batch (a penalty based on distance in the real active coordinates, and an additional one for repeating the same categorical levels nearby). The batch is evaluated by `optimize_parallel` on scoped threads (`workers`, defaulting to $q$) using a dynamic work queue. Results are processed in proposal order, so for a deterministic objective the output does not depend on the number of workers or on thread timing. The bandit and trust region are updated per evaluation in that order. The sequential entry points `optimize` and `optimize_eval` accept stateful objectives and evaluate sequentially even when $q>1$.

Batching reduces wall-clock time only when evaluation latency dominates; this paper makes no claim that a batch is as sample-efficient as sequential evaluation, but in early side by side tests it was found sequential is better than batch evaluation. The Python binding exposes `batch_size` and `levels`, but Python objectives run under the interpreter lock, so they are not evaluated in parallel. Constraint APIs are Rust-only.

## Reproducibility

AHCO uses a fixed seed and the ChaCha8 generator. For a deterministic objective, the same crate version and the same configuration (including batch size) reproduce a run exactly. The Python extension calls the same Rust core.

## Limitations

- Inputs are the unit cube with unordered categorical levels; there are no integer or log-scale types and no noise model.
- Screening relies on correlations and effect sizes. It can miss non-monotone effects that are not roughly quadratic, purely interactive effects, and (in the stability term) interior optima, which the linear correlation does not capture. Interactions involving categorical coordinates are not detected.
- The relationship thresholds and strategy settings are heuristics chosen by hand and have not been tuned or ablated; the proxy-star policy has not yet been benchmarked.
- The grouped strategy's shared credit can confuse correlated dimensions, and its temporary group surrogate can be unreliable with few observations. Parallel objective batches and tests on substantially higher-dimensional problems remain future work.
- Rank-transformed scores discard magnitude information.
- Sequential runs apply no penalty against recently evaluated points. Preliminary checks suggest sequential runs can stagnate late in a run, so results should be validated per problem.
- `Penalty` weights depend on the scale of the objective, and the sentinel $-10^{100}$ assumes ordinary scores exceed it.

# Research impact statement

AHCO is at an early stage. The Rust library and Python extension are published as `ahco-rust` on PyPI (version [VERSION]), with source at <https://github.com/eaheog-DecisionScience/AHCO-rust> and the distribution at <https://pypi.org/project/ahco-rust/>. The Rust test suite checks exact budget use, bounded results and monotone best-so-far history, repeatability under a fixed seed, non-finite scores, invalid configurations, categorical snapping, and constraint handling. The integration tests also exercise mixed categorical search, objective errors, and a fixed-seed eight-run continuous search. The reproducible benchmark suite described above records all per-evaluation points and scores.

No external deployment or publication is documented here. The current benchmark supplies repeated-seed evidence for a small fixed suite, not a guarantee of general accuracy or speed superiority. Independent real-world datasets and splits, ablations, and broader baselines remain important next steps.

# AI usage disclosure

AI (Large Language Models) primarily Microsoft Copilot is used as an equal assistant in code editor from templates telesense recommendations while Antrophic Claude Sonnet 5.5 is used as a critic for codebase and stress testing while AIs role was from an assitant perspective and all the AI generated texts or code snippets are reviewed.

# Acknowledgements

No external funding was received for this work.

# References
