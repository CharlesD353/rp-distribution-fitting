# rp-distribution-fitting

Streamlit app and utilities for fitting parametric distributions to percentile
constraints and analyzing risk‑adjusted expected values.

## Introduction

This project is a lightweight toolkit for turning expert‑elicited percentiles
into full probability distributions, then examining how different risk
preferences change expected value.

Key ideas:
- You provide a set of percentile constraints (e.g., p10, p50, p90).
- The app fits multiple parametric distributions to those constraints.
- You can compare fits, visualize PDFs/CDFs, and explore sensitivity.
- Risk adjustments (upside skepticism, downside protection) let you model
  conservative or loss‑averse decision rules.

The fitting logic lives in
`rp-distribution-fitting/distributions.py`, risk
adjustments in `rp-distribution-fitting/risk_analysis.py`,
and the Streamlit UI in `rp-distribution-fitting/app.py`.

## How Fitting Connects to Risk Adjustments

The workflow is a two‑step pipeline:
1. **Fit distributions** to your percentile constraints. Each fitted model
   provides a full PDF/CDF/PPF and summary stats.
2. **Apply risk adjustments** to the fitted distributions. The adjustments are
   computed by integrating over the fitted PDF, so risk results are entirely
   determined by the fitted distribution.

Concretely:
- Fitting produces a `FitResult` with a frozen distribution object (PDF/PPF).
- Risk functions call that PDF to compute expectations.
- If the fitted distribution changes (different percentiles, different family,
  or a different best‑fit), the risk‑adjusted values will change accordingly.

The main risk adjustments are:
- **Risk neutral EV**: standard expected value over the fitted distribution.
- **Upside skepticism**: truncate the upper tail at a percentile and
  renormalize.
- **Downside protection**: apply a loss‑averse utility around a reference point.
- **Combined**: truncation plus loss‑averse utility.

If you want to change how risk interacts with the fit, the entry points are
`rp-distribution-fitting/risk_analysis.py` and the bounds
or truncation parameters in `rp-distribution-fitting/app.py`.

## Formal Risk Aversion Models (Duffy 2023)

In addition to the informal adjustments above, the app implements the three
formal risk aversion models from Laura Duffy's Rethink Priorities paper
["How Can Risk Aversion Affect Your Cause Prioritization?"](https://rethinkpriorities.org/research-area/how-can-risk-aversion-affect-your-cause-prioritization/)
(2023). These provide academically grounded, parameterised alternatives to the
informal heuristics.

All three models share a common sampling approach: 10,000 deterministic quantile
samples are drawn from the fitted distribution via its inverse CDF (PPF) at
evenly spaced probability points. This avoids Monte Carlo noise and ensures
reproducible results.

### DMREU — Difference-Making Risk-Weighted Expected Utility

From Duffy (2023), p. 35. Outcomes are sorted worst-to-best and assigned
probability-weighted decision weights via a power function.

**Formula:**

    DMREU(A) = Σ dᵢ · [m(P(i)) − m(P(i+1))]

where:
- `dᵢ` are outcomes sorted worst-to-best
- `P(i) = 1 − i/N` is the cumulative probability of making a difference at
  least as good as outcome i
- `m(P) = Pᵃ` is the risk-weighting function

**Parameter: `p`** (thought-experiment probability, as in Table 12, p. 68)

The parameter `p` represents the answer to a calibration question: "What
probability of saving 1,000 lives would make you indifferent to saving 10 lives
for certain?" It is converted to the power exponent via `a = −2 / log₁₀(p)`.

| p    | Exponent a | Interpretation         |
|------|-----------|------------------------|
| 0.01 | 1.0       | Risk-neutral           |
| 0.05 | ~1.54     | Moderate risk aversion |
| 0.10 | 2.0       | High risk aversion     |

**Slider range:** 0.01 to 0.10 (step 0.01), default 0.01 (risk-neutral).

### WLU — Weighted Linear Utility

From Duffy (2023), pp. 39–42. Applies stakes-sensitive weights that
down-weight large positive outcomes and up-weight large negative outcomes.

**Formula:**

    WLU(A) = (1/N) · Σ ŵᵢ · xᵢ

where the weighting function is:
- `w(x; c) = 1 / (1 + xᶜ)` for x ≥ 0
- `w(x; c) = 2 − 1 / (1 + (−x)ᶜ)` for x < 0

and `ŵᵢ = w(xᵢ) / mean(w)` are the normalised weights.

**Parameter: `c`** (concavity, as in Tables 13–14, pp. 70–71)

| c    | Interpretation         |
|------|------------------------|
| 0.00 | Risk-neutral (all weights equal) |
| 0.05 | Low risk aversion      |
| 0.25 | High risk aversion     |

**Slider range:** 0.00 to 0.25 (step 0.01), default 0.00 (risk-neutral).

### Ambiguity Aversion — Expected Difference Made

From Duffy (2023), pp. 42–45. Applies a cubic weighting function to
rank-ordered outcomes, overweighting worst-ranked and underweighting
best-ranked.

**Implementation note:** In the paper, ambiguity aversion is a second-order
model — it aggregates across multiple expected-utility estimates (e.g., from
different moral or empirical models) under model uncertainty. Our implementation
applies the same cubic weighting shape as a single-distribution proxy: outcomes
from one fitted distribution are rank-ordered and reweighted, capturing the
directional intent (be more conservative when uncertain) without requiring a
set of competing models. This is a pragmatic simplification; the weighting
function and parameter range (k = 0–8) match the paper exactly.

**Formula:**

    w(i) = (1/N) · (−k · (i/(N−1) − 0.5)³ + 1)

where outcomes are sorted worst-to-best and `i/(N−1)` is the rank fraction
(0 = worst, 1 = best).

**Parameter: `k`** (ambiguity strength, as in Table 15, p. 74)

| k   | Weight range | Interpretation           |
|-----|-------------|--------------------------|
| 0   | [1, 1]      | Ambiguity-neutral        |
| 4   | [0.5, 1.5]  | Mild ambiguity aversion  |
| 8   | [0, 2]      | Strong ambiguity aversion|

**Slider range:** 0.0 to 8.0 (step 0.5), default 0.0 (ambiguity-neutral).

### Using the formal models

In the Streamlit app, the formal model parameters appear in the sidebar under
**Formal Risk Models (Duffy 2023)**. The **Formal Risk Models** tab shows:

1. A summary table comparing risk-neutral EV with all three model outputs
2. A grouped bar chart for visual comparison across distributions
3. Sensitivity analysis charts showing how each model responds as its parameter
   varies across the full range (computed for the best-fit distribution)
4. Expandable explanations of each model

Set all parameters to their defaults (p=0.01, c=0.0, k=0.0) to recover the
risk-neutral expected value. Increasing any parameter makes the model more
conservative (lower expected value for right-skewed distributions).

The formal models can also be used programmatically:

```python
from distributions import fit_distribution
from risk_analysis import compute_dmreu, compute_wlu, compute_ambiguity_aversion

fit = fit_distribution("normal", {0.10: -30, 0.50: 10, 0.90: 50})

# DMREU with moderate risk aversion
dmreu_val = compute_dmreu(fit, p=0.05)

# WLU with low concavity
wlu_val = compute_wlu(fit, c=0.05)

# Ambiguity aversion, mild
aa_val = compute_ambiguity_aversion(fit, k=4.0)
```

Or via the high-level `analyze()` function with `RiskParams`:

```python
from risk_analysis import analyze, RiskParams

params = RiskParams(dmreu_p=0.05, wlu_c=0.10, ambiguity_k=4.0)
result = analyze(fit, params)

print(result.dmreu_ev, result.wlu_ev, result.ambiguity_aversion_ev)
```

## Percentile CSV Export (p1 to p99)

The app now supports downloading a CSV with 1st–99th percentile rows for every
fitted distribution, including EV/EU fields.

- In the **Risk Adjustments** tab, click
  **Download 1-99 percentile EV/EU CSV**.
- The export includes one row per distribution × percentile (`p1` ... `p99`).
- Key columns include:
  - `ev_percentile_value` (raw fitted percentile outcome)
  - `eu_percentile_value` (loss-averse utility transform at that percentile)
  - summary EV/EU columns (e.g. `risk_neutral_ev`, `downside_protection_eu`,
    `combined_eu`)

Programmatic helpers:

```python
from distributions import percentile_table, percentile_table_all
from risk_analysis import ev_eu_percentile_table, ev_eu_percentile_table_all

# p1..p99 outcome values for one fit
df_pct = percentile_table(fit)

# p1..p99 outcome values for all fits
df_pct_all = percentile_table_all(fits)

# p1..p99 EV/EU table for one fit
df_ev_eu = ev_eu_percentile_table(fit, params)

# p1..p99 EV/EU table for all fits
df_ev_eu_all = ev_eu_percentile_table_all(fits, params)
```

## Why Both EV and EU?

They answer different decision questions:
- **EV (Expected Value)** is the baseline, risk‑neutral expectation. It is the
  right metric if you treat gains and losses symmetrically and care only about
  the long‑run average outcome.
- **EU (Expected Utility)** incorporates risk preferences. For example, EU may reflect
   *loss aversion* by weighting negative outcomes more heavily than
  equally sized gains, while EV will ignore this.

## How Many Percentiles Do You Need?

A practical rule is: **use at least as many percentiles as the distribution’s
parameter count**, and **more is better**. Extra points turn the fit into an
over‑determined problem, which makes the optimizer choose the closest overall
shape rather than matching just a few points exactly.

Heuristic guidance by distribution:
- **Normal (2 params)**: 2 points can work (e.g., p50 + p10/p90 for spread), but 3–4 improves stability.
- **Lognormal (3 params)**: 3 points is the minimum; 4–5 points is better. All values must be positive.
- **Skew‑normal (3 params)**: needs 3 points; 4–5 points helps identify skew.
- **Student’s t (3 params)**: 3 points minimum; 4–5 helps pin down tail thickness.
- **GEV (3 params)**: 3 points minimum; 4–5 helps identify tail type.
- **Log‑Student’s t (3 params)**: 3 points minimum; 4–5 preferred. All values must be positive.

If you only have 2 points, the app will still run, but only 2‑parameter
families (like normal) can be strongly identified. In practice, **3–5
percentiles (e.g., p10, p25, p50, p75, p90)** is a good default for stable fits.

## Fast Mode (for tests or quick iteration)

Optimization can be slow. You can enable a faster, lower‑fidelity fit mode.

### Pytest

```bash
pytest -q --fast
```

### Environment Variable

```bash
RP_FAST=1 pytest -q
```

Fast mode reduces optimizer iterations, the number of methods tried, and the
number of random restarts. Expect less accurate fits, but much faster runtimes.
