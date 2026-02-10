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
adjustments in `rp-distribution-fitting/risk.py`,
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
`rp-distribution-fitting/risk.py` and the bounds
or truncation parameters in `rp-distribution-fitting/app.py`.

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
