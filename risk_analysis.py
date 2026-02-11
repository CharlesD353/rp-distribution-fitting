"""
Risk adjustment computations.

Takes fitted distributions and computes expected values under different
risk preferences:

Informal adjustments:
  - Risk neutral: standard EV
  - Upside skepticism: truncate at a high percentile, renormalize
  - Downside protection: loss-averse utility function
  - Combined: both truncation and loss aversion

Formal models (Duffy 2023, Rethink Priorities):
  - DMREU: Difference-Making Risk-Weighted Expected Utility
  - WLU: Weighted Linear Utility
  - Ambiguity Aversion: Expected Difference Made with ambiguity weighting
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd
from scipy import integrate

from distributions import FitResult, percentile_table


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class RiskParams:
    """Configuration for risk adjustments."""
    # Informal model parameters
    truncation_percentile: float = 0.99
    loss_aversion_lambda: float = 2.5
    reference_point: float = 0.0
    integration_bounds_q: tuple[float, float] = (0.0001, 0.9999)
    # Formal model parameters (Duffy 2023)
    dmreu_p: float = 0.01       # thought-experiment probability; 0.01 = risk-neutral
    wlu_c: float = 0.0          # concavity; 0 = risk-neutral
    ambiguity_k: float = 0.0    # cubic coefficient; 0 = ambiguity-neutral
    n_samples: int = 10_000     # sample count for formal models


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class RiskResult:
    """Risk-adjusted expected values for a single distribution."""
    distribution_name: str
    # Informal adjustments
    risk_neutral_ev: float
    upside_skepticism_ev: float
    downside_protection_eu: float
    combined_eu: float
    # Formal models (Duffy 2023)
    dmreu_ev: float
    wlu_ev: float
    ambiguity_aversion_ev: float
    params: RiskParams

    def to_dict(self) -> dict:
        return {
            "distribution": self.distribution_name,
            "risk_neutral_ev": self.risk_neutral_ev,
            "upside_skepticism_ev": self.upside_skepticism_ev,
            "downside_protection_eu": self.downside_protection_eu,
            "combined_eu": self.combined_eu,
            "dmreu_ev": self.dmreu_ev,
            "wlu_ev": self.wlu_ev,
            "ambiguity_aversion_ev": self.ambiguity_aversion_ev,
        }


# ---------------------------------------------------------------------------
# Utility function
# ---------------------------------------------------------------------------

def _loss_aversion_utility(x: float, reference: float, lam: float) -> float:
    """Piecewise linear utility (Kahneman-Tversky style).

    Above reference: u = x - reference  (gains at face value)
    Below reference: u = lambda * (x - reference)  (losses amplified)
    """
    gain = x - reference
    return gain if gain >= 0 else lam * gain


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------

def compute_risk_neutral(
    fit: FitResult,
    bounds_q: tuple[float, float] = (0.0001, 0.9999),
) -> float:
    """Standard expected value: E[X] = integral(x * f(x) dx)."""
    lb = fit.ppf(bounds_q[0])
    ub = fit.ppf(bounds_q[1])
    result, _ = integrate.quad(lambda x: x * fit.pdf(x), lb, ub)
    return result


def compute_upside_skepticism(
    fit: FitResult,
    truncation_percentile: float = 0.99,
    bounds_q: tuple[float, float] = (0.0001, 0.9999),
) -> float:
    """EV after truncating the distribution at the given percentile.

    E[X | X <= ppf(trunc)] = integral(x * f(x), lb, trunc_val) / trunc_pct
    """
    lower_q, upper_q = bounds_q
    trunc_q = min(truncation_percentile, upper_q)
    if trunc_q <= lower_q:
        raise ValueError("truncation_percentile must be greater than bounds_q[0].")
    lb = fit.ppf(lower_q)
    trunc_val = fit.ppf(trunc_q)
    numerator, _ = integrate.quad(lambda x: x * fit.pdf(x), lb, trunc_val)
    divisor = float(fit.cdf(trunc_val) - fit.cdf(lb))
    return numerator / divisor


def compute_downside_protection(
    fit: FitResult,
    loss_aversion_lambda: float = 2.5,
    reference_point: float = 0.0,
    bounds_q: tuple[float, float] = (0.0001, 0.9999),
) -> float:
    """Expected utility with piecewise linear loss aversion.

    EU = integral(u(x) * f(x) dx) where u is the loss-averse utility.
    The result is in "utility units" — comparable across distributions
    but shifted relative to raw EV.
    """
    lb = fit.ppf(bounds_q[0])
    ub = fit.ppf(bounds_q[1])
    result, _ = integrate.quad(
        lambda x: _loss_aversion_utility(x, reference_point, loss_aversion_lambda) * fit.pdf(x),
        lb,
        ub,
    )
    # Add back reference point so the result is in the same units as EV
    return result + reference_point


def compute_combined(
    fit: FitResult,
    truncation_percentile: float = 0.99,
    loss_aversion_lambda: float = 2.5,
    reference_point: float = 0.0,
    bounds_q: tuple[float, float] = (0.0001, 0.9999),
) -> float:
    """Both truncation AND loss aversion applied together.

    EU = integral(u(x) * f(x), lb, trunc_val) / trunc_pct + reference_point
    """
    lower_q, upper_q = bounds_q
    trunc_q = min(truncation_percentile, upper_q)
    if trunc_q <= lower_q:
        raise ValueError("truncation_percentile must be greater than bounds_q[0].")
    lb = fit.ppf(lower_q)
    trunc_val = fit.ppf(trunc_q)
    numerator, _ = integrate.quad(
        lambda x: _loss_aversion_utility(x, reference_point, loss_aversion_lambda) * fit.pdf(x),
        lb,
        trunc_val,
    )
    divisor = float(fit.cdf(trunc_val) - fit.cdf(lb))
    return numerator / divisor + reference_point


# ---------------------------------------------------------------------------
# Shared sampling for formal models
# ---------------------------------------------------------------------------

def _generate_samples(
    fit: FitResult,
    n_samples: int = 10_000,
    bounds_q: tuple[float, float] = (0.0001, 0.9999),
) -> np.ndarray:
    """Generate evenly-spaced quantile samples from a fitted distribution.

    Deterministic (no randomness): draws at uniform quantile spacing via ppf.
    """
    quantile_points = np.linspace(bounds_q[0], bounds_q[1], n_samples)
    return fit.ppf(quantile_points)


# ---------------------------------------------------------------------------
# Formal models (Duffy 2023)
# ---------------------------------------------------------------------------

def compute_dmreu(
    fit: FitResult,
    p: float = 0.01,
    n_samples: int = 10_000,
    bounds_q: tuple[float, float] = (0.0001, 0.9999),
) -> float:
    """Difference-Making Risk-Weighted Expected Utility (Duffy 2023, p.35).

    DMREU(A) = Σ d_i * [m(P(i)) - m(P(i+1))]

    where d_i are outcomes sorted worst-to-best, P(i) = 1 - i/N is the
    cumulative probability of making a difference at least as good, and
    m(P) = P^a is the risk-weighting function.

    The parameter p is the thought-experiment probability: "What chance of
    saving 1000 lives would make you indifferent to saving 10 for certain?"
    Converted to exponent via a = -2 / log10(p).

    p = 0.01 → a = 1.0 (risk-neutral)
    p = 0.05 → a ≈ 1.54 (moderate risk aversion)
    p = 0.10 → a = 2.0 (high risk aversion)
    """
    # Convert thought-experiment probability to power exponent
    if p <= 0 or p >= 1:
        raise ValueError(f"DMREU p must be in (0, 1), got {p}")
    a = -2.0 / math.log10(p)

    samples = _generate_samples(fit, n_samples, bounds_q)
    d = np.sort(samples)  # worst to best
    N = len(d)

    # P(i) = 1 - i/N for i = 0, ..., N; P(N) = 0
    P = 1.0 - np.arange(N + 1) / N
    m_P = np.power(P, a)
    # Weight for outcome i: m(P(i)) - m(P(i+1))
    weights = m_P[:-1] - m_P[1:]

    return float(np.dot(d, weights))


def compute_wlu(
    fit: FitResult,
    c: float = 0.0,
    n_samples: int = 10_000,
    bounds_q: tuple[float, float] = (0.0001, 0.9999),
) -> float:
    """Weighted Linear Utility (Duffy 2023, pp.39-42).

    WLU(A) = (1/N) * Σ ŵ_i * x_i

    where the weighting function is:
      w(x; c) = 1 / (1 + x^c)        for x ≥ 0
      w(x; c) = 2 - 1 / (1 + (-x)^c) for x < 0
    and ŵ_i = w(x_i) / mean(w(x_j)).

    Worse outcomes receive higher weight; better outcomes get lower weight.

    c = 0: risk-neutral (all weights equal after normalization)
    c = 0.05: low risk aversion
    c = 0.25: high risk aversion
    """
    samples = _generate_samples(fit, n_samples, bounds_q)

    if c <= 0:
        return float(np.mean(samples))

    abs_samples = np.abs(samples)
    # Guard against underflow/overflow for very large values
    powered = np.power(np.clip(abs_samples, 0, 1e15), c)

    w_positive = 1.0 / (1.0 + powered)        # for x >= 0
    w_negative = 2.0 - 1.0 / (1.0 + powered)  # for x < 0

    weights = np.where(samples >= 0, w_positive, w_negative)

    w_mean = np.mean(weights)
    if w_mean <= 0:
        return float(np.mean(samples))  # fallback

    w_hat = weights / w_mean
    return float(np.mean(w_hat * samples))


def compute_ambiguity_aversion(
    fit: FitResult,
    k: float = 0.0,
    n_samples: int = 10_000,
    bounds_q: tuple[float, float] = (0.0001, 0.9999),
) -> float:
    """Expected Difference Made with Ambiguity Aversion (Duffy 2023, pp.42-45).

    Sorts outcomes worst-to-best and applies a cubic weighting function:
      w(i) = (1/N) * (-k * (i/(N-1) - 0.5)^3 + 1)

    Overweights worst-ranked outcomes, underweights best-ranked.

    k = 0: ambiguity-neutral (flat weights)
    k = 4: mild ambiguity aversion (paper's f₂, range [0.5, 1.5])
    k = 8: strong ambiguity aversion (paper's f₁, range [0, 2])

    Note: The paper defines this as a second-order model aggregating across
    multiple EU estimates under model uncertainty.  This implementation applies
    the same cubic weighting as a single-distribution proxy (rank-weighting
    first-order outcomes), capturing the directional intent without requiring
    a set of competing models.
    """
    samples = _generate_samples(fit, n_samples, bounds_q)

    if k == 0.0:
        return float(np.mean(samples))

    d = np.sort(samples)  # worst to best
    N = len(d)

    rank_fractions = np.arange(N) / (N - 1)  # 0 = worst, 1 = best
    weights = -k * (rank_fractions - 0.5) ** 3 + 1.0
    # Clamp to non-negative (defensive, only matters if k > 8)
    weights = np.maximum(weights, 0.0)
    # Normalize so weights sum to N (weighted mean = dot(w, d) / N)
    w_sum = np.sum(weights)
    if w_sum <= 0:
        return float(np.mean(samples))  # fallback
    weights = weights * (N / w_sum)

    return float(np.mean(weights * d))


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def analyze(
    fit: FitResult,
    params: RiskParams | None = None,
) -> RiskResult:
    """Compute all risk adjustments for a single fitted distribution."""
    p = params or RiskParams()
    return RiskResult(
        distribution_name=fit.name,
        risk_neutral_ev=compute_risk_neutral(fit, p.integration_bounds_q),
        upside_skepticism_ev=compute_upside_skepticism(
            fit, p.truncation_percentile, p.integration_bounds_q
        ),
        downside_protection_eu=compute_downside_protection(
            fit, p.loss_aversion_lambda, p.reference_point, p.integration_bounds_q
        ),
        combined_eu=compute_combined(
            fit,
            p.truncation_percentile,
            p.loss_aversion_lambda,
            p.reference_point,
            p.integration_bounds_q,
        ),
        dmreu_ev=compute_dmreu(
            fit, p.dmreu_p, p.n_samples, p.integration_bounds_q
        ),
        wlu_ev=compute_wlu(
            fit, p.wlu_c, p.n_samples, p.integration_bounds_q
        ),
        ambiguity_aversion_ev=compute_ambiguity_aversion(
            fit, p.ambiguity_k, p.n_samples, p.integration_bounds_q
        ),
        params=p,
    )


def analyze_all(
    fits: list[FitResult],
    params: RiskParams | None = None,
) -> pd.DataFrame:
    """Compute risk adjustments for all fitted distributions.

    Returns a DataFrame with columns:
        distribution, risk_neutral_ev, upside_skepticism_ev,
        downside_protection_eu, combined_eu, dmreu_ev, wlu_ev,
        ambiguity_aversion_ev, fit_error
    """
    rows = []
    for fit in fits:
        result = analyze(fit, params)
        row = result.to_dict()
        row["fit_error"] = fit.error
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Percentile EV/EU export helpers
# ---------------------------------------------------------------------------

def ev_eu_percentile_table(
    fit: FitResult,
    params: RiskParams | None = None,
    percentile_points: list[int] | None = None,
    precomputed: RiskResult | None = None,
) -> pd.DataFrame:
    """Return p1..p99 (or custom) EV/EU percentile table for one distribution.

    Columns include:
      - ev_percentile_value: raw fitted outcome at percentile p
      - eu_percentile_value: loss-averse utility transform of that outcome
      - risk summary metrics (EV/EU) repeated for easy CSV filtering/grouping

    If *precomputed* is provided, its summary dict is used directly instead
    of calling ``analyze()`` again.
    """
    p = params or RiskParams()
    pct_df = percentile_table(fit, percentile_points).rename(
        columns={"value": "ev_percentile_value"}
    )
    pct_df["eu_percentile_value"] = pct_df["ev_percentile_value"].apply(
        lambda x: _loss_aversion_utility(x, p.reference_point, p.loss_aversion_lambda)
        + p.reference_point
    )

    summary = (precomputed or analyze(fit, p)).to_dict()
    for key, value in summary.items():
        if key != "distribution":
            pct_df[key] = value

    pct_df["fit_error"] = fit.error
    return pct_df


def ev_eu_percentile_table_all(
    fits: list[FitResult],
    params: RiskParams | None = None,
    percentile_points: list[int] | None = None,
) -> pd.DataFrame:
    """Return p1..p99 EV/EU percentile table across all fitted distributions."""
    if not fits:
        return pd.DataFrame(
            columns=[
                "distribution",
                "percentile",
                "quantile",
                "ev_percentile_value",
                "eu_percentile_value",
                "risk_neutral_ev",
                "upside_skepticism_ev",
                "downside_protection_eu",
                "combined_eu",
                "dmreu_ev",
                "wlu_ev",
                "ambiguity_aversion_ev",
                "fit_error",
            ]
        )

    frames = [ev_eu_percentile_table(fit, params, percentile_points) for fit in fits]
    return pd.concat(frames, ignore_index=True)
