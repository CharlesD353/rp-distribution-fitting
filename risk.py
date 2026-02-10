"""
Risk adjustment computations.

Takes fitted distributions and computes expected values under different
risk preferences:
  - Risk neutral: standard EV
  - Upside skepticism: truncate at a high percentile, renormalize
  - Downside protection: loss-averse utility function
  - Combined: both truncation and loss aversion
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import integrate

from distributions import FitResult


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class RiskParams:
    """Configuration for risk adjustments."""
    truncation_percentile: float = 0.99
    loss_aversion_lambda: float = 2.5
    reference_point: float = 0.0
    integration_bounds_q: tuple[float, float] = (0.0001, 0.9999)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class RiskResult:
    """Risk-adjusted expected values for a single distribution."""
    distribution_name: str
    risk_neutral_ev: float
    upside_skepticism_ev: float
    downside_protection_eu: float
    combined_eu: float
    params: RiskParams

    def to_dict(self) -> dict:
        return {
            "distribution": self.distribution_name,
            "risk_neutral_ev": self.risk_neutral_ev,
            "upside_skepticism_ev": self.upside_skepticism_ev,
            "downside_protection_eu": self.downside_protection_eu,
            "combined_eu": self.combined_eu,
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
    return numerator / (trunc_q - lower_q)


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
    return numerator / (trunc_q - lower_q) + reference_point


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def analyze(
    fit: FitResult,
    params: RiskParams | None = None,
) -> RiskResult:
    """Compute all four risk adjustments for a single fitted distribution."""
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
        params=p,
    )


def analyze_all(
    fits: list[FitResult],
    params: RiskParams | None = None,
) -> pd.DataFrame:
    """Compute risk adjustments for all fitted distributions.

    Returns a DataFrame with columns:
        distribution, risk_neutral_ev, upside_skepticism_ev,
        downside_protection_eu, combined_eu, fit_error
    """
    rows = []
    for fit in fits:
        result = analyze(fit, params)
        row = result.to_dict()
        row["fit_error"] = fit.error
        rows.append(row)
    return pd.DataFrame(rows)
