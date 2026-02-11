"""
Distribution fitting engine.

Fits parametric distributions to percentile constraints (e.g. "p10=5, p50=20, p90=100")
using scipy.optimize. Supports normal, lognormal, skew-normal, Student's t, GEV, and
log-Student's t distributions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import integrate, optimize, stats

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

PercentileSpec = dict[float, float]  # {quantile: target_value}


# ---------------------------------------------------------------------------
# Log-Student's t distribution (custom)
# ---------------------------------------------------------------------------

class LogStudentT:
    """X = exp(T) where T ~ Student-t(df, loc, scale).

    A heavy-tailed alternative to the lognormal for positive quantities.
    """

    def __init__(self, df: float, loc: float = 0.0, scale: float = 1.0):
        self._t = stats.t(df, loc=loc, scale=scale)
        self._df = df
        self._loc = loc
        self._scale = scale

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(x > 0, self._t.pdf(np.log(x)) / x, 0.0)
        return result

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x > 0, self._t.cdf(np.log(x)), 0.0)

    def ppf(self, q):
        return np.exp(self._t.ppf(q))

    def mean(self):
        lb = float(self.ppf(0.001))
        ub = float(self.ppf(0.999))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result, _ = integrate.quad(lambda x: x * self.pdf(x), lb, ub, limit=100)
        return result

    def std(self):
        mu = self.mean()
        lb = float(self.ppf(0.001))
        ub = float(self.ppf(0.999))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            var, _ = integrate.quad(lambda x: (x - mu) ** 2 * self.pdf(x), lb, ub, limit=100)
        return np.sqrt(max(var, 0.0))

    def median(self):
        return float(self.ppf(0.5))

    def stats(self, moments="mv"):
        """Mimic scipy interface for moments."""
        result = []
        if "m" in moments:
            result.append(self.mean())
        if "v" in moments:
            result.append(self.std() ** 2)
        if "s" in moments:
            result.append(np.nan)  # skewness not computed
        if "k" in moments:
            result.append(np.nan)  # kurtosis not computed
        return tuple(result)


# ---------------------------------------------------------------------------
# Distribution registry
# ---------------------------------------------------------------------------

class DistConfig(NamedTuple):
    """Configuration for a fittable distribution."""
    dist_class: object              # scipy.stats distribution or LogStudentT class
    param_names: list[str]          # names of parameters to optimize
    initial_guess: list[float]      # starting point for optimizer
    bounds: list[tuple[float, float]]  # (lo, hi) for each param
    is_custom: bool = False         # True for LogStudentT
    positive_only: bool = False     # True if distribution support is strictly > 0


DISTRIBUTIONS: dict[str, DistConfig] = {
    "normal": DistConfig(
        dist_class=stats.norm,
        param_names=["loc", "scale"],
        initial_guess=[0.0, 10.0],
        bounds=[(-1e6, 1e6), (1e-6, 1e6)],
    ),
    "lognormal": DistConfig(
        dist_class=stats.lognorm,
        param_names=["s", "loc", "scale"],
        initial_guess=[1.0, 0.0, 10.0],
        bounds=[(1e-6, 10.0), (-1e6, 1e6), (1e-6, 1e6)],
        positive_only=True,
    ),
    "skew_normal": DistConfig(
        dist_class=stats.skewnorm,
        param_names=["a", "loc", "scale"],
        initial_guess=[0.0, 0.0, 10.0],
        bounds=[(-20.0, 20.0), (-1e6, 1e6), (1e-6, 1e6)],
    ),
    "students_t": DistConfig(
        dist_class=stats.t,
        param_names=["df", "loc", "scale"],
        initial_guess=[5.0, 0.0, 10.0],
        bounds=[(2.01, 100.0), (-1e6, 1e6), (1e-6, 1e6)],
    ),
    "gev": DistConfig(
        dist_class=stats.genextreme,
        param_names=["c", "loc", "scale"],
        initial_guess=[0.0, 0.0, 10.0],
        bounds=[(-0.5, 0.5), (-1e6, 1e6), (1e-6, 1e6)],
    ),
    "log_students_t": DistConfig(
        dist_class=LogStudentT,
        param_names=["df", "loc", "scale"],
        initial_guess=[5.0, 0.0, 1.0],
        bounds=[(2.01, 100.0), (-50.0, 50.0), (1e-6, 50.0)],
        is_custom=True,
        positive_only=True,
    ),
}


# ---------------------------------------------------------------------------
# FitResult
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """Result of fitting a distribution to percentile constraints."""
    name: str
    params: dict[str, float]
    error: float
    percentiles: PercentileSpec
    _dist: object = field(repr=False)  # frozen scipy dist or LogStudentT instance

    def pdf(self, x):
        return self._dist.pdf(x)

    def cdf(self, x):
        return self._dist.cdf(x)

    def ppf(self, q):
        result = self._dist.ppf(q)
        if np.ndim(result) == 0:
            return float(result)
        return np.asarray(result, dtype=float)

    def mean(self):
        return float(self._dist.mean())

    def std(self):
        return float(self._dist.std())

    def median(self):
        return float(self._dist.ppf(0.5))


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _make_frozen(config: DistConfig, param_values: list[float]):
    """Create a frozen distribution from config and parameter values."""
    kwargs = dict(zip(config.param_names, param_values))
    return config.dist_class(**kwargs)


def _objective(params, config: DistConfig, percentiles: PercentileSpec) -> float:
    """Relative squared error between distribution PPF and target percentiles."""
    # Check positivity constraints
    kwargs = dict(zip(config.param_names, params))
    for k in ("scale", "s", "df"):
        if k in kwargs and kwargs[k] <= 0:
            return 1e10
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            frozen = _make_frozen(config, list(params))
            total = 0.0
            for q, target in percentiles.items():
                predicted = float(frozen.ppf(q))
                if not np.isfinite(predicted):
                    return 1e10
                denom = max(abs(target), 1e-6)
                total += ((predicted - target) / denom) ** 2
            return total
    except (ValueError, OverflowError, RuntimeWarning):
        return 1e10


def _smart_initial_guess(config: DistConfig, percentiles: PercentileSpec) -> list[float]:
    """Derive a better initial guess from the percentile data."""
    sorted_pairs = sorted(percentiles.items())
    quantiles = [q for q, _ in sorted_pairs]
    values = [v for _, v in sorted_pairs]
    median_val = np.interp(0.5, quantiles, values) if 0.5 not in percentiles else percentiles[0.5]
    spread = max(values) - min(values)

    guess = list(config.initial_guess)
    for i, name in enumerate(config.param_names):
        if name == "loc":
            if config.is_custom:
                # For log-t, loc is of the underlying log-space
                positive_vals = [v for v in values if v > 0]
                if positive_vals:
                    guess[i] = np.log(np.median(positive_vals))
                else:
                    guess[i] = 0.0
            else:
                guess[i] = median_val
        elif name == "scale":
            if config.is_custom:
                guess[i] = max(1.0, np.std(np.log(np.array([v for v in values if v > 0]) + 1e-6)))
            elif config.dist_class == stats.lognorm:
                guess[i] = max(abs(median_val), 1.0)
            else:
                guess[i] = max(spread / 3, 1e-3)
        elif name == "s":
            # lognormal shape: estimate from log-space spread
            positive_vals = [v for v in values if v > 0]
            if len(positive_vals) >= 2:
                guess[i] = max(np.std(np.log(positive_vals)), 0.1)
            else:
                guess[i] = 1.0

    return guess


def _fit_settings() -> dict:
    """Return optimization settings, optionally in fast mode.

    Fast mode is enabled by setting RP_FAST=1 (or true/yes) in the environment.
    """
    fast_flag = os.getenv("RP_FAST", "").strip().lower()
    fast = fast_flag in {"1", "true", "yes", "on"}
    if fast:
        return {
            "methods": ["Nelder-Mead", "L-BFGS-B"],
            "maxiter": 800,
            "tol": 1e-6,
            "perturbations": 2,
        }
    return {
        "methods": ["Nelder-Mead", "Powell", "L-BFGS-B"],
        "maxiter": 5000,
        "tol": 1e-10,
        "perturbations": 5,
    }


def fit_distribution(
    name: str,
    percentiles: PercentileSpec,
) -> FitResult:
    """Fit a single named distribution to percentile constraints.

    Args:
        name: Distribution name (key in DISTRIBUTIONS).
        percentiles: Mapping of {quantile: target_value}, e.g. {0.10: 5, 0.50: 20, 0.90: 100}.

    Returns:
        FitResult with the frozen distribution and diagnostics.

    Raises:
        ValueError: If distribution name is unknown or fitting fails completely.
    """
    if name not in DISTRIBUTIONS:
        raise ValueError(f"Unknown distribution: {name!r}. Choose from: {list(DISTRIBUTIONS.keys())}")

    config = DISTRIBUTIONS[name]
    if config.positive_only and any(v <= 0 for v in percentiles.values()):
        raise ValueError(f"{name!r} distribution requires all percentile values to be positive.")
    guess = _smart_initial_guess(config, percentiles)

    settings = _fit_settings()

    # Try multiple optimizer methods for robustness
    best_result = None
    best_error = float("inf")

    for method in settings["methods"]:
        try:
            kwargs = {}
            if method == "L-BFGS-B":
                kwargs["bounds"] = config.bounds

            result = optimize.minimize(
                _objective,
                x0=guess,
                args=(config, percentiles),
                method=method,
                options={"maxiter": settings["maxiter"], "xatol": settings["tol"], "fatol": settings["tol"]}
                if method == "Nelder-Mead"
                else {"maxiter": settings["maxiter"]},
                **kwargs,
            )
            if result.fun < best_error:
                best_error = result.fun
                best_result = result
        except Exception:
            continue

    # Also try with a few random perturbations of the initial guess
    rng = np.random.RandomState(42)
    for _ in range(settings["perturbations"]):
        perturbed = [
            np.clip(g * rng.uniform(0.5, 2.0), lo, hi)
            for g, (lo, hi) in zip(guess, config.bounds)
        ]
        try:
            result = optimize.minimize(
                _objective,
                x0=perturbed,
                args=(config, percentiles),
                method="Nelder-Mead",
                options={"maxiter": settings["maxiter"], "xatol": settings["tol"], "fatol": settings["tol"]},
            )
            if result.fun < best_error:
                best_error = result.fun
                best_result = result
        except Exception:
            continue

    if best_result is None:
        raise ValueError(f"Failed to fit {name!r} distribution to the given percentiles.")

    params = dict(zip(config.param_names, best_result.x))
    frozen = _make_frozen(config, list(best_result.x))

    return FitResult(
        name=name,
        params=params,
        error=best_error,
        percentiles=percentiles,
        _dist=frozen,
    )


def fit_all(
    percentiles: PercentileSpec,
    distributions: list[str] | None = None,
) -> list[FitResult]:
    """Fit all (or specified) distributions to the same percentile constraints.

    Returns list sorted by fit error (best first). Distributions that fail
    to fit are omitted with a warning printed to stderr.
    """
    names = distributions or list(DISTRIBUTIONS.keys())
    results = []
    for name in names:
        config = DISTRIBUTIONS.get(name)
        if config and config.positive_only and any(v <= 0 for v in percentiles.values()):
            continue
        try:
            results.append(fit_distribution(name, percentiles))
        except ValueError as e:
            import sys
            print(f"Warning: {e}", file=sys.stderr)
    results.sort(key=lambda r: r.error)
    return results


def compare_fits(results: list[FitResult]) -> pd.DataFrame:
    """Return a comparison table of fitted distributions.

    Columns: distribution, params, fit_error, mean, median, std, skewness, kurtosis.
    """
    rows = []
    for r in results:
        params_str = ", ".join(f"{k}={v:.4g}" for k, v in r.params.items())
        try:
            mean_val = r.mean()
        except Exception:
            mean_val = np.nan
        try:
            std_val = r.std()
        except Exception:
            std_val = np.nan

        # Skewness and kurtosis via moments if available
        try:
            if hasattr(r._dist, "stats"):
                moment_vals = r._dist.stats(moments="sk")
                skew = float(moment_vals[0]) if np.isfinite(moment_vals[0]) else np.nan
                kurt = float(moment_vals[1]) if np.isfinite(moment_vals[1]) else np.nan
            else:
                skew = np.nan
                kurt = np.nan
        except Exception:
            skew = np.nan
            kurt = np.nan

        rows.append({
            "distribution": r.name,
            "parameters": params_str,
            "fit_error": r.error,
            "mean": mean_val,
            "median": r.median(),
            "std": std_val,
            "skewness": skew,
            "kurtosis": kurt,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Percentile table helpers
# ---------------------------------------------------------------------------

def _validate_percentile_points(percentile_points: list[int] | None = None) -> list[int]:
    """Validate and normalize percentile points for export tables."""
    points = list(range(1, 100)) if percentile_points is None else sorted(set(percentile_points))
    if not points:
        raise ValueError("At least one percentile point is required.")
    invalid = [p for p in points if p < 1 or p > 99]
    if invalid:
        raise ValueError(
            f"Percentiles must be integers in [1, 99], got invalid values: {invalid}"
        )
    return points


def percentile_table(
    fit: FitResult,
    percentile_points: list[int] | None = None,
) -> pd.DataFrame:
    """Return percentile outcomes for one fitted distribution.

    Default points are p1...p99.
    """
    points = _validate_percentile_points(percentile_points)
    rows = []
    for p in points:
        q = p / 100.0
        rows.append(
            {
                "distribution": fit.name,
                "percentile": p,
                "quantile": q,
                "value": fit.ppf(q),
            }
        )
    return pd.DataFrame(rows)


def percentile_table_all(
    fits: list[FitResult],
    percentile_points: list[int] | None = None,
) -> pd.DataFrame:
    """Return percentile outcomes for multiple fitted distributions."""
    points = _validate_percentile_points(percentile_points)
    frames = [percentile_table(fit, points) for fit in fits]
    if not frames:
        return pd.DataFrame(columns=["distribution", "percentile", "quantile", "value"])
    return pd.concat(frames, ignore_index=True)
