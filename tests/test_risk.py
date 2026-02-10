"""Tests for risk adjustment computations."""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distributions import fit_distribution
from risk import (
    analyze, analyze_all, RiskParams,
    compute_risk_neutral, compute_upside_skepticism,
    compute_downside_protection, compute_combined,
)


class TestRiskAdjustments:

    @pytest.fixture
    def normal_fit_with_downside(self):
        """Normal dist with substantial mass below zero."""
        return fit_distribution("normal", {0.10: -30.0, 0.50: 10.0, 0.90: 50.0})

    @pytest.fixture
    def lognormal_fit_positive(self):
        """Lognormal dist, all positive."""
        return fit_distribution("lognormal", {0.10: 5.0, 0.50: 20.0, 0.90: 100.0})

    def test_risk_neutral_close_to_dist_mean(self, normal_fit_with_downside):
        ev = compute_risk_neutral(normal_fit_with_downside)
        mean = normal_fit_with_downside.mean()
        assert abs(ev - mean) / max(abs(mean), 1) < 0.05

    def test_upside_skepticism_reduces_ev(self, normal_fit_with_downside):
        ev = compute_risk_neutral(normal_fit_with_downside)
        trunc = compute_upside_skepticism(normal_fit_with_downside, 0.99)
        assert trunc < ev

    def test_loss_aversion_reduces_eu_when_downside_exists(self, normal_fit_with_downside):
        ev = compute_risk_neutral(normal_fit_with_downside)
        eu = compute_downside_protection(normal_fit_with_downside, loss_aversion_lambda=2.5)
        assert eu < ev

    def test_loss_aversion_minimal_for_positive_dist(self, lognormal_fit_positive):
        ev = compute_risk_neutral(lognormal_fit_positive)
        eu = compute_downside_protection(lognormal_fit_positive, loss_aversion_lambda=2.5)
        # Should be close since almost no mass below 0
        assert abs(eu - ev) / max(abs(ev), 1) < 0.1

    def test_combined_is_most_conservative(self, normal_fit_with_downside):
        result = analyze(normal_fit_with_downside)
        assert result.combined_eu <= result.risk_neutral_ev + 0.01
        assert result.combined_eu <= result.upside_skepticism_ev + 0.01

    def test_lambda_1_equals_risk_neutral(self, normal_fit_with_downside):
        """Loss aversion of 1.0 means no asymmetry."""
        ev = compute_risk_neutral(normal_fit_with_downside)
        eu = compute_downside_protection(normal_fit_with_downside, loss_aversion_lambda=1.0)
        assert abs(eu - ev) / max(abs(ev), 1) < 0.02

    def test_higher_lambda_more_conservative(self, normal_fit_with_downside):
        eu_low = compute_downside_protection(normal_fit_with_downside, loss_aversion_lambda=1.5)
        eu_high = compute_downside_protection(normal_fit_with_downside, loss_aversion_lambda=3.0)
        assert eu_high < eu_low

    def test_stricter_truncation_more_conservative(self, normal_fit_with_downside):
        ev_99 = compute_upside_skepticism(normal_fit_with_downside, 0.99)
        ev_95 = compute_upside_skepticism(normal_fit_with_downside, 0.95)
        assert ev_95 < ev_99


class TestAnalyzeAll:

    def test_returns_dataframe(self):
        from distributions import fit_all
        pcts = {0.10: -10.0, 0.50: 5.0, 0.90: 20.0}
        fits = fit_all(pcts)
        df = analyze_all(fits)
        assert "distribution" in df.columns
        assert "risk_neutral_ev" in df.columns
        assert len(df) == len(fits)
