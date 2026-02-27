"""Tests for risk adjustment computations."""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distributions import fit_distribution
from risk_analysis import (
    analyze, analyze_all, RiskParams,
    compute_risk_neutral, compute_upside_skepticism,
    compute_downside_protection, compute_combined,
    compute_dmreu, compute_wlu, compute_ambiguity_aversion,
    ev_eu_percentile_table, ev_eu_percentile_table_all,
    FormalModelRun, compute_formal_run, compute_formal_runs_all,
    _apply_probability_rounding, _generate_samples,
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
        assert result.combined_eu <= result.downside_protection_eu + 0.01

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

    def test_returns_formal_model_columns(self):
        from distributions import fit_all
        pcts = {0.10: -10.0, 0.50: 5.0, 0.90: 20.0}
        fits = fit_all(pcts)
        df = analyze_all(fits)
        assert "dmreu_ev" in df.columns
        assert "wlu_ev" in df.columns
        assert "ambiguity_aversion_ev" in df.columns


# ---------------------------------------------------------------------------
# Formal models (Duffy 2023)
# ---------------------------------------------------------------------------


class TestDMREU:

    @pytest.fixture
    def normal_fit(self):
        return fit_distribution("normal", {0.10: -30.0, 0.50: 10.0, 0.90: 50.0})

    @pytest.fixture
    def lognormal_fit(self):
        return fit_distribution("lognormal", {0.10: 5.0, 0.50: 20.0, 0.90: 100.0})

    def test_p001_recovers_risk_neutral(self, normal_fit):
        """DMREU with p=0.01 (a=1.0) should equal risk-neutral EV."""
        ev = compute_risk_neutral(normal_fit)
        dmreu = compute_dmreu(normal_fit, p=0.01)
        assert abs(dmreu - ev) / max(abs(ev), 1) < 0.03

    def test_higher_risk_aversion_more_conservative(self, normal_fit):
        """Higher p (more risk-averse) should yield lower value."""
        d_low = compute_dmreu(normal_fit, p=0.02)
        d_high = compute_dmreu(normal_fit, p=0.10)
        assert d_high < d_low

    def test_positive_distribution(self, lognormal_fit):
        """DMREU should work for all-positive distributions."""
        ev = compute_risk_neutral(lognormal_fit)
        d = compute_dmreu(lognormal_fit, p=0.05)
        assert d < ev
        assert d > 0


class TestWLU:

    @pytest.fixture
    def normal_fit(self):
        return fit_distribution("normal", {0.10: -30.0, 0.50: 10.0, 0.90: 50.0})

    @pytest.fixture
    def lognormal_fit(self):
        return fit_distribution("lognormal", {0.10: 5.0, 0.50: 20.0, 0.90: 100.0})

    def test_c0_recovers_risk_neutral(self, normal_fit):
        """WLU with c=0 should equal risk-neutral EV."""
        ev = compute_risk_neutral(normal_fit)
        wlu = compute_wlu(normal_fit, c=0.0)
        assert abs(wlu - ev) / max(abs(ev), 1) < 0.03

    def test_higher_c_more_conservative_positive_dist(self, lognormal_fit):
        """For a right-skewed positive distribution, higher c → lower value."""
        w_low = compute_wlu(lognormal_fit, c=0.05)
        w_high = compute_wlu(lognormal_fit, c=0.25)
        assert w_high < w_low

    def test_symmetric_distribution_effect(self, normal_fit):
        """For a roughly symmetric distribution, WLU with c>0 should still reduce EV
        because large positive values are downweighted."""
        ev = compute_risk_neutral(normal_fit)
        wlu = compute_wlu(normal_fit, c=0.15)
        assert wlu < ev


class TestAmbiguityAversion:

    @pytest.fixture
    def normal_fit(self):
        return fit_distribution("normal", {0.10: -30.0, 0.50: 10.0, 0.90: 50.0})

    def test_k0_recovers_risk_neutral(self, normal_fit):
        """Ambiguity aversion with k=0 should equal risk-neutral EV."""
        ev = compute_risk_neutral(normal_fit)
        aa = compute_ambiguity_aversion(normal_fit, k=0.0)
        assert abs(aa - ev) / max(abs(ev), 1) < 0.03

    def test_higher_k_more_conservative(self, normal_fit):
        """Higher k should yield lower (more conservative) value."""
        a_low = compute_ambiguity_aversion(normal_fit, k=2.0)
        a_high = compute_ambiguity_aversion(normal_fit, k=8.0)
        assert a_high < a_low

    def test_k8_reasonable_result(self, normal_fit):
        """At k=8 (strong ambiguity aversion), result should still be finite
        and more conservative than risk-neutral."""
        ev = compute_risk_neutral(normal_fit)
        aa = compute_ambiguity_aversion(normal_fit, k=8.0)
        assert np.isfinite(aa)
        assert aa < ev


class TestFormalModelsIntegration:

    def test_analyze_includes_formal_models(self):
        fit = fit_distribution("normal", {0.10: -30.0, 0.50: 10.0, 0.90: 50.0})
        result = analyze(fit)
        assert hasattr(result, "dmreu_ev")
        assert hasattr(result, "wlu_ev")
        assert hasattr(result, "ambiguity_aversion_ev")

    def test_neutral_params_all_approx_risk_neutral(self):
        """With all formal params at neutral defaults, formal models should
        approximately equal risk-neutral EV."""
        fit = fit_distribution("normal", {0.10: -30.0, 0.50: 10.0, 0.90: 50.0})
        params = RiskParams(dmreu_p=0.01, wlu_c=0.0, ambiguity_k=0.0)
        result = analyze(fit, params)
        ev = result.risk_neutral_ev
        assert abs(result.dmreu_ev - ev) / max(abs(ev), 1) < 0.03
        assert abs(result.wlu_ev - ev) / max(abs(ev), 1) < 0.03
        assert abs(result.ambiguity_aversion_ev - ev) / max(abs(ev), 1) < 0.03

    def test_to_dict_includes_formal_models(self):
        fit = fit_distribution("normal", {0.10: -30.0, 0.50: 10.0, 0.90: 50.0})
        result = analyze(fit)
        d = result.to_dict()
        assert "dmreu_ev" in d
        assert "wlu_ev" in d
        assert "ambiguity_aversion_ev" in d


class TestEvEuPercentileTables:

    def test_single_distribution_table_defaults_to_1_to_99(self):
        fit = fit_distribution("normal", {0.10: -30.0, 0.50: 10.0, 0.90: 50.0})
        df = ev_eu_percentile_table(fit)
        assert len(df) == 99
        assert df["percentile"].min() == 1
        assert df["percentile"].max() == 99
        assert "ev_percentile_value" in df.columns
        assert "eu_percentile_value" in df.columns
        assert "risk_neutral_ev" in df.columns
        assert "downside_protection_eu" in df.columns

    def test_summary_metrics_are_constant_within_distribution_table(self):
        fit = fit_distribution("normal", {0.10: -30.0, 0.50: 10.0, 0.90: 50.0})
        df = ev_eu_percentile_table(fit)
        assert df["risk_neutral_ev"].nunique() == 1
        assert df["downside_protection_eu"].nunique() == 1
        assert df["combined_eu"].nunique() == 1

    def test_multi_distribution_table_has_all_rows(self):
        from distributions import fit_all

        pcts = {0.10: -10.0, 0.50: 5.0, 0.90: 20.0}
        fits = fit_all(pcts, distributions=["normal", "students_t"])
        df = ev_eu_percentile_table_all(fits)
        assert len(df) == 99 * len(fits)
        assert set(df["distribution"]) == {f.name for f in fits}


# ---------------------------------------------------------------------------
# Flexible formal model runs
# ---------------------------------------------------------------------------


class TestFormalModelRun:

    def test_auto_label_wlu(self):
        run = FormalModelRun(model="wlu", param=0.05)
        assert run.label == "WLU (c=0.05)"

    def test_auto_label_dmreu(self):
        run = FormalModelRun(model="dmreu", param=0.03)
        assert run.label == "DMREU (p=0.03)"

    def test_auto_label_ambiguity(self):
        run = FormalModelRun(model="ambiguity", param=4.0)
        assert run.label == "AMBIGUITY (k=4.0)"

    def test_custom_label(self):
        run = FormalModelRun(model="wlu", param=0.10, label="My WLU")
        assert run.label == "My WLU"

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            FormalModelRun(model="nonexistent", param=1.0)


class TestComputeFormalRun:

    @pytest.fixture
    def normal_fit(self):
        return fit_distribution("normal", {0.10: -30.0, 0.50: 10.0, 0.90: 50.0})

    def test_wlu_run_matches_direct_call(self, normal_fit):
        run = FormalModelRun(model="wlu", param=0.05)
        via_run = compute_formal_run(normal_fit, run)
        direct = compute_wlu(normal_fit, c=0.05)
        assert abs(via_run - direct) < 1e-10

    def test_dmreu_run_matches_direct_call(self, normal_fit):
        run = FormalModelRun(model="dmreu", param=0.05)
        via_run = compute_formal_run(normal_fit, run)
        direct = compute_dmreu(normal_fit, p=0.05)
        assert abs(via_run - direct) < 1e-10

    def test_ambiguity_run_matches_direct_call(self, normal_fit):
        run = FormalModelRun(model="ambiguity", param=4.0)
        via_run = compute_formal_run(normal_fit, run)
        direct = compute_ambiguity_aversion(normal_fit, k=4.0)
        assert abs(via_run - direct) < 1e-10


class TestComputeFormalRunsAll:

    def test_returns_correct_columns(self):
        from distributions import fit_all
        pcts = {0.10: -10.0, 0.50: 5.0, 0.90: 20.0}
        fits = fit_all(pcts, distributions=["normal", "students_t"])
        runs = [
            FormalModelRun(model="wlu", param=0.01),
            FormalModelRun(model="wlu", param=0.05),
            FormalModelRun(model="wlu", param=0.10),
        ]
        df = compute_formal_runs_all(fits, runs)
        assert "distribution" in df.columns
        assert "risk_neutral_ev" in df.columns
        assert "fit_error" in df.columns
        for run in runs:
            assert run.label in df.columns
        assert len(df) == len(fits)

    def test_higher_wlu_c_more_conservative(self):
        from distributions import fit_all
        pcts = {0.10: 5.0, 0.50: 20.0, 0.90: 100.0}
        fits = fit_all(pcts, distributions=["lognormal"])
        runs = [
            FormalModelRun(model="wlu", param=0.01),
            FormalModelRun(model="wlu", param=0.10),
        ]
        df = compute_formal_runs_all(fits, runs)
        assert df[runs[1].label].iloc[0] < df[runs[0].label].iloc[0]

    def test_mixed_model_types(self):
        from distributions import fit_all
        pcts = {0.10: -10.0, 0.50: 5.0, 0.90: 20.0}
        fits = fit_all(pcts, distributions=["normal"])
        runs = [
            FormalModelRun(model="dmreu", param=0.05),
            FormalModelRun(model="wlu", param=0.05),
            FormalModelRun(model="ambiguity", param=4.0),
        ]
        df = compute_formal_runs_all(fits, runs)
        for run in runs:
            assert run.label in df.columns
            assert np.isfinite(df[run.label].iloc[0])


# ---------------------------------------------------------------------------
# Probability rounding
# ---------------------------------------------------------------------------


class TestProbabilityRounding:

    @pytest.fixture
    def lognormal_fit(self):
        return fit_distribution("lognormal", {0.10: 5.0, 0.50: 20.0, 0.90: 100.0})

    @pytest.fixture
    def normal_fit(self):
        return fit_distribution("normal", {0.10: -30.0, 0.50: 10.0, 0.90: 50.0})

    def test_epsilon_zero_is_noop(self, lognormal_fit):
        samples = _generate_samples(lognormal_fit)
        rounded = _apply_probability_rounding(samples, lognormal_fit, 0.0)
        np.testing.assert_array_equal(samples, rounded)

    def test_high_positive_values_zeroed(self, lognormal_fit):
        """With epsilon=0.05, the top ~5% of positive outcomes should be zeroed."""
        samples = _generate_samples(lognormal_fit)
        rounded = _apply_probability_rounding(samples, lognormal_fit, 0.05)
        threshold = lognormal_fit.ppf(0.95)
        assert np.all(rounded[samples > threshold] == 0.0)
        assert np.all(rounded[samples <= threshold] == samples[samples <= threshold])

    def test_negative_values_untouched(self, normal_fit):
        """Rounding only affects positive values; negatives remain unchanged."""
        samples = _generate_samples(normal_fit)
        rounded = _apply_probability_rounding(samples, normal_fit, 0.05)
        neg_mask = samples < 0
        np.testing.assert_array_equal(samples[neg_mask], rounded[neg_mask])

    def test_rounding_reduces_wlu_value(self, lognormal_fit):
        """WLU with rounding should give a lower value than without."""
        run_no_round = FormalModelRun(model="wlu", param=0.05, epsilon=0.0)
        run_round = FormalModelRun(model="wlu", param=0.05, epsilon=0.05)
        val_no = compute_formal_run(lognormal_fit, run_no_round)
        val_yes = compute_formal_run(lognormal_fit, run_round)
        assert val_yes < val_no

    def test_rounding_reduces_dmreu_value(self, lognormal_fit):
        """DMREU with rounding should give a lower value than without."""
        run_no_round = FormalModelRun(model="dmreu", param=0.05, epsilon=0.0)
        run_round = FormalModelRun(model="dmreu", param=0.05, epsilon=0.05)
        val_no = compute_formal_run(lognormal_fit, run_no_round)
        val_yes = compute_formal_run(lognormal_fit, run_round)
        assert val_yes < val_no

    def test_larger_epsilon_more_conservative(self, lognormal_fit):
        """A larger epsilon should zero out more samples and give a lower value."""
        run_small = FormalModelRun(model="wlu", param=0.05, epsilon=0.01)
        run_large = FormalModelRun(model="wlu", param=0.05, epsilon=0.10)
        val_small = compute_formal_run(lognormal_fit, run_small)
        val_large = compute_formal_run(lognormal_fit, run_large)
        assert val_large < val_small

    def test_label_includes_epsilon(self):
        run = FormalModelRun(model="wlu", param=0.05, epsilon=0.01)
        assert "e=0.01" in run.label

    def test_label_omits_epsilon_when_zero(self):
        run = FormalModelRun(model="wlu", param=0.05, epsilon=0.0)
        assert "e=" not in run.label
