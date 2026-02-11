"""Tests for distribution fitting."""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distributions import (
    fit_distribution,
    fit_all,
    compare_fits,
    DISTRIBUTIONS,
    FitResult,
    percentile_table,
    percentile_table_all,
)


class TestFitDistribution:

    @pytest.fixture
    def symmetric_percentiles(self):
        return {0.10: -30.0, 0.50: 10.0, 0.90: 50.0}

    @pytest.fixture
    def positive_skewed_percentiles(self):
        return {0.10: 5.0, 0.50: 20.0, 0.90: 100.0}

    def test_normal_fits_symmetric_data(self, symmetric_percentiles):
        result = fit_distribution("normal", symmetric_percentiles)
        assert result.error < 0.01
        for q, v in symmetric_percentiles.items():
            assert abs(result.ppf(q) - v) / max(abs(v), 1) < 0.05

    def test_lognormal_fits_positive_data(self, positive_skewed_percentiles):
        result = fit_distribution("lognormal", positive_skewed_percentiles)
        assert result.error < 0.1
        assert result.mean() > 0

    def test_skew_normal_fits(self, symmetric_percentiles):
        result = fit_distribution("skew_normal", symmetric_percentiles)
        assert result.error < 0.1

    def test_students_t_fits(self, symmetric_percentiles):
        result = fit_distribution("students_t", symmetric_percentiles)
        assert result.error < 0.1

    def test_gev_fits(self, symmetric_percentiles):
        result = fit_distribution("gev", symmetric_percentiles)
        assert result.error < 0.1

    def test_log_students_t_fits_positive(self, positive_skewed_percentiles):
        result = fit_distribution("log_students_t", positive_skewed_percentiles)
        assert result.error < 0.5  # more tolerance for custom dist
        assert result.mean() > 0

    def test_unknown_distribution_raises(self):
        with pytest.raises(ValueError, match="Unknown distribution"):
            fit_distribution("nonexistent", {0.5: 10.0})

    def test_positive_only_distribution_rejects_nonpositive(self):
        with pytest.raises(ValueError, match="requires all percentile values to be positive"):
            fit_distribution("lognormal", {0.10: -5.0, 0.50: 10.0, 0.90: 50.0})


class TestFitAll:

    def test_returns_all_distributions(self):
        pcts = {0.10: -30.0, 0.50: 10.0, 0.90: 50.0}
        results = fit_all(pcts)
        # May be fewer if some fail, but should get most
        assert len(results) >= 4

    def test_sorted_by_error(self):
        pcts = {0.10: 5.0, 0.50: 20.0, 0.90: 100.0}
        results = fit_all(pcts)
        errors = [r.error for r in results]
        assert errors == sorted(errors)

    def test_subset_of_distributions(self):
        pcts = {0.10: -10.0, 0.50: 5.0, 0.90: 20.0}
        results = fit_all(pcts, distributions=["normal", "students_t"])
        names = {r.name for r in results}
        assert names <= {"normal", "students_t"}

    def test_positive_only_distributions_skipped_with_negative(self):
        pcts = {0.10: -5.0, 0.50: 10.0, 0.90: 30.0}
        results = fit_all(pcts)
        names = {r.name for r in results}
        assert "lognormal" not in names
        assert "log_students_t" not in names

    def test_overdetermined_system(self):
        pcts = {0.05: 1, 0.25: 5, 0.50: 10, 0.75: 18, 0.95: 40}
        results = fit_all(pcts)
        assert len(results) >= 3
        # Best fit should be reasonable
        assert results[0].error < 0.5


class TestCompareFits:

    def test_returns_dataframe_with_expected_columns(self):
        pcts = {0.10: -10.0, 0.50: 5.0, 0.90: 20.0}
        results = fit_all(pcts)
        df = compare_fits(results)
        assert "distribution" in df.columns
        assert "fit_error" in df.columns
        assert "mean" in df.columns
        assert len(df) == len(results)


class TestPercentileTables:

    def test_percentile_table_defaults_to_1_to_99(self):
        fit = fit_distribution("normal", {0.10: -10.0, 0.50: 5.0, 0.90: 20.0})
        df = percentile_table(fit)
        assert len(df) == 99
        assert df["percentile"].min() == 1
        assert df["percentile"].max() == 99
        assert "value" in df.columns

    def test_percentile_table_custom_points(self):
        fit = fit_distribution("normal", {0.10: -10.0, 0.50: 5.0, 0.90: 20.0})
        df = percentile_table(fit, [10, 50, 90])
        assert list(df["percentile"]) == [10, 50, 90]
        assert len(df) == 3

    def test_percentile_table_rejects_invalid_points(self):
        fit = fit_distribution("normal", {0.10: -10.0, 0.50: 5.0, 0.90: 20.0})
        with pytest.raises(ValueError, match="\\[1, 99\\]"):
            percentile_table(fit, [0, 50, 101])

    def test_percentile_table_all_stacks_distributions(self):
        pcts = {0.10: -10.0, 0.50: 5.0, 0.90: 20.0}
        fits = fit_all(pcts, distributions=["normal", "students_t"])
        df = percentile_table_all(fits)
        assert len(df) == 99 * len(fits)
        assert set(df["distribution"]) == {f.name for f in fits}
