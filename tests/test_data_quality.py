"""Tests for src/data_quality.py."""

import pandas as pd
import pytest

from src.data_quality import generate_quality_report, get_quality_summary


# ── generate_quality_report ───────────────────────────────────────────────────

class TestGenerateQualityReport:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "col_a": [1.0, 2.0, None, 4.0],
                "col_b": ["x", "y", "y", None],
                "col_c": [10, 20, 30, 40],
            }
        )

    def test_report_has_one_row_per_column(self, sample_df):
        report = generate_quality_report(sample_df)
        assert len(report) == len(sample_df.columns)

    def test_missing_count_correct(self, sample_df):
        report = generate_quality_report(sample_df)
        row = report[report["Column"] == "col_a"].iloc[0]
        assert row["Missing Count"] == 1

    def test_missing_percentage_correct(self, sample_df):
        report = generate_quality_report(sample_df)
        row = report[report["Column"] == "col_a"].iloc[0]
        # 1 missing out of 4 rows = 25%
        assert row["Missing %"] == 25.0

    def test_missing_percentage_zero_when_no_missing(self, sample_df):
        report = generate_quality_report(sample_df)
        row = report[report["Column"] == "col_c"].iloc[0]
        assert row["Missing %"] == 0.0

    def test_unique_values_excludes_nan(self, sample_df):
        report = generate_quality_report(sample_df)
        row = report[report["Column"] == "col_b"].iloc[0]
        # "x" and "y" are unique; None is excluded by nunique
        assert row["Unique Values"] == 2

    def test_non_null_count_correct(self, sample_df):
        report = generate_quality_report(sample_df)
        row = report[report["Column"] == "col_a"].iloc[0]
        assert row["Non-Null Count"] == 3

    def test_data_type_column_is_string(self, sample_df):
        report = generate_quality_report(sample_df)
        for dtype_val in report["Data Type"]:
            assert isinstance(dtype_val, str)

    def test_required_columns_present(self, sample_df):
        report = generate_quality_report(sample_df)
        for expected_col in [
            "Column",
            "Data Type",
            "Non-Null Count",
            "Missing Count",
            "Missing %",
            "Unique Values",
        ]:
            assert expected_col in report.columns

    def test_empty_dataframe_returns_empty_report(self):
        report = generate_quality_report(pd.DataFrame())
        assert report.empty

    def test_no_missing_values_all_zeros(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        report = generate_quality_report(df)
        assert report["Missing Count"].sum() == 0
        assert report["Missing %"].sum() == 0.0


# ── get_quality_summary ───────────────────────────────────────────────────────

class TestGetQualitySummary:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "a": [1, 2, 2, None],
                "b": ["x", "y", "y", "z"],
            }
        )

    def test_row_count_correct(self, sample_df):
        summary = get_quality_summary(sample_df)
        assert summary["row_count"] == 4

    def test_col_count_correct(self, sample_df):
        summary = get_quality_summary(sample_df)
        assert summary["col_count"] == 2

    def test_duplicate_count_correct(self, sample_df):
        summary = get_quality_summary(sample_df)
        # Row (2, "y") appears twice → 1 duplicate
        assert summary["duplicate_count"] == 1

    def test_total_missing_correct(self, sample_df):
        summary = get_quality_summary(sample_df)
        assert summary["total_missing"] == 1

    def test_total_missing_pct_correct(self, sample_df):
        summary = get_quality_summary(sample_df)
        # 1 missing out of 4*2=8 cells = 12.5%
        assert summary["total_missing_pct"] == 12.5

    def test_empty_dataframe_returns_zeros(self):
        summary = get_quality_summary(pd.DataFrame())
        assert summary["row_count"] == 0
        assert summary["duplicate_count"] == 0
        assert summary["total_missing"] == 0
        assert summary["total_missing_pct"] == 0.0

    def test_no_duplicates(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        summary = get_quality_summary(df)
        assert summary["duplicate_count"] == 0

    def test_no_missing_values(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        summary = get_quality_summary(df)
        assert summary["total_missing"] == 0
        assert summary["total_missing_pct"] == 0.0
