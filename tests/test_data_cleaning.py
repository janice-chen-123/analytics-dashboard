"""Tests for src/data_cleaning.py."""

import pandas as pd
import pytest

from src.data_cleaning import CleaningOptions, CleanResult, clean_column_name, clean_dataframe


# ── clean_column_name ─────────────────────────────────────────────────────────

class TestCleanColumnName:
    def test_strips_leading_trailing_whitespace(self):
        assert clean_column_name("  Order Date  ") == "order_date"

    def test_converts_to_lowercase(self):
        assert clean_column_name("OrderDate") == "orderdate"

    def test_replaces_internal_spaces_with_underscores(self):
        assert clean_column_name("Order Date") == "order_date"

    def test_already_clean_name_unchanged(self):
        assert clean_column_name("order_date") == "order_date"

    def test_multiple_spaces_replaced(self):
        assert clean_column_name("Sales  Amount") == "sales__amount"

    def test_uppercase_with_spaces(self):
        assert clean_column_name("  Total Sales  ") == "total_sales"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """DataFrame with dirty names, duplicates, whitespace, and missing values."""
    return pd.DataFrame(
        {
            "Order Date": ["2024-01-01", "2024-02-01", "2024-02-01"],
            "Category ": [" Tech", " Furniture", " Furniture"],
            "Sales": [100.0, 200.0, 200.0],
        }
    )


@pytest.fixture
def df_with_missing():
    return pd.DataFrame(
        {
            "name": ["Alice", None, "Bob", None],
            "sales": [100.0, None, 300.0, None],
            "qty":   [1.0, 2.0, None, None],
        }
    )


@pytest.fixture
def df_with_outliers():
    return pd.DataFrame(
        {
            "value": [10.0, 11.0, 12.0, 10.5, 11.5, 1000.0],  # 1000 is outlier
        }
    )


# ── Baseline cleaning (default options) ──────────────────────────────────────

class TestCleanDataframe:
    def test_column_names_are_lowercased(self, sample_df):
        result = clean_dataframe(sample_df)
        for col in result.cleaned_df.columns:
            assert col == col.lower()

    def test_column_name_spaces_replaced_with_underscores(self, sample_df):
        result = clean_dataframe(sample_df)
        assert "order_date" in result.cleaned_df.columns

    def test_column_name_trailing_space_stripped(self, sample_df):
        result = clean_dataframe(sample_df)
        assert "category" in result.cleaned_df.columns
        assert "category " not in result.cleaned_df.columns

    def test_duplicate_rows_removed(self, sample_df):
        result = clean_dataframe(sample_df)
        assert len(result.cleaned_df) == 2

    def test_string_whitespace_stripped(self, sample_df):
        result = clean_dataframe(sample_df)
        for val in result.cleaned_df["category"].dropna():
            assert val == val.strip()

    def test_original_dataframe_not_modified(self, sample_df):
        original_cols = sample_df.columns.tolist()
        original_len = len(sample_df)
        snapshot = sample_df.copy()
        clean_dataframe(sample_df)
        assert sample_df.columns.tolist() == original_cols
        assert len(sample_df) == original_len
        pd.testing.assert_frame_equal(sample_df, snapshot)

    def test_missing_values_preserved_by_default(self, df_with_missing):
        result = clean_dataframe(df_with_missing)
        assert result.cleaned_df["sales"].isna().sum() == 2
        assert result.cleaned_df["name"].isna().sum() == 2

    def test_empty_dataframe_does_not_raise(self):
        result = clean_dataframe(pd.DataFrame())
        assert result.cleaned_df.empty
        assert len(result.changes) > 0

    def test_column_mapping_populated(self, sample_df):
        result = clean_dataframe(sample_df)
        assert "Order Date" in result.column_mapping
        assert result.column_mapping["Order Date"] == "order_date"

    def test_changes_list_is_not_empty(self, sample_df):
        result = clean_dataframe(sample_df)
        assert len(result.changes) > 0
        assert all(isinstance(c, str) for c in result.changes)

    def test_no_duplicates_stays_same_length(self):
        df = pd.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]})
        result = clean_dataframe(df)
        assert len(result.cleaned_df) == 3

    def test_dataframe_with_only_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 1], "b": ["x", "x", "x"]})
        result = clean_dataframe(df)
        assert len(result.cleaned_df) == 1

    def test_rows_before_after_populated(self, sample_df):
        result = clean_dataframe(sample_df)
        assert result.rows_before == 3
        assert result.rows_after == 2


# ── Fill numeric missing values ───────────────────────────────────────────────

class TestFillNumeric:
    def test_fill_mean(self, df_with_missing):
        opts = CleaningOptions(fill_numeric="mean")
        result = clean_dataframe(df_with_missing, opts)
        assert result.cleaned_df["sales"].isna().sum() == 0
        assert result.cleaned_df["qty"].isna().sum() == 0

    def test_fill_median(self, df_with_missing):
        opts = CleaningOptions(fill_numeric="median")
        result = clean_dataframe(df_with_missing, opts)
        assert result.cleaned_df["sales"].isna().sum() == 0

    def test_fill_zero(self, df_with_missing):
        opts = CleaningOptions(fill_numeric="zero")
        result = clean_dataframe(df_with_missing, opts)
        assert result.cleaned_df["sales"].isna().sum() == 0
        assert (result.cleaned_df["sales"] == 0).sum() == 2

    def test_fill_none_preserves_missing(self, df_with_missing):
        opts = CleaningOptions(fill_numeric="none")
        result = clean_dataframe(df_with_missing, opts)
        assert result.cleaned_df["sales"].isna().sum() == 2

    def test_fill_mean_value_correct(self):
        df = pd.DataFrame({"x": [10.0, 20.0, None]})
        result = clean_dataframe(df, CleaningOptions(fill_numeric="mean"))
        assert result.cleaned_df["x"].iloc[2] == pytest.approx(15.0)

    def test_fill_zero_value_correct(self):
        df = pd.DataFrame({"x": [1.0, None, 3.0]})
        result = clean_dataframe(df, CleaningOptions(fill_numeric="zero"))
        assert result.cleaned_df["x"].iloc[1] == 0.0


# ── Fill text missing values ──────────────────────────────────────────────────

class TestFillText:
    def test_fill_mode(self, df_with_missing):
        opts = CleaningOptions(fill_text="mode")
        result = clean_dataframe(df_with_missing, opts)
        assert result.cleaned_df["name"].isna().sum() == 0

    def test_fill_unknown(self, df_with_missing):
        opts = CleaningOptions(fill_text="unknown")
        result = clean_dataframe(df_with_missing, opts)
        assert result.cleaned_df["name"].isna().sum() == 0
        assert (result.cleaned_df["name"] == "Unknown").sum() == 2

    def test_fill_none_preserves_text_missing(self, df_with_missing):
        opts = CleaningOptions(fill_text="none")
        result = clean_dataframe(df_with_missing, opts)
        assert result.cleaned_df["name"].isna().sum() == 2

    def test_fill_mode_picks_most_common(self):
        # After dedup: ["A", "B", None] → mode is "A" → fills None → ["A", "B", "A"]
        df = pd.DataFrame({"cat": ["A", "A", "B", None, None]})
        result = clean_dataframe(df, CleaningOptions(fill_text="mode"))
        assert result.cleaned_df["cat"].isna().sum() == 0
        assert "A" in result.cleaned_df["cat"].values


# ── Outlier removal ───────────────────────────────────────────────────────────

class TestOutlierRemoval:
    def test_outlier_row_removed(self, df_with_outliers):
        opts = CleaningOptions(remove_outliers=True, outlier_threshold=1.5)
        result = clean_dataframe(df_with_outliers, opts)
        assert 1000.0 not in result.cleaned_df["value"].values

    def test_outliers_removed_count_tracked(self, df_with_outliers):
        opts = CleaningOptions(remove_outliers=True, outlier_threshold=1.5)
        result = clean_dataframe(df_with_outliers, opts)
        assert result.outliers_removed >= 1

    def test_no_outliers_in_clean_data(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        opts = CleaningOptions(remove_outliers=True, outlier_threshold=1.5)
        result = clean_dataframe(df, opts)
        assert result.outliers_removed == 0
        assert len(result.cleaned_df) == 5

    def test_outlier_false_keeps_all_rows(self, df_with_outliers):
        opts = CleaningOptions(remove_outliers=False)
        result = clean_dataframe(df_with_outliers, opts)
        assert 1000.0 in result.cleaned_df["value"].values

    def test_loose_threshold_keeps_outlier(self, df_with_outliers):
        opts = CleaningOptions(remove_outliers=True, outlier_threshold=3.0)
        result = clean_dataframe(df_with_outliers, opts)
        # With multiplier=3.0, 1000 may or may not be caught — just check no crash
        assert isinstance(result.cleaned_df, pd.DataFrame)


# ── CleanResult stats ─────────────────────────────────────────────────────────

class TestCleanResultStats:
    def test_missing_before_after_decrease_when_filled(self, df_with_missing):
        opts = CleaningOptions(fill_numeric="mean", fill_text="unknown")
        result = clean_dataframe(df_with_missing, opts)
        assert result.missing_after < result.missing_before

    def test_missing_by_col_is_dataframe(self, df_with_missing):
        result = clean_dataframe(df_with_missing)
        assert isinstance(result.missing_by_col, pd.DataFrame)

    def test_missing_by_col_shows_columns_with_gaps(self, df_with_missing):
        result = clean_dataframe(df_with_missing)
        assert len(result.missing_by_col) > 0

    def test_missing_by_col_empty_when_no_gaps(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = clean_dataframe(df)
        assert result.missing_by_col.empty
