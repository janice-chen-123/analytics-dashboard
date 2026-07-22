"""Tests for src/data_cleaning.py."""

import pandas as pd
import pytest

from src.data_cleaning import clean_column_name, clean_dataframe


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


# ── clean_dataframe ───────────────────────────────────────────────────────────

class TestCleanDataframe:
    @pytest.fixture
    def sample_df(self):
        """DataFrame with dirty column names, duplicate rows, and string whitespace."""
        return pd.DataFrame(
            {
                "Order Date": ["2024-01-01", "2024-02-01", "2024-02-01"],
                "Category ": [" Tech", " Furniture", " Furniture"],
                "Sales": [100.0, 200.0, 200.0],
            }
        )

    def test_column_names_are_lowercased(self, sample_df):
        result = clean_dataframe(sample_df)
        for col in result.cleaned_df.columns:
            assert col == col.lower(), f"Column '{col}' is not lowercase"

    def test_column_name_spaces_replaced_with_underscores(self, sample_df):
        result = clean_dataframe(sample_df)
        assert "order_date" in result.cleaned_df.columns

    def test_column_name_trailing_space_stripped(self, sample_df):
        result = clean_dataframe(sample_df)
        assert "category" in result.cleaned_df.columns
        assert "category " not in result.cleaned_df.columns

    def test_duplicate_rows_removed(self, sample_df):
        result = clean_dataframe(sample_df)
        # Rows 1 and 2 become identical after stripping → only 1 kept
        assert len(result.cleaned_df) == 2

    def test_string_whitespace_stripped(self, sample_df):
        result = clean_dataframe(sample_df)
        for val in result.cleaned_df["category"].dropna():
            assert val == val.strip(), f"Value '{val}' still has whitespace"

    def test_original_dataframe_not_modified(self, sample_df):
        original_cols = sample_df.columns.tolist()
        original_len = len(sample_df)
        snapshot = sample_df.copy()

        clean_dataframe(sample_df)

        assert sample_df.columns.tolist() == original_cols
        assert len(sample_df) == original_len
        pd.testing.assert_frame_equal(sample_df, snapshot)

    def test_missing_values_are_preserved(self):
        df = pd.DataFrame(
            {
                "name": ["Alice", None, "Bob"],
                "sales": [100.0, None, 300.0],
            }
        )
        result = clean_dataframe(df)
        assert result.cleaned_df["sales"].isna().sum() == 1
        assert result.cleaned_df["name"].isna().sum() == 1

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

    def test_no_duplicates_in_clean_data_stays_same_length(self):
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "value": ["a", "b", "c"],
            }
        )
        result = clean_dataframe(df)
        assert len(result.cleaned_df) == 3

    def test_dataframe_with_only_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 1], "b": ["x", "x", "x"]})
        result = clean_dataframe(df)
        assert len(result.cleaned_df) == 1
