"""Tests for src/database.py."""

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.database import (
    DEFAULT_TABLE,
    UnsafeSQLError,
    _validate_sql,
    run_query,
    save_dataframe,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Return a path inside pytest's temporary directory — auto-cleaned after each test."""
    return tmp_path / "test_analytics.db"


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["A001", "A002", "A003"],
            "category": ["Tech", "Office", "Tech"],
            "sales": [100.0, 200.0, 150.0],
            "profit": [20.0, -5.0, 30.0],
        }
    )


# ── save_dataframe ─────────────────────────────────────────────────────────────

class TestSaveDataframe:
    def test_returns_row_count(self, sample_df, tmp_db):
        n = save_dataframe(sample_df, db_path=tmp_db)
        assert n == 3

    def test_db_file_is_created(self, sample_df, tmp_db):
        save_dataframe(sample_df, db_path=tmp_db)
        assert tmp_db.exists()

    def test_data_is_readable_after_save(self, sample_df, tmp_db):
        save_dataframe(sample_df, db_path=tmp_db)
        result = run_query(f"SELECT * FROM {DEFAULT_TABLE}", db_path=tmp_db)
        assert len(result) == 3
        assert list(result.columns) == ["order_id", "category", "sales", "profit"]

    def test_index_is_not_saved(self, sample_df, tmp_db):
        save_dataframe(sample_df, db_path=tmp_db)
        result = run_query(f"SELECT * FROM {DEFAULT_TABLE}", db_path=tmp_db)
        assert "index" not in result.columns

    def test_replace_overwrites_old_data(self, sample_df, tmp_db):
        save_dataframe(sample_df, db_path=tmp_db)
        new_df = pd.DataFrame({"order_id": ["X1"], "sales": [999.0]})
        save_dataframe(new_df, db_path=tmp_db)
        result = run_query(f"SELECT * FROM {DEFAULT_TABLE}", db_path=tmp_db)
        assert len(result) == 1
        assert result["order_id"].iloc[0] == "X1"

    def test_empty_dataframe_raises(self, tmp_db):
        with pytest.raises(ValueError, match="empty"):
            save_dataframe(pd.DataFrame(), db_path=tmp_db)

    def test_custom_table_name(self, sample_df, tmp_db):
        save_dataframe(sample_df, table_name="custom_table", db_path=tmp_db)
        result = run_query("SELECT * FROM custom_table", db_path=tmp_db)
        assert len(result) == 3

    def test_values_are_correct(self, sample_df, tmp_db):
        save_dataframe(sample_df, db_path=tmp_db)
        result = run_query(f"SELECT sales FROM {DEFAULT_TABLE} ORDER BY sales", db_path=tmp_db)
        assert list(result["sales"]) == [100.0, 150.0, 200.0]


# ── run_query ─────────────────────────────────────────────────────────────────

class TestRunQuery:
    def test_select_all_returns_dataframe(self, sample_df, tmp_db):
        save_dataframe(sample_df, db_path=tmp_db)
        result = run_query(f"SELECT * FROM {DEFAULT_TABLE}", db_path=tmp_db)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_select_with_where_clause(self, sample_df, tmp_db):
        save_dataframe(sample_df, db_path=tmp_db)
        result = run_query(
            f"SELECT * FROM {DEFAULT_TABLE} WHERE sales > 120", db_path=tmp_db
        )
        assert len(result) == 2

    def test_select_aggregation(self, sample_df, tmp_db):
        save_dataframe(sample_df, db_path=tmp_db)
        result = run_query(
            f"SELECT SUM(sales) AS total FROM {DEFAULT_TABLE}", db_path=tmp_db
        )
        assert result["total"].iloc[0] == 450.0

    def test_db_not_found_raises_file_not_found(self, tmp_db):
        with pytest.raises(FileNotFoundError):
            run_query("SELECT 1", db_path=tmp_db)

    def test_select_case_insensitive(self, sample_df, tmp_db):
        save_dataframe(sample_df, db_path=tmp_db)
        result = run_query(f"SELECT * FROM {DEFAULT_TABLE}", db_path=tmp_db)
        assert len(result) == 3


# ── SQL validation (_validate_sql) ────────────────────────────────────────────

class TestValidateSQL:
    """Unit-tests for the internal validator — no DB required."""

    def test_valid_select_passes(self):
        _validate_sql("SELECT * FROM sales_data")  # should not raise

    def test_select_uppercase_passes(self):
        _validate_sql("SELECT id FROM sales_data WHERE id = 1")

    def test_select_with_leading_whitespace_passes(self):
        _validate_sql("  SELECT * FROM sales_data  ")

    def test_empty_string_raises(self):
        with pytest.raises(UnsafeSQLError):
            _validate_sql("")

    def test_whitespace_only_raises(self):
        with pytest.raises(UnsafeSQLError):
            _validate_sql("   ")

    # ── Forbidden statement types ─────────────────────────────────────────────

    def test_delete_raises(self):
        with pytest.raises(UnsafeSQLError):
            _validate_sql("DELETE FROM sales_data")

    def test_drop_raises(self):
        with pytest.raises(UnsafeSQLError):
            _validate_sql("DROP TABLE sales_data")

    def test_update_raises(self):
        with pytest.raises(UnsafeSQLError):
            _validate_sql("UPDATE sales_data SET sales = 0")

    def test_insert_raises(self):
        with pytest.raises(UnsafeSQLError):
            _validate_sql("INSERT INTO sales_data VALUES (1)")

    def test_alter_raises(self):
        with pytest.raises(UnsafeSQLError):
            _validate_sql("ALTER TABLE sales_data ADD COLUMN x INT")

    def test_truncate_raises(self):
        with pytest.raises(UnsafeSQLError):
            _validate_sql("TRUNCATE TABLE sales_data")

    # ── Injection attempt after SELECT ───────────────────────────────────────

    def test_select_with_semicolon_drop_raises(self):
        with pytest.raises(UnsafeSQLError):
            _validate_sql("SELECT * FROM sales_data; DROP TABLE sales_data")

    def test_select_with_semicolon_delete_raises(self):
        with pytest.raises(UnsafeSQLError):
            _validate_sql("SELECT 1; DELETE FROM sales_data")
