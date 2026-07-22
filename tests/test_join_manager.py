"""Tests for src/join_manager.py."""

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.join_manager import (
    JoinError,
    execute_join,
    get_sqlite_tables,
    get_table_columns,
    load_and_join_sqlite,
    sanitize_table_name,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def df_left():
    return pd.DataFrame({
        "order_id": ["A1", "A2", "A3"],
        "sales":    [100.0, 200.0, 300.0],
    })


@pytest.fixture
def df_right():
    return pd.DataFrame({
        "order_id": ["A1", "A2", "A4"],
        "category": ["Tech", "Office", "Sports"],
    })


@pytest.fixture
def tmp_db(tmp_path):
    """SQLite DB with two tables for testing."""
    db = tmp_path / "test.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE orders (order_id TEXT, sales REAL)")
        conn.executemany("INSERT INTO orders VALUES (?,?)",
                         [("A1", 100.0), ("A2", 200.0), ("A3", 300.0)])
        conn.execute("CREATE TABLE details (order_id TEXT, category TEXT)")
        conn.executemany("INSERT INTO details VALUES (?,?)",
                         [("A1", "Tech"), ("A2", "Office"), ("A4", "Sports")])
    return db


# ── execute_join ──────────────────────────────────────────────────────────────

class TestExecuteJoin:
    def test_inner_join_returns_matching_rows(self, df_left, df_right):
        result = execute_join(df_left, df_right, "inner", "order_id", "order_id")
        assert len(result) == 2
        assert set(result["order_id"]) == {"A1", "A2"}

    def test_left_join_keeps_all_left_rows(self, df_left, df_right):
        result = execute_join(df_left, df_right, "left", "order_id", "order_id")
        assert len(result) == 3
        assert "A3" in result["order_id"].values

    def test_right_join_keeps_all_right_rows(self, df_left, df_right):
        result = execute_join(df_left, df_right, "right", "order_id", "order_id")
        assert len(result) == 3
        assert "A4" in result["order_id"].values

    def test_outer_join_keeps_all_rows(self, df_left, df_right):
        result = execute_join(df_left, df_right, "outer", "order_id", "order_id")
        assert len(result) == 4

    def test_result_has_columns_from_both(self, df_left, df_right):
        result = execute_join(df_left, df_right, "inner", "order_id", "order_id")
        assert "sales" in result.columns
        assert "category" in result.columns

    def test_missing_left_key_raises(self, df_left, df_right):
        with pytest.raises(JoinError, match="not found in the left table"):
            execute_join(df_left, df_right, "inner", "nonexistent", "order_id")

    def test_missing_right_key_raises(self, df_left, df_right):
        with pytest.raises(JoinError, match="not found in the right table"):
            execute_join(df_left, df_right, "inner", "order_id", "nonexistent")

    def test_invalid_how_raises(self, df_left, df_right):
        with pytest.raises(JoinError, match="Invalid join type"):
            execute_join(df_left, df_right, "cross", "order_id", "order_id")

    def test_overlapping_columns_get_suffixes(self, df_left):
        df_right2 = pd.DataFrame({"order_id": ["A1"], "sales": [999.0]})
        result = execute_join(df_left, df_right2, "inner", "order_id", "order_id")
        assert "sales_left" in result.columns
        assert "sales_right" in result.columns


# ── get_sqlite_tables ─────────────────────────────────────────────────────────

class TestGetSqliteTables:
    def test_returns_table_names(self, tmp_db):
        tables = get_sqlite_tables(tmp_db)
        assert "orders" in tables
        assert "details" in tables

    def test_returns_empty_list_if_db_missing(self, tmp_path):
        tables = get_sqlite_tables(tmp_path / "nonexistent.db")
        assert tables == []


# ── get_table_columns ─────────────────────────────────────────────────────────

class TestGetTableColumns:
    def test_returns_column_names(self, tmp_db):
        cols = get_table_columns("orders", tmp_db)
        assert "order_id" in cols
        assert "sales" in cols

    def test_invalid_table_raises(self, tmp_db):
        with pytest.raises(JoinError, match="not found"):
            get_table_columns("nonexistent", tmp_db)


# ── load_and_join_sqlite ──────────────────────────────────────────────────────

class TestLoadAndJoinSqlite:
    def test_inner_join_sqlite(self, tmp_db):
        result = load_and_join_sqlite("orders", "details", "inner", "order_id", "order_id", tmp_db)
        assert len(result) == 2
        assert "category" in result.columns
        assert "sales" in result.columns

    def test_left_join_sqlite(self, tmp_db):
        result = load_and_join_sqlite("orders", "details", "left", "order_id", "order_id", tmp_db)
        assert len(result) == 3

    def test_missing_db_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_and_join_sqlite("a", "b", "inner", "id", "id", tmp_path / "missing.db")

    def test_invalid_table_raises(self, tmp_db):
        with pytest.raises(JoinError):
            load_and_join_sqlite("orders", "ghost", "inner", "order_id", "order_id", tmp_db)

    def test_invalid_column_raises(self, tmp_db):
        with pytest.raises(JoinError):
            load_and_join_sqlite("orders", "details", "inner", "bad_col", "order_id", tmp_db)


# ── sanitize_table_name ───────────────────────────────────────────────────────

class TestSanitizeTableName:
    def test_basic_filename(self):
        assert sanitize_table_name("sales.csv") == "uploaded_sales"

    def test_spaces_replaced(self):
        assert sanitize_table_name("my data.xlsx") == "uploaded_my_data"

    def test_uppercase_lowercased(self):
        assert sanitize_table_name("SalesData.csv") == "uploaded_salesdata"

    def test_special_chars_removed(self):
        assert sanitize_table_name("P4 Bookings (Clean).csv") == "uploaded_p4_bookings_clean"

    def test_prefix_always_present(self):
        name = sanitize_table_name("anything.csv")
        assert name.startswith("uploaded_")
