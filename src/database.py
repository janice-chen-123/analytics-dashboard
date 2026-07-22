"""
database.py — SQLite persistence layer for the AI Data Analyst Dashboard.

Design rules:
- Only SELECT queries are permitted through run_query().
- save_dataframe() always replaces the existing table (Phase 1 behaviour).
- No Streamlit imports — fully testable in isolation.
- The database file lives at database/analytics.db relative to the project root.
"""

import re
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

# ── Path configuration ────────────────────────────────────────────────────────

# Resolved relative to this file's location: src/ → project root → database/
_PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DB_PATH = _PROJECT_ROOT / "database" / "analytics.db"
DEFAULT_TABLE = "sales_data"

# SQL keywords that must never be executed through run_query()
_FORBIDDEN_KEYWORDS = {"delete", "drop", "update", "insert", "alter", "create", "replace", "truncate"}


# ── Exceptions ────────────────────────────────────────────────────────────────

class UnsafeSQLError(ValueError):
    """Raised when run_query() receives a non-SELECT or otherwise dangerous statement."""


# ── Public API ────────────────────────────────────────────────────────────────

def save_dataframe(
    df: pd.DataFrame,
    table_name: str = DEFAULT_TABLE,
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    """
    Write a DataFrame to a SQLite table, replacing any existing data.

    Args:
        df:         The cleaned DataFrame to persist.
        table_name: Target SQLite table name (default: 'sales_data').
        db_path:    Path to the SQLite file (created if it does not exist).

    Returns:
        Number of rows written.

    Raises:
        ValueError: If df is empty.
        sqlite3.Error: On any database-level failure.
    """
    if df.empty:
        raise ValueError("Cannot save an empty DataFrame to the database.")

    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        df.to_sql(
            name=table_name,
            con=conn,
            if_exists="replace",
            index=False,
        )

    return len(df)


def run_query(
    sql: str,
    db_path: Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """
    Execute a read-only SELECT query and return the result as a DataFrame.

    Only SELECT statements are allowed. Any query that starts with a forbidden
    keyword (DELETE, DROP, UPDATE, INSERT, ALTER, CREATE, REPLACE, TRUNCATE)
    is rejected before reaching the database.

    Note: This is a first-version safety check, not a complete SQL injection
    defence. Do not expose run_query() to arbitrary user input in production.

    Args:
        sql:     The SQL SELECT statement to execute.
        db_path: Path to the SQLite file.

    Returns:
        A DataFrame containing the query results (may be empty).

    Raises:
        UnsafeSQLError: If the statement is not a SELECT or contains forbidden keywords.
        FileNotFoundError: If the database file does not exist.
        sqlite3.Error: On any database-level failure.
    """
    _validate_sql(sql)

    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at '{db_path}'. "
            "Save a dataset first using save_dataframe()."
        )

    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(sql, conn)


def load_table(
    table_name: str,
    db_path: Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """
    Load an entire SQLite table as a DataFrame.

    Args:
        table_name: Name of the table to load.
        db_path:    Path to the SQLite file.

    Returns:
        DataFrame with all rows from the table.

    Raises:
        FileNotFoundError: If the database does not exist.
        UnsafeSQLError: Delegates to run_query for safety.
    """
    return run_query(f'SELECT * FROM "{table_name}"', db_path=db_path)


def get_table_info(
    table_name: str = DEFAULT_TABLE,
    db_path: Path = DEFAULT_DB_PATH,
) -> Optional[pd.DataFrame]:
    """
    Return column names and types for a table, or None if the table does not exist.

    Args:
        table_name: Name of the table to inspect.
        db_path:    Path to the SQLite file.

    Returns:
        DataFrame with columns [name, type] or None.
    """
    if not db_path.exists():
        return None
    try:
        return run_query(f"SELECT * FROM pragma_table_info('{table_name}')", db_path=db_path)
    except Exception:
        return None


# ── Internal helpers ──────────────────────────────────────────────────────────

def _validate_sql(sql: str) -> None:
    """
    Raise UnsafeSQLError if sql is not a safe SELECT statement.

    Checks performed (case-insensitive, after stripping whitespace):
    1. Statement must start with 'select'.
    2. Statement must not contain any forbidden keywords as whole words.
    """
    if not sql or not sql.strip():
        raise UnsafeSQLError("SQL statement is empty.")

    normalised = sql.strip().lower()

    if not normalised.startswith("select"):
        raise UnsafeSQLError(
            f"Only SELECT statements are allowed. "
            f"Received: '{sql.strip()[:60]}'"
        )

    # Check for forbidden keywords anywhere in the statement (whole-word match).
    # This catches e.g. "SELECT * FROM t; DROP TABLE t"
    for keyword in _FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", normalised):
            raise UnsafeSQLError(
                f"Forbidden keyword '{keyword.upper()}' detected in SQL statement."
            )
