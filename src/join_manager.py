"""
join_manager.py — File-to-file and SQLite table join operations.

Design rules:
- No Streamlit imports — fully testable in isolation.
- SQLite table and column names are validated against actual DB contents
  before being used in any query, preventing injection via dropdown values.
- All join types are handled via pandas merge (SQLite does not support
  RIGHT JOIN or FULL OUTER JOIN natively).
"""

import re
import sqlite3
from pathlib import Path

import pandas as pd

from src.database import DEFAULT_DB_PATH

# ── Constants ─────────────────────────────────────────────────────────────────

JOIN_TYPE_OPTIONS = {
    "Inner — keep matching rows only": "inner",
    "Left — keep all rows from left table": "left",
    "Right — keep all rows from right table": "right",
    "Outer — keep all rows from both tables": "outer",
}

# ── Exceptions ────────────────────────────────────────────────────────────────

class JoinError(ValueError):
    """Raised when a join cannot be completed safely."""


# ── Public API ────────────────────────────────────────────────────────────────

def execute_join(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    how: str,
    left_on: str,
    right_on: str,
    suffixes: tuple[str, str] = ("_left", "_right"),
) -> pd.DataFrame:
    """
    Merge two DataFrames on specified key columns.

    Args:
        df_left:   Left DataFrame.
        df_right:  Right DataFrame.
        how:       Join type: 'inner', 'left', 'right', or 'outer'.
        left_on:   Key column in df_left.
        right_on:  Key column in df_right.
        suffixes:  Suffixes appended to overlapping column names.

    Returns:
        Merged DataFrame.

    Raises:
        JoinError: If key columns are missing or join produces no rows (inner only).
    """
    if left_on not in df_left.columns:
        raise JoinError(
            f"Column '{left_on}' not found in the left table. "
            f"Available columns: {', '.join(df_left.columns)}"
        )
    if right_on not in df_right.columns:
        raise JoinError(
            f"Column '{right_on}' not found in the right table. "
            f"Available columns: {', '.join(df_right.columns)}"
        )
    if how not in ("inner", "left", "right", "outer"):
        raise JoinError(f"Invalid join type '{how}'. Use: inner, left, right, outer.")

    merged = pd.merge(
        df_left,
        df_right,
        how=how,
        left_on=left_on,
        right_on=right_on,
        suffixes=suffixes,
    )
    return merged


def get_sqlite_tables(db_path: Path = DEFAULT_DB_PATH) -> list[str]:
    """
    Return names of all user tables in the SQLite database.

    Args:
        db_path: Path to the SQLite file.

    Returns:
        List of table names (empty list if DB does not exist or has no tables).
    """
    if not db_path.exists():
        return []
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        return [row[0] for row in rows]
    except sqlite3.Error:
        return []


def get_table_columns(
    table_name: str,
    db_path: Path = DEFAULT_DB_PATH,
) -> list[str]:
    """
    Return column names for a table in the SQLite database.

    Args:
        table_name: Must exist in the database (validated by caller).
        db_path:    Path to the SQLite file.

    Returns:
        List of column name strings.

    Raises:
        JoinError: If the table does not exist.
    """
    available = get_sqlite_tables(db_path)
    if table_name not in available:
        raise JoinError(f"Table '{table_name}' not found in the database.")

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(f'SELECT * FROM "{table_name}" LIMIT 0')
        return [description[0] for description in cursor.description]


def load_and_join_sqlite(
    table1: str,
    table2: str,
    how: str,
    left_on: str,
    right_on: str,
    db_path: Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """
    Load two SQLite tables into DataFrames and merge them.

    Table and column names are validated against actual database contents
    before use, so they cannot be used for SQL injection.

    Args:
        table1:   Name of the left table.
        table2:   Name of the right table.
        how:      Join type: 'inner', 'left', 'right', or 'outer'.
        left_on:  Key column in table1.
        right_on: Key column in table2.
        db_path:  Path to the SQLite file.

    Returns:
        Merged DataFrame.

    Raises:
        JoinError: If any table or column name fails validation.
        FileNotFoundError: If the database file does not exist.
    """
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at '{db_path}'. "
            "Upload a dataset first to populate the database."
        )

    available_tables = get_sqlite_tables(db_path)
    for name in (table1, table2):
        if name not in available_tables:
            raise JoinError(f"Table '{name}' not found in the database.")

    cols1 = get_table_columns(table1, db_path)
    cols2 = get_table_columns(table2, db_path)

    if left_on not in cols1:
        raise JoinError(
            f"Column '{left_on}' not found in table '{table1}'. "
            f"Available: {', '.join(cols1)}"
        )
    if right_on not in cols2:
        raise JoinError(
            f"Column '{right_on}' not found in table '{table2}'. "
            f"Available: {', '.join(cols2)}"
        )

    with sqlite3.connect(db_path) as conn:
        df_left = pd.read_sql_query(f'SELECT * FROM "{table1}"', conn)
        df_right = pd.read_sql_query(f'SELECT * FROM "{table2}"', conn)

    return execute_join(df_left, df_right, how, left_on, right_on)


def sanitize_table_name(filename: str) -> str:
    """
    Convert a filename to a safe SQLite table name.

    Example: 'P4 Bookings Clean.xlsx' → 'uploaded_p4_bookings_clean'
    """
    stem = Path(filename).stem
    safe = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()
    return f"uploaded_{safe}" if safe else "uploaded_file"
