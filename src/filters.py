"""
filters.py — Column mapping configuration and sidebar filter helpers.

Responsibilities:
- Identify numeric vs non-numeric columns from a DataFrame
- Build the column mapping selector (which column is sales, date, etc.)
- Validate that selected columns exist in the DataFrame
- Apply date / category / region filters to produce a filtered DataFrame
"""

from typing import Optional
import pandas as pd


# ── Column type helpers ───────────────────────────────────────────────────────

def col_has_data(df: pd.DataFrame, col: Optional[str]) -> bool:
    """
    Return True if col is non-None, exists in df, and has at least one non-null value.

    Used by visualizations and ai_summary to guard against missing/empty columns
    without raising KeyError or returning misleading empty charts.
    """
    return bool(col and col in df.columns and df[col].notna().any())


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return column names whose dtype is numeric (int or float)."""
    return df.select_dtypes(include=["number"]).columns.tolist()


def get_all_columns(df: pd.DataFrame) -> list[str]:
    """Return all column names."""
    return df.columns.tolist()


# ── Column config validation ──────────────────────────────────────────────────

def validate_col_config(df: pd.DataFrame, col_config: dict) -> dict[str, str]:
    """
    Remove any column selections that no longer exist in df.

    This protects against stale config when a new file is uploaded.

    Args:
        df:         The current (cleaned) DataFrame.
        col_config: The user's column mapping dict from session state.

    Returns:
        A sanitized copy of col_config with invalid selections set to None.
    """
    valid = {}
    existing = set(df.columns.tolist())
    for role, col in col_config.items():
        valid[role] = col if (col is not None and col in existing) else None
    return valid


# ── Filter application ────────────────────────────────────────────────────────

def apply_filters(
    df: pd.DataFrame,
    col_config: dict,
    selected_regions: Optional[list[str]],
    selected_categories: Optional[list[str]],
    date_start: Optional[object],
    date_end: Optional[object],
) -> pd.DataFrame:
    """
    Apply sidebar filter selections to a DataFrame.

    Filters are applied only when:
    - The corresponding column is configured in col_config
    - The selection is not empty / None

    Missing values in filter columns are kept (not silently dropped).

    Args:
        df:                   The cleaned DataFrame.
        col_config:           User column mapping (role → column name).
        selected_regions:     List of region values to keep, or None for all.
        selected_categories:  List of category values to keep, or None for all.
        date_start:           Start date (datetime.date), or None for no lower bound.
        date_end:             End date (datetime.date), or None for no upper bound.

    Returns:
        Filtered DataFrame (may be the same object if no filters applied).
    """
    filtered = df.copy()

    region_col = col_config.get("region")
    category_col = col_config.get("category")
    date_col = col_config.get("date")

    # ── Region filter ─────────────────────────────────────────────────────────
    if region_col and region_col in filtered.columns and selected_regions:
        filtered = filtered[filtered[region_col].isin(selected_regions)]

    # ── Category filter ───────────────────────────────────────────────────────
    if category_col and category_col in filtered.columns and selected_categories:
        filtered = filtered[filtered[category_col].isin(selected_categories)]

    # ── Date range filter ─────────────────────────────────────────────────────
    if date_col and date_col in filtered.columns and (date_start or date_end):
        parsed = pd.to_datetime(filtered[date_col], errors="coerce")
        mask = pd.Series([True] * len(filtered), index=filtered.index)
        if date_start:
            mask &= parsed >= pd.Timestamp(date_start)
        if date_end:
            mask &= parsed <= pd.Timestamp(date_end)
        filtered = filtered[mask]

    return filtered.reset_index(drop=True)
