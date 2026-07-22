"""
data_quality.py — Data quality statistics.

Computes per-column and overall quality metrics from a DataFrame.
These functions are pure Python — no Streamlit dependency.
"""

import pandas as pd


def generate_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-column quality statistics table.

    Args:
        df: Any DataFrame (raw or cleaned).

    Returns:
        A DataFrame with one row per column containing:
        Column, Data Type, Non-Null Count, Missing Count, Missing %, Unique Values.
        Returns an empty DataFrame if the input is empty.
    """
    if df.empty:
        return pd.DataFrame()

    n_rows = len(df)
    rows = []
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        missing_pct = round((missing_count / n_rows) * 100, 1) if n_rows > 0 else 0.0
        unique_count = int(df[col].nunique(dropna=True))
        non_null = int(df[col].notna().sum())

        rows.append(
            {
                "Column": col,
                "Data Type": str(df[col].dtype),
                "Non-Null Count": non_null,
                "Missing Count": missing_count,
                "Missing %": missing_pct,
                "Unique Values": unique_count,
            }
        )

    return pd.DataFrame(rows)


def get_quality_summary(df: pd.DataFrame) -> dict:
    """
    Compute overall dataset-level quality metrics.

    Args:
        df: Any DataFrame (raw or cleaned).

    Returns:
        Dict with keys:
          row_count, col_count, duplicate_count,
          total_missing, total_missing_pct.
        All counts are 0 for an empty DataFrame.
    """
    if df.empty:
        return {
            "row_count": 0,
            "col_count": 0,
            "duplicate_count": 0,
            "total_missing": 0,
            "total_missing_pct": 0.0,
        }

    total_cells = df.shape[0] * df.shape[1]
    total_missing = int(df.isna().sum().sum())
    missing_pct = round((total_missing / total_cells) * 100, 1) if total_cells > 0 else 0.0

    return {
        "row_count": len(df),
        "col_count": len(df.columns),
        "duplicate_count": int(df.duplicated().sum()),
        "total_missing": total_missing,
        "total_missing_pct": missing_pct,
    }
