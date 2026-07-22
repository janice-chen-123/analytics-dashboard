"""
data_cleaning.py — Conservative DataFrame cleaning.

Rules:
- Never modify the original DataFrame (always work on a copy).
- Never drop rows with missing values.
- Never auto-fill missing values.
- Clean column names: strip whitespace, lowercase, spaces → underscores.
- Remove fully duplicate rows.
- Strip leading/trailing whitespace from string column values.
"""

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class CleanResult:
    """
    Result returned by clean_dataframe().

    cleaned_df:     The cleaned copy of the original DataFrame.
    changes:        Human-readable list of what was changed (shown in the UI).
    column_mapping: Maps original column name → cleaned column name.
    """

    cleaned_df: pd.DataFrame
    changes: list[str] = field(default_factory=list)
    column_mapping: dict[str, str] = field(default_factory=dict)


def clean_column_name(name: str) -> str:
    """
    Normalize a single column name.

    Steps: strip whitespace → lowercase → replace spaces with underscores.

    Args:
        name: The original column name string.

    Returns:
        The cleaned column name string.
    """
    return name.strip().lower().replace(" ", "_")


def clean_dataframe(df: pd.DataFrame) -> CleanResult:
    """
    Apply conservative cleaning to a DataFrame and return the result.

    Order of operations:
    1. Clean column names (normalize first so later steps use clean names).
    2. Strip whitespace from string column values (normalize values before dedup).
    3. Remove fully duplicate rows (after normalization for best detection).

    Missing values are intentionally preserved. Rows with NaN are NOT removed.

    Args:
        df: The raw DataFrame to clean.

    Returns:
        CleanResult containing the cleaned DataFrame, a changes log, and
        a mapping of old → new column names.
    """
    if df.empty:
        return CleanResult(
            cleaned_df=df.copy(),
            changes=["Input DataFrame is empty — no cleaning applied."],
            column_mapping={},
        )

    cleaned = df.copy()
    changes: list[str] = []

    # ── Step 1: Clean column names ────────────────────────────────────────────
    original_cols = cleaned.columns.tolist()
    new_cols = [clean_column_name(col) for col in original_cols]
    col_mapping = dict(zip(original_cols, new_cols))

    renamed = [(old, new) for old, new in col_mapping.items() if old != new]
    if renamed:
        cleaned.columns = new_cols
        renamed_str = ", ".join(f'"{o}" → "{n}"' for o, n in renamed)
        changes.append(f"Renamed {len(renamed)} column(s): {renamed_str}.")
    else:
        changes.append("Column names already clean — no renaming needed.")

    # ── Step 2: Strip whitespace from string columns ──────────────────────────
    stripped_cols: list[str] = []
    for col in cleaned.columns:
        if cleaned[col].dtype == object:
            before = cleaned[col].copy()
            cleaned[col] = cleaned[col].str.strip()
            if not before.equals(cleaned[col]):
                stripped_cols.append(col)

    if stripped_cols:
        changes.append(
            f"Stripped leading/trailing whitespace from {len(stripped_cols)} "
            f"string column(s): {', '.join(stripped_cols)}."
        )
    else:
        changes.append("No leading/trailing whitespace found in string columns.")

    # ── Step 3: Remove duplicate rows ─────────────────────────────────────────
    n_before = len(cleaned)
    cleaned = cleaned.drop_duplicates(keep="first").reset_index(drop=True)
    n_removed = n_before - len(cleaned)

    if n_removed > 0:
        changes.append(f"Removed {n_removed} fully duplicate row(s).")
    else:
        changes.append("No fully duplicate rows found.")

    return CleanResult(
        cleaned_df=cleaned,
        changes=changes,
        column_mapping=col_mapping,
    )
