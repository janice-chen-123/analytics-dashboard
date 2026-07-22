"""
data_cleaning.py — DataFrame cleaning with user-configurable options.

Always-on steps (conservative baseline):
- Clean column names: strip whitespace, lowercase, spaces → underscores.
- Strip leading/trailing whitespace from string column values.
- Remove fully duplicate rows.

Optional steps (controlled via CleaningOptions):
- Fill numeric missing values (mean / median / zero).
- Fill text missing values (most common value / "Unknown").
- Remove outliers using the IQR method.
"""

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class CleaningOptions:
    """User-configurable cleaning behaviour."""
    fill_numeric: str = "none"       # "none" | "mean" | "median" | "zero"
    fill_text: str = "none"          # "none" | "mode" | "unknown"
    remove_outliers: bool = False
    outlier_threshold: float = 1.5   # IQR multiplier (1.5 = standard, 3.0 = loose)


@dataclass
class CleanResult:
    """Result returned by clean_dataframe()."""
    cleaned_df: pd.DataFrame
    changes: list[str] = field(default_factory=list)
    column_mapping: dict = field(default_factory=dict)
    rows_before: int = 0
    rows_after: int = 0
    missing_before: int = 0
    missing_after: int = 0
    outliers_removed: int = 0
    missing_by_col: pd.DataFrame = field(default_factory=pd.DataFrame)


def clean_column_name(name: str) -> str:
    """
    Normalize a single column name.

    Steps: strip whitespace → lowercase → replace spaces with underscores.
    """
    return name.strip().lower().replace(" ", "_")


def clean_dataframe(
    df: pd.DataFrame,
    options: CleaningOptions | None = None,
) -> CleanResult:
    """
    Apply cleaning to a DataFrame and return the result.

    Always-on steps (order matters):
    1. Clean column names.
    2. Strip whitespace from string values.
    3. Remove fully duplicate rows.

    Optional steps (applied after baseline):
    4. Fill numeric missing values.
    5. Fill text missing values.
    6. Remove outliers (IQR method).

    Args:
        df:      The raw DataFrame to clean. Never modified in place.
        options: Cleaning behaviour. Defaults to conservative (no fills, no outlier removal).

    Returns:
        CleanResult with the cleaned DataFrame, a detailed changes log,
        and before/after statistics.
    """
    if options is None:
        options = CleaningOptions()

    if df.empty:
        return CleanResult(
            cleaned_df=df.copy(),
            changes=["Input DataFrame is empty — no cleaning applied."],
            column_mapping={},
        )

    cleaned = df.copy()
    changes: list[str] = []
    rows_before = len(cleaned)
    missing_before = int(cleaned.isna().sum().sum())

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
            f"Stripped whitespace from {len(stripped_cols)} string column(s): "
            f"{', '.join(stripped_cols)}."
        )
    else:
        changes.append("No leading/trailing whitespace found in string columns.")

    # ── Step 3: Remove duplicate rows ─────────────────────────────────────────
    n_before_dedup = len(cleaned)
    cleaned = cleaned.drop_duplicates(keep="first").reset_index(drop=True)
    n_removed_dedup = n_before_dedup - len(cleaned)

    if n_removed_dedup > 0:
        changes.append(f"Removed {n_removed_dedup} fully duplicate row(s).")
    else:
        changes.append("No fully duplicate rows found.")

    # ── Step 4: Fill numeric missing values ───────────────────────────────────
    if options.fill_numeric != "none":
        numeric_cols = cleaned.select_dtypes(include="number").columns.tolist()
        filled_count = 0
        for col in numeric_cols:
            n_missing = int(cleaned[col].isna().sum())
            if n_missing == 0:
                continue
            if options.fill_numeric == "mean":
                fill_val = cleaned[col].mean()
            elif options.fill_numeric == "median":
                fill_val = cleaned[col].median()
            else:  # "zero"
                fill_val = 0
            cleaned[col] = cleaned[col].fillna(fill_val)
            filled_count += n_missing

        if filled_count > 0:
            label = {
                "mean": "column mean",
                "median": "column median",
                "zero": "0",
            }[options.fill_numeric]
            changes.append(
                f"Filled {filled_count} missing value(s) in numeric columns "
                f"using {label}."
            )
        else:
            changes.append("No missing values found in numeric columns.")

    # ── Step 5: Fill text missing values ──────────────────────────────────────
    if options.fill_text != "none":
        text_cols = cleaned.select_dtypes(include="object").columns.tolist()
        filled_count = 0
        for col in text_cols:
            n_missing = int(cleaned[col].isna().sum())
            if n_missing == 0:
                continue
            if options.fill_text == "mode":
                mode_vals = cleaned[col].mode()
                fill_val = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"
            else:  # "unknown"
                fill_val = "Unknown"
            cleaned[col] = cleaned[col].fillna(fill_val)
            filled_count += n_missing

        if filled_count > 0:
            label = "most common value" if options.fill_text == "mode" else '"Unknown"'
            changes.append(
                f"Filled {filled_count} missing value(s) in text columns "
                f"using {label}."
            )
        else:
            changes.append("No missing values found in text columns.")

    # ── Step 6: Remove outliers (IQR method) ──────────────────────────────────
    outliers_removed = 0
    if options.remove_outliers:
        numeric_cols = cleaned.select_dtypes(include="number").columns.tolist()
        outlier_mask = pd.Series(False, index=cleaned.index)

        for col in numeric_cols:
            col_data = cleaned[col].dropna()
            if len(col_data) < 4:
                continue
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - options.outlier_threshold * iqr
            upper = q3 + options.outlier_threshold * iqr
            outlier_mask |= (cleaned[col] < lower) | (cleaned[col] > upper)

        outliers_removed = int(outlier_mask.sum())
        if outliers_removed > 0:
            cleaned = cleaned[~outlier_mask].reset_index(drop=True)
            changes.append(
                f"Removed {outliers_removed} outlier row(s) "
                f"(IQR × {options.outlier_threshold})."
            )
        else:
            changes.append(
                f"No outliers found (IQR × {options.outlier_threshold})."
            )

    # ── Build per-column missing value comparison ─────────────────────────────
    missing_by_col = _build_missing_comparison(df, cleaned, col_mapping)

    return CleanResult(
        cleaned_df=cleaned,
        changes=changes,
        column_mapping=col_mapping,
        rows_before=rows_before,
        rows_after=len(cleaned),
        missing_before=missing_before,
        missing_after=int(cleaned.isna().sum().sum()),
        outliers_removed=outliers_removed,
        missing_by_col=missing_by_col,
    )


def _build_missing_comparison(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    col_mapping: dict,
) -> pd.DataFrame:
    """Build a per-column before/after missing value comparison table."""
    rows = []
    for orig_col, clean_col in col_mapping.items():
        before = int(df_raw[orig_col].isna().sum()) if orig_col in df_raw.columns else 0
        after = int(df_clean[clean_col].isna().sum()) if clean_col in df_clean.columns else 0
        if before > 0 or after > 0:
            rows.append({
                "Column": clean_col,
                "Missing Before": before,
                "Missing After": after,
                "Filled": before - after if before > after else 0,
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Column", "Missing Before", "Missing After", "Filled"]
    )
