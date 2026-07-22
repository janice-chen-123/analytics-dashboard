"""
analytics.py — KPI calculation from a filtered DataFrame.

Design principle: Python calculates the numbers. AI explains the numbers.

All functions are pure Python — no Streamlit dependency, fully testable.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class KPIResult:
    """
    All KPI values computed from a single calculate_kpis() call.

    Fields are None when:
    - The required column was not configured by the user.
    - The column has no non-null values.
    - Division by zero would occur (profit_margin_pct, avg_order_value).

    has_order_id: True when an Order ID column was configured, which changes
    the label and denominator for avg_order_value in the UI.
    """

    total_sales: Optional[float]
    total_profit: Optional[float]
    profit_margin_pct: Optional[float]   # e.g. 23.5 means 23.5%
    avg_order_value: Optional[float]
    num_records: int
    num_orders: Optional[int]            # None when no order_id column
    total_quantity: Optional[float]
    has_order_id: bool


def _safe_sum(df: pd.DataFrame, col: Optional[str]) -> Optional[float]:
    """
    Sum a column safely.

    Returns None if:
    - col is None (not configured)
    - col does not exist in df
    - The column has zero non-null values

    Returns the numeric sum otherwise (may be 0.0 if all values are 0).
    """
    if not col or col not in df.columns:
        return None
    series = df[col]
    if series.notna().sum() == 0:
        return None
    return float(series.sum())


def calculate_kpis(df: pd.DataFrame, col_config: dict) -> KPIResult:
    """
    Compute all KPIs from a filtered DataFrame using user column mapping.

    Args:
        df:         The filtered DataFrame (may be empty).
        col_config: Maps role strings to column name strings or None.
                    Expected keys: sales, profit, quantity, order_id,
                                   date, category, region.

    Returns:
        KPIResult with all computed metrics.
        Fields are None when data or configuration is insufficient.

    Notes:
        - Profit Margin = Total Profit / Total Sales × 100
        - If Sales = 0, Profit Margin returns None (avoids division by zero).
        - Avg Order Value = Total Sales / Unique Orders  (when order_id set)
                          = Total Sales / Record Count   (when order_id not set)
        - Missing values in numeric columns are ignored by sum (skipna=True).
    """
    sales_col = col_config.get("sales")
    profit_col = col_config.get("profit")
    quantity_col = col_config.get("quantity")
    order_id_col = col_config.get("order_id")

    has_order_id = bool(order_id_col and order_id_col in (df.columns if not df.empty else []))

    if df.empty:
        return KPIResult(
            total_sales=None,
            total_profit=None,
            profit_margin_pct=None,
            avg_order_value=None,
            num_records=0,
            num_orders=0 if has_order_id else None,
            total_quantity=None,
            has_order_id=has_order_id,
        )

    num_records: int = len(df)

    # ── Sums ──────────────────────────────────────────────────────────────────
    total_sales = _safe_sum(df, sales_col)
    total_profit = _safe_sum(df, profit_col)
    total_quantity = _safe_sum(df, quantity_col)

    # ── Profit margin ─────────────────────────────────────────────────────────
    profit_margin_pct: Optional[float] = None
    if total_sales is not None and total_profit is not None and total_sales != 0:
        profit_margin_pct = round(total_profit / total_sales * 100, 2)

    # ── Order count ───────────────────────────────────────────────────────────
    num_orders: Optional[int] = None
    if has_order_id:
        num_orders = int(df[order_id_col].nunique())

    # ── Average order / record value ──────────────────────────────────────────
    avg_order_value: Optional[float] = None
    if total_sales is not None:
        if has_order_id and num_orders and num_orders > 0:
            avg_order_value = round(total_sales / num_orders, 2)
        elif not has_order_id and num_records > 0:
            avg_order_value = round(total_sales / num_records, 2)

    return KPIResult(
        total_sales=round(total_sales, 2) if total_sales is not None else None,
        total_profit=round(total_profit, 2) if total_profit is not None else None,
        profit_margin_pct=profit_margin_pct,
        avg_order_value=avg_order_value,
        num_records=num_records,
        num_orders=num_orders,
        total_quantity=round(total_quantity, 2) if total_quantity is not None else None,
        has_order_id=has_order_id,
    )
