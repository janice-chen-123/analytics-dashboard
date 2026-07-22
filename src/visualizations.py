"""
visualizations.py — Plotly chart builders for the AI Data Analyst Dashboard.

Design rules:
- Every function returns a plotly Figure or None (never raises on empty/missing data).
- All sorting, aggregation, and date parsing happen here — not in app.py.
- No Streamlit imports; these functions are pure Python and fully testable.
"""

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.filters import col_has_data as _col_ok


def _empty_fig(message: str) -> go.Figure:
    """Return a blank figure with a centered annotation — used instead of None in some contexts."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color="gray"),
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
    )
    return fig


# ── Chart 1: Sales by Category ────────────────────────────────────────────────

def chart_sales_by_category(
    df: pd.DataFrame,
    category_col: Optional[str],
    sales_col: Optional[str],
    top_n: int = 15,
) -> Optional[go.Figure]:
    """
    Horizontal bar chart of total sales per category, sorted descending.

    Returns None if category or sales column is missing / empty.
    top_n caps the number of bars to prevent a crowded chart.
    """
    if not _col_ok(df, category_col) or not _col_ok(df, sales_col):
        return None

    agg = (
        df.groupby(category_col, as_index=False)[sales_col]
        .sum()
        .sort_values(sales_col, ascending=True)  # ascending=True → largest bar at top
        .tail(top_n)
    )

    if agg.empty:
        return None

    fig = px.bar(
        agg,
        x=sales_col,
        y=category_col,
        orientation="h",
        title="Sales by Category",
        labels={sales_col: "Total Sales ($)", category_col: "Category"},
        color=sales_col,
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        coloraxis_showscale=False,
        yaxis_title=None,
        xaxis_title="Total Sales ($)",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ── Chart 2: Profit by Region ─────────────────────────────────────────────────

def chart_profit_by_region(
    df: pd.DataFrame,
    region_col: Optional[str],
    profit_col: Optional[str],
) -> Optional[go.Figure]:
    """
    Bar chart of total profit per region, sorted descending.
    Loss regions shown in red to make them visually distinct.

    Returns None if region or profit column is missing / empty.
    """
    if not _col_ok(df, region_col) or not _col_ok(df, profit_col):
        return None

    agg = (
        df.groupby(region_col, as_index=False)[profit_col]
        .sum()
        .sort_values(profit_col, ascending=False)
    )

    if agg.empty:
        return None

    colors = ["#d62728" if v < 0 else "#2ca02c" for v in agg[profit_col]]

    fig = go.Figure(
        go.Bar(
            x=agg[region_col],
            y=agg[profit_col],
            marker_color=colors,
            text=[f"${v:,.0f}" for v in agg[profit_col]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Profit by Region",
        xaxis_title="Region",
        yaxis_title="Total Profit ($)",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    return fig


# ── Chart 3: Monthly Sales Trend ──────────────────────────────────────────────

def chart_monthly_sales_trend(
    df: pd.DataFrame,
    date_col: Optional[str],
    sales_col: Optional[str],
) -> Optional[go.Figure]:
    """
    Line chart of monthly total sales over time.

    - Parses dates with errors='coerce' and drops unparseable rows before plotting.
    - Groups by YYYY-MM period and sorts chronologically.

    Returns None if date or sales column is missing / has no valid data after parsing.
    """
    if not _col_ok(df, date_col) or not _col_ok(df, sales_col):
        return None

    work = df[[date_col, sales_col]].copy()
    work["_parsed_date"] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=["_parsed_date", sales_col])

    if work.empty:
        return None

    work["_month"] = work["_parsed_date"].dt.to_period("M")
    agg = (
        work.groupby("_month", as_index=False)[sales_col]
        .sum()
        .sort_values("_month")
    )
    agg["_month_str"] = agg["_month"].astype(str)

    fig = px.line(
        agg,
        x="_month_str",
        y=sales_col,
        title="Monthly Sales Trend",
        labels={"_month_str": "Month", sales_col: "Total Sales ($)"},
        markers=True,
    )
    fig.update_traces(line_color="#1f77b4", marker_size=6)
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Total Sales ($)",
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_tickangle=-45,
    )
    return fig


# ── Chart 4: Profit vs Sales (Scatter) ───────────────────────────────────────

def chart_profit_vs_sales(
    df: pd.DataFrame,
    sales_col: Optional[str],
    profit_col: Optional[str],
    category_col: Optional[str] = None,
) -> Optional[go.Figure]:
    """
    Scatter plot of Profit vs Sales, optionally coloured by category.

    Useful for spotting high-revenue but low/negative-profit outliers.
    Returns None if sales or profit column is missing / empty.
    """
    if not _col_ok(df, sales_col) or not _col_ok(df, profit_col):
        return None

    plot_df = df[[sales_col, profit_col]].copy()
    color_param: Optional[str] = None

    if _col_ok(df, category_col):
        plot_df[category_col] = df[category_col]
        color_param = category_col

    plot_df = plot_df.dropna(subset=[sales_col, profit_col])
    if plot_df.empty:
        return None

    fig = px.scatter(
        plot_df,
        x=sales_col,
        y=profit_col,
        color=color_param,
        title="Profit vs Sales",
        labels={sales_col: "Sales ($)", profit_col: "Profit ($)"},
        opacity=0.7,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="red", annotation_text="Break-even")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ── Chart 5: Top N Categories by Sales ───────────────────────────────────────

def chart_top_categories(
    df: pd.DataFrame,
    category_col: Optional[str],
    sales_col: Optional[str],
    top_n: int = 10,
) -> Optional[go.Figure]:
    """
    Horizontal bar chart showing the top N categories ranked by total sales.

    Returns None if category or sales column is missing / empty.
    Capped at top_n (default 10) to keep the chart readable.
    """
    if not _col_ok(df, category_col) or not _col_ok(df, sales_col):
        return None

    agg = (
        df.groupby(category_col, as_index=False)[sales_col]
        .sum()
        .nlargest(top_n, sales_col)
        .sort_values(sales_col, ascending=True)
    )

    if agg.empty:
        return None

    fig = px.bar(
        agg,
        x=sales_col,
        y=category_col,
        orientation="h",
        title=f"Top {top_n} Categories by Sales",
        labels={sales_col: "Total Sales ($)", category_col: "Category"},
        color=sales_col,
        color_continuous_scale="Teal",
    )
    fig.update_layout(
        coloraxis_showscale=False,
        yaxis_title=None,
        xaxis_title="Total Sales ($)",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig
