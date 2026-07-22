"""
ai_summary.py — Build analysis context and call the OpenAI API.

Design principle: Python calculates the numbers. AI explains the numbers.

- build_analysis_context() computes all aggregates from the filtered DataFrame.
- generate_ai_summary() sends only that JSON context to the LLM — never raw rows.
- No Streamlit imports: fully testable in isolation.
"""

import json
from datetime import datetime
from typing import Optional

import pandas as pd

from src.analytics import KPIResult
from src.filters import col_has_data as _col_ok


# ── Context builder ───────────────────────────────────────────────────────────

def build_analysis_context(
    df: pd.DataFrame,
    col_config: dict,
    kpi: KPIResult,
    quality_summary: dict,
    active_filters: Optional[dict] = None,
    top_n: int = 10,
) -> dict:
    """
    Build a JSON-serialisable dict summarising the filtered dataset.

    Only aggregated statistics are included — no raw rows are passed to the AI.

    Args:
        df:              Filtered DataFrame (the data currently on screen).
        col_config:      User column-role mapping (role → column name or None).
        kpi:             Pre-computed KPIResult for this filtered view.
        quality_summary: Output of get_quality_summary() on the raw DataFrame.
        active_filters:  Dict describing which filters are currently active.
        top_n:           How many top categories to include in the context.

    Returns:
        A dict ready for json.dumps().
    """
    sales_col     = col_config.get("sales")
    profit_col    = col_config.get("profit")
    category_col  = col_config.get("category")
    region_col    = col_config.get("region")
    date_col      = col_config.get("date")
    quantity_col  = col_config.get("quantity")

    ctx: dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "filtered_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values_in_raw_data": quality_summary.get("total_missing", "N/A"),
            "duplicate_rows_in_raw_data": quality_summary.get("duplicate_count", "N/A"),
        },
        "column_mapping": {
            role: col for role, col in col_config.items() if col is not None
        },
        "active_filters": active_filters or {},
        "kpis": _kpi_to_dict(kpi),
    }

    # ── Top categories by sales ───────────────────────────────────────────────
    if _col_ok(df, category_col) and _col_ok(df, sales_col):
        top_cats = (
            df.groupby(category_col)[sales_col]
            .sum()
            .nlargest(top_n)
            .round(2)
            .to_dict()
        )
        ctx["top_categories_by_sales"] = top_cats
    else:
        ctx["top_categories_by_sales"] = None

    # ── Profit by region ──────────────────────────────────────────────────────
    if _col_ok(df, region_col) and _col_ok(df, profit_col):
        profit_by_region = (
            df.groupby(region_col)[profit_col]
            .sum()
            .round(2)
            .to_dict()
        )
        ctx["profit_by_region"] = profit_by_region
        ctx["loss_regions"] = [r for r, p in profit_by_region.items() if p < 0]
    else:
        ctx["profit_by_region"] = None
        ctx["loss_regions"] = []

    # ── Monthly sales trend (last 12 months shown) ────────────────────────────
    if _col_ok(df, date_col) and _col_ok(df, sales_col):
        work = df[[date_col, sales_col]].copy()
        work["_date"] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.dropna(subset=["_date", sales_col])
        if not work.empty:
            work["_month"] = work["_date"].dt.to_period("M")
            monthly = (
                work.groupby("_month")[sales_col]
                .sum()
                .sort_index()
                .tail(12)
            )
            ctx["monthly_sales_trend"] = {
                str(k): round(float(v), 2) for k, v in monthly.items()
            }
        else:
            ctx["monthly_sales_trend"] = None
    else:
        ctx["monthly_sales_trend"] = None

    # ── Numeric column descriptive stats ─────────────────────────────────────
    numeric_stats: dict = {}
    for role, col in [("sales", sales_col), ("profit", profit_col), ("quantity", quantity_col)]:
        if _col_ok(df, col):
            s = df[col].dropna()
            numeric_stats[role] = {
                "min":  round(float(s.min()), 2),
                "max":  round(float(s.max()), 2),
                "mean": round(float(s.mean()), 2),
                "median": round(float(s.median()), 2),
            }
    ctx["numeric_stats"] = numeric_stats if numeric_stats else None

    return ctx


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(context: dict) -> str:
    """
    Construct the system + user prompt sent to the LLM.

    The prompt enforces that the AI:
    - Only references the provided aggregated data.
    - Does not fabricate numbers, company names, or industry facts.
    - Does not claim causality from correlation.
    - States uncertainty explicitly.
    - Includes a Data Limitations section.
    - Gives actionable, data-grounded recommendations.
    """
    context_json = json.dumps(context, indent=2, ensure_ascii=False)

    system_prompt = (
        "You are a senior business analyst assistant. "
        "Your role is to interpret pre-computed data summaries and produce clear, "
        "structured business insights. "
        "You must follow these rules strictly:\n"
        "1. ONLY analyse the data provided in the JSON context below.\n"
        "2. NEVER fabricate numbers, percentages, customer names, company details, "
        "or industry benchmarks not present in the data.\n"
        "3. NEVER invent external reasons for trends (e.g. 'due to the holiday season') "
        "unless the data explicitly supports it.\n"
        "4. Do NOT treat correlation as causation.\n"
        "5. When the data is insufficient to draw a conclusion, say so explicitly.\n"
        "6. Always include a Data Limitations section acknowledging what the data "
        "cannot tell us.\n"
        "7. Recommendations must be directly traceable to specific numbers in the data.\n"
        "8. Do not add a preamble like 'Sure!' or 'Great question!'. Start directly "
        "with the report."
    )

    user_prompt = (
        "Based on the following aggregated dataset summary, write a structured "
        "business analysis report.\n\n"
        f"DATA CONTEXT:\n{context_json}\n\n"
        "Format your response using exactly these five sections:\n\n"
        "## 1. Executive Summary\n"
        "(2-3 sentences summarising overall performance)\n\n"
        "## 2. Key Findings\n"
        "(Bullet points covering the most important patterns in the data)\n\n"
        "## 3. Risks or Anomalies\n"
        "(Loss-making regions, low margins, missing data concerns, unusual patterns)\n\n"
        "## 4. Recommended Actions\n"
        "(Specific, data-grounded suggestions — each tied to a number from the context)\n\n"
        "## 5. Data Limitations\n"
        "(What this dataset cannot tell us, and what additional data would help)\n\n"
        "Use plain business language. Avoid technical jargon. "
        "Dollar amounts should be formatted with $ and commas."
    )

    return system_prompt, user_prompt


# ── API caller ────────────────────────────────────────────────────────────────

def generate_ai_summary(
    context: dict,
    api_key: str,
    model: str,
    max_tokens: int = 1500,
) -> str:
    """
    Call the OpenAI Chat Completions API and return the AI-generated report text.

    Args:
        context:    The analysis context dict from build_analysis_context().
        api_key:    OpenAI API key (never logged or returned in error messages).
        model:      Model name, e.g. 'gpt-4o-mini'.
        max_tokens: Maximum tokens for the completion.

    Returns:
        The AI-generated markdown report as a string.

    Raises:
        RuntimeError: Wraps any OpenAI API error with a user-safe message
                      (no API key or raw exception details exposed).
    """
    try:
        from openai import OpenAI  # local import keeps module testable without openai installed
    except ImportError as exc:
        raise RuntimeError(
            "The 'openai' package is not installed. "
            "Run: pip install openai"
        ) from exc

    system_prompt, user_prompt = build_prompt(context)

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    except Exception as exc:
        # Surface a user-safe error — never expose the raw exception which
        # may contain the API key or internal details.
        raise RuntimeError(
            f"AI analysis failed. Please check your API key and try again. "
            f"(Error type: {type(exc).__name__})"
        ) from exc


# ── Report formatter ──────────────────────────────────────────────────────────

def format_report_for_download(report_text: str, context: dict) -> str:
    """
    Wrap the AI report with a header for the downloadable .txt file.

    Args:
        report_text: The raw markdown text returned by generate_ai_summary().
        context:     The analysis context used to generate this report.

    Returns:
        A plain-text string ready for st.download_button().
    """
    generated_at = context.get("generated_at", "N/A")
    filtered_rows = context.get("dataset", {}).get("filtered_rows", "N/A")
    filters = context.get("active_filters", {})

    header_lines = [
        "=" * 60,
        "AI DATA ANALYST DASHBOARD — BUSINESS INSIGHTS REPORT",
        "=" * 60,
        f"Generated : {generated_at}",
        f"Rows analysed : {filtered_rows:,}" if isinstance(filtered_rows, int) else f"Rows analysed : {filtered_rows}",
    ]
    if filters:
        header_lines.append(f"Active filters : {json.dumps(filters, ensure_ascii=False)}")
    header_lines += [
        "",
        "IMPORTANT: This report was generated by an AI based on pre-computed",
        "aggregated statistics. It does not contain raw personal data.",
        "Verify all figures against the original dataset before acting on them.",
        "=" * 60,
        "",
    ]

    return "\n".join(header_lines) + report_text


def _kpi_to_dict(kpi: KPIResult) -> dict:
    """Convert KPIResult dataclass to a plain dict for JSON serialisation."""
    return {
        "total_sales":       kpi.total_sales,
        "total_profit":      kpi.total_profit,
        "profit_margin_pct": kpi.profit_margin_pct,
        "avg_order_value":   kpi.avg_order_value,
        "num_records":       kpi.num_records,
        "num_orders":        kpi.num_orders,
        "total_quantity":    kpi.total_quantity,
        "has_order_id":      kpi.has_order_id,
    }
