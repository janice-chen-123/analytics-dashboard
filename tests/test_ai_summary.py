"""Tests for src/ai_summary.py — context builder, prompt builder, and formatter."""

import json

import pandas as pd
import pytest

from src.ai_summary import (
    build_analysis_context,
    build_prompt,
    format_report_for_download,
    _kpi_to_dict,
)
from src.analytics import calculate_kpis


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "order_id":   ["A001", "A002", "A003", "A004", "A005"],
        "order_date": ["2024-01-10", "2024-01-20", "2024-02-05", "2024-02-18", "2024-03-01"],
        "category":   ["Tech", "Office", "Tech", "Office", "Sports"],
        "region":     ["East", "West", "East", "West", "North"],
        "sales":      [500.0, 200.0, 300.0, 150.0, 400.0],
        "profit":     [100.0, -20.0, 60.0, -10.0, 80.0],
        "quantity":   [2, 1, 3, 1, 2],
    })


@pytest.fixture
def full_config() -> dict:
    return {
        "date":     "order_date",
        "category": "category",
        "region":   "region",
        "sales":    "sales",
        "profit":   "profit",
        "quantity": "quantity",
        "order_id": "order_id",
    }


@pytest.fixture
def quality_summary() -> dict:
    return {"total_missing": 2, "duplicate_count": 1, "row_count": 5, "col_count": 7}


@pytest.fixture
def context(sample_df, full_config, quality_summary) -> dict:
    kpi = calculate_kpis(sample_df, full_config)
    return build_analysis_context(sample_df, full_config, kpi, quality_summary)


# ── build_analysis_context ────────────────────────────────────────────────────

class TestBuildAnalysisContext:
    def test_returns_dict(self, context):
        assert isinstance(context, dict)

    def test_required_top_level_keys(self, context):
        for key in ("generated_at", "dataset", "column_mapping", "kpis",
                    "top_categories_by_sales", "profit_by_region", "loss_regions",
                    "monthly_sales_trend", "numeric_stats"):
            assert key in context, f"Missing key: {key}"

    def test_dataset_row_count(self, context, sample_df):
        assert context["dataset"]["filtered_rows"] == len(sample_df)

    def test_kpis_total_sales(self, context):
        assert context["kpis"]["total_sales"] == 1550.0

    def test_kpis_total_profit(self, context):
        assert context["kpis"]["total_profit"] == 210.0

    def test_top_categories_includes_tech(self, context):
        assert "Tech" in context["top_categories_by_sales"]

    def test_top_categories_sorted_descending(self, context):
        values = list(context["top_categories_by_sales"].values())
        assert values == sorted(values, reverse=True)

    def test_profit_by_region_present(self, context):
        assert context["profit_by_region"] is not None
        assert "East" in context["profit_by_region"]
        assert "West" in context["profit_by_region"]

    def test_loss_regions_identified(self, context):
        # West has profit -20 + -10 = -30 → should be in loss_regions
        assert "West" in context["loss_regions"]

    def test_monthly_sales_trend_present(self, context):
        assert context["monthly_sales_trend"] is not None
        assert len(context["monthly_sales_trend"]) > 0

    def test_numeric_stats_present(self, context):
        assert context["numeric_stats"] is not None
        assert "sales" in context["numeric_stats"]
        assert "profit" in context["numeric_stats"]

    def test_numeric_stats_min_max(self, context):
        sales_stats = context["numeric_stats"]["sales"]
        assert sales_stats["min"] == 150.0
        assert sales_stats["max"] == 500.0

    def test_quality_summary_passed_through(self, context, quality_summary):
        assert context["dataset"]["missing_values_in_raw_data"] == quality_summary["total_missing"]
        assert context["dataset"]["duplicate_rows_in_raw_data"] == quality_summary["duplicate_count"]

    def test_context_is_json_serialisable(self, context):
        dumped = json.dumps(context)
        assert isinstance(dumped, str)

    def test_active_filters_empty_by_default(self, context):
        assert context["active_filters"] == {}

    def test_active_filters_passed_through(self, sample_df, full_config, quality_summary):
        kpi = calculate_kpis(sample_df, full_config)
        filters = {"regions": "['East']", "categories": "None"}
        ctx = build_analysis_context(sample_df, full_config, kpi, quality_summary,
                                     active_filters=filters)
        assert ctx["active_filters"] == filters

    # ── Missing/None column handling ──────────────────────────────────────────

    def test_no_category_col_returns_none(self, sample_df, quality_summary):
        cfg = {"date": "order_date", "category": None, "region": "region",
               "sales": "sales", "profit": "profit", "quantity": None, "order_id": None}
        kpi = calculate_kpis(sample_df, cfg)
        ctx = build_analysis_context(sample_df, cfg, kpi, quality_summary)
        assert ctx["top_categories_by_sales"] is None

    def test_no_region_col_returns_none(self, sample_df, quality_summary):
        cfg = {"date": "order_date", "category": "category", "region": None,
               "sales": "sales", "profit": "profit", "quantity": None, "order_id": None}
        kpi = calculate_kpis(sample_df, cfg)
        ctx = build_analysis_context(sample_df, cfg, kpi, quality_summary)
        assert ctx["profit_by_region"] is None
        assert ctx["loss_regions"] == []

    def test_no_date_col_returns_none_trend(self, sample_df, quality_summary):
        cfg = {"date": None, "category": "category", "region": "region",
               "sales": "sales", "profit": "profit", "quantity": None, "order_id": None}
        kpi = calculate_kpis(sample_df, cfg)
        ctx = build_analysis_context(sample_df, cfg, kpi, quality_summary)
        assert ctx["monthly_sales_trend"] is None

    def test_empty_dataframe_does_not_raise(self, full_config, quality_summary):
        kpi = calculate_kpis(pd.DataFrame(), full_config)
        ctx = build_analysis_context(pd.DataFrame(), full_config, kpi, quality_summary)
        assert ctx["dataset"]["filtered_rows"] == 0
        assert ctx["top_categories_by_sales"] is None


# ── build_prompt ──────────────────────────────────────────────────────────────

class TestBuildPrompt:
    def test_returns_two_strings(self, context):
        system_p, user_p = build_prompt(context)
        assert isinstance(system_p, str)
        assert isinstance(user_p, str)

    def test_system_prompt_contains_no_fabricate_rule(self, context):
        system_p, _ = build_prompt(context)
        assert "NEVER fabricate" in system_p

    def test_user_prompt_contains_all_sections(self, context):
        _, user_p = build_prompt(context)
        for section in ("Executive Summary", "Key Findings", "Risks or Anomalies",
                        "Recommended Actions", "Data Limitations"):
            assert section in user_p, f"Missing section: {section}"

    def test_user_prompt_contains_context_json(self, context):
        _, user_p = build_prompt(context)
        assert "DATA CONTEXT" in user_p
        # The context JSON should be embedded
        assert "filtered_rows" in user_p

    def test_system_prompt_forbids_causation_claim(self, context):
        system_p, _ = build_prompt(context)
        assert "causation" in system_p or "correlation" in system_p


# ── format_report_for_download ────────────────────────────────────────────────

class TestFormatReportForDownload:
    def test_output_is_string(self, context):
        result = format_report_for_download("## 1. Executive Summary\nTest.", context)
        assert isinstance(result, str)

    def test_header_present(self, context):
        result = format_report_for_download("Report body.", context)
        assert "AI DATA ANALYST DASHBOARD" in result

    def test_report_body_included(self, context):
        body = "## 1. Executive Summary\nSales look good."
        result = format_report_for_download(body, context)
        assert body in result

    def test_disclaimer_present(self, context):
        result = format_report_for_download("body", context)
        assert "IMPORTANT" in result

    def test_generated_at_present(self, context):
        result = format_report_for_download("body", context)
        assert context["generated_at"] in result

    def test_row_count_present(self, context):
        result = format_report_for_download("body", context)
        assert "5" in result  # 5 rows in sample_df
