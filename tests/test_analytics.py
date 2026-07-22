"""Tests for src/analytics.py."""

import pandas as pd
import pytest

from src.analytics import calculate_kpis, KPIResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Standard four-row DataFrame with known totals."""
    return pd.DataFrame(
        {
            "order_id": ["A001", "A002", "A002", "A003"],
            "sales":    [100.0,  200.0,  200.0,  300.0],
            "profit":   [20.0,   -10.0,  -10.0,  60.0],
            "quantity": [1,       2,      2,      3],
        }
    )
    # total_sales   = 800.0
    # total_profit  = 60.0
    # profit_margin = 60 / 800 * 100 = 7.5%
    # num_records   = 4
    # num_orders    = 3  (A001, A002, A003)
    # avg_order_val = 800 / 3 ≈ 266.67
    # total_qty     = 8


@pytest.fixture
def full_config():
    return {
        "date": None,
        "category": None,
        "region": None,
        "sales": "sales",
        "profit": "profit",
        "quantity": "quantity",
        "order_id": "order_id",
    }


@pytest.fixture
def no_order_config():
    return {
        "date": None,
        "category": None,
        "region": None,
        "sales": "sales",
        "profit": "profit",
        "quantity": "quantity",
        "order_id": None,
    }


# ── Total Sales ───────────────────────────────────────────────────────────────

class TestTotalSales:
    def test_total_sales_correct(self, sample_df, full_config):
        result = calculate_kpis(sample_df, full_config)
        assert result.total_sales == 800.0

    def test_total_sales_none_when_not_configured(self, sample_df, full_config):
        cfg = {**full_config, "sales": None}
        result = calculate_kpis(sample_df, cfg)
        assert result.total_sales is None

    def test_total_sales_none_when_column_missing(self, sample_df, full_config):
        cfg = {**full_config, "sales": "nonexistent_column"}
        result = calculate_kpis(sample_df, cfg)
        assert result.total_sales is None

    def test_total_sales_none_when_all_missing(self, full_config):
        df = pd.DataFrame({"order_id": ["A1"], "sales": [None], "profit": [None], "quantity": [None]})
        result = calculate_kpis(df, full_config)
        assert result.total_sales is None


# ── Total Profit ──────────────────────────────────────────────────────────────

class TestTotalProfit:
    def test_total_profit_correct(self, sample_df, full_config):
        result = calculate_kpis(sample_df, full_config)
        assert result.total_profit == 60.0

    def test_total_profit_can_be_negative(self, full_config):
        df = pd.DataFrame({
            "order_id": ["A1", "A2"],
            "sales":    [100.0, 50.0],
            "profit":   [-30.0, -20.0],
            "quantity": [1, 1],
        })
        result = calculate_kpis(df, full_config)
        assert result.total_profit == -50.0

    def test_total_profit_none_when_not_configured(self, sample_df, full_config):
        cfg = {**full_config, "profit": None}
        result = calculate_kpis(sample_df, cfg)
        assert result.total_profit is None


# ── Profit Margin ─────────────────────────────────────────────────────────────

class TestProfitMargin:
    def test_profit_margin_correct(self, sample_df, full_config):
        result = calculate_kpis(sample_df, full_config)
        assert result.profit_margin_pct == 7.5

    def test_profit_margin_none_when_sales_is_zero(self, full_config):
        df = pd.DataFrame({
            "order_id": ["A1"],
            "sales":    [0.0],
            "profit":   [50.0],
            "quantity": [1],
        })
        result = calculate_kpis(df, full_config)
        assert result.profit_margin_pct is None

    def test_profit_margin_none_when_no_sales_column(self, sample_df, full_config):
        cfg = {**full_config, "sales": None}
        result = calculate_kpis(sample_df, cfg)
        assert result.profit_margin_pct is None

    def test_profit_margin_none_when_no_profit_column(self, sample_df, full_config):
        cfg = {**full_config, "profit": None}
        result = calculate_kpis(sample_df, cfg)
        assert result.profit_margin_pct is None

    def test_profit_margin_negative(self, full_config):
        df = pd.DataFrame({
            "order_id": ["A1"],
            "sales":    [100.0],
            "profit":   [-25.0],
            "quantity": [1],
        })
        result = calculate_kpis(df, full_config)
        assert result.profit_margin_pct == -25.0


# ── Avg Order / Record Value ──────────────────────────────────────────────────

class TestAvgOrderValue:
    def test_avg_order_value_with_order_id(self, sample_df, full_config):
        result = calculate_kpis(sample_df, full_config)
        # 800 / 3 orders ≈ 266.67
        assert result.avg_order_value == round(800.0 / 3, 2)
        assert result.has_order_id is True

    def test_avg_record_value_without_order_id(self, sample_df, no_order_config):
        result = calculate_kpis(sample_df, no_order_config)
        # 800 / 4 records = 200.0
        assert result.avg_order_value == 200.0
        assert result.has_order_id is False

    def test_avg_order_value_none_when_no_sales(self, sample_df, full_config):
        cfg = {**full_config, "sales": None}
        result = calculate_kpis(sample_df, cfg)
        assert result.avg_order_value is None


# ── Record and Order Counts ───────────────────────────────────────────────────

class TestCounts:
    def test_num_records_correct(self, sample_df, full_config):
        result = calculate_kpis(sample_df, full_config)
        assert result.num_records == 4

    def test_num_orders_correct(self, sample_df, full_config):
        result = calculate_kpis(sample_df, full_config)
        assert result.num_orders == 3  # A001, A002, A003

    def test_num_orders_none_without_order_id(self, sample_df, no_order_config):
        result = calculate_kpis(sample_df, no_order_config)
        assert result.num_orders is None

    def test_total_quantity_correct(self, sample_df, full_config):
        result = calculate_kpis(sample_df, full_config)
        assert result.total_quantity == 8.0

    def test_total_quantity_none_when_not_configured(self, sample_df, full_config):
        cfg = {**full_config, "quantity": None}
        result = calculate_kpis(sample_df, cfg)
        assert result.total_quantity is None


# ── Empty DataFrame ───────────────────────────────────────────────────────────

class TestEmptyDataFrame:
    def test_empty_df_num_records_is_zero(self, full_config):
        result = calculate_kpis(pd.DataFrame(), full_config)
        assert result.num_records == 0

    def test_empty_df_sales_is_none(self, full_config):
        result = calculate_kpis(pd.DataFrame(), full_config)
        assert result.total_sales is None

    def test_empty_df_profit_is_none(self, full_config):
        result = calculate_kpis(pd.DataFrame(), full_config)
        assert result.total_profit is None

    def test_empty_df_profit_margin_is_none(self, full_config):
        result = calculate_kpis(pd.DataFrame(), full_config)
        assert result.profit_margin_pct is None

    def test_empty_df_does_not_raise(self, full_config):
        result = calculate_kpis(pd.DataFrame(), full_config)
        assert isinstance(result, KPIResult)


# ── Missing Values in Numeric Columns ────────────────────────────────────────

class TestMissingValues:
    def test_partial_missing_sales_ignored_in_sum(self, full_config):
        df = pd.DataFrame({
            "order_id": ["A1", "A2", "A3"],
            "sales":    [100.0, None, 200.0],
            "profit":   [10.0, None, 20.0],
            "quantity": [1, None, 2],
        })
        result = calculate_kpis(df, full_config)
        assert result.total_sales == 300.0
        assert result.total_profit == 30.0
        assert result.total_quantity == 3.0

    def test_all_missing_sales_returns_none(self, full_config):
        df = pd.DataFrame({
            "order_id": ["A1", "A2"],
            "sales":    [None, None],
            "profit":   [10.0, 20.0],
            "quantity": [1, 2],
        })
        result = calculate_kpis(df, full_config)
        assert result.total_sales is None
        assert result.profit_margin_pct is None
