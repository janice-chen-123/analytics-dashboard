"""
app.py — Main Streamlit entry point for AI Data Analyst Dashboard.

Responsibilities:
- Page layout and configuration
- Calling modules from src/
- Managing Streamlit session state
- Displaying results to the user

Business logic lives in src/, not here.
"""

import pandas as pd
import streamlit as st

from src.config import APP_TITLE, APP_SUBTITLE, APP_VERSION, MAX_FILE_SIZE_MB
from src.data_loader import load_csv, get_file_summary
from src.data_cleaning import clean_dataframe, CleaningOptions
from src.data_quality import generate_quality_report, get_quality_summary
from src.filters import (
    get_numeric_columns,
    get_all_columns,
    validate_col_config,
    apply_filters,
)
from src.analytics import calculate_kpis
from src.database import save_dataframe as db_save, DEFAULT_DB_PATH, DEFAULT_TABLE
from src.ai_summary import (
    build_analysis_context,
    generate_ai_summary,
    format_report_for_download,
)
from src.config import get_api_key, DEFAULT_MODEL, DEFAULT_MAX_TOKENS
from src.visualizations import (
    chart_sales_by_category,
    chart_profit_by_region,
    chart_monthly_sales_trend,
    chart_profit_vs_sales,
    chart_top_categories,
)

# ── Page configuration (must be the first Streamlit call) ────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Cached computation wrappers ───────────────────────────────────────────────
# col_config is a dict (not hashable), so we pass it as a sorted tuple of items.

@st.cache_data
def _cached_kpis(df: pd.DataFrame, cfg_items: tuple):
    return calculate_kpis(df, dict(cfg_items))

@st.cache_data
def _cached_chart_sales_by_category(df: pd.DataFrame, category_col, sales_col):
    return chart_sales_by_category(df, category_col, sales_col)

@st.cache_data
def _cached_chart_profit_by_region(df: pd.DataFrame, region_col, profit_col):
    return chart_profit_by_region(df, region_col, profit_col)

@st.cache_data
def _cached_chart_monthly_sales_trend(df: pd.DataFrame, date_col, sales_col):
    return chart_monthly_sales_trend(df, date_col, sales_col)

@st.cache_data
def _cached_chart_profit_vs_sales(df: pd.DataFrame, sales_col, profit_col, category_col):
    return chart_profit_vs_sales(df, sales_col, profit_col, category_col)

@st.cache_data
def _cached_chart_top_categories(df: pd.DataFrame, category_col, sales_col):
    return chart_top_categories(df, category_col, sales_col)


# ── Session state ─────────────────────────────────────────────────────────────

def init_session_state() -> None:
    """Initialize all session state keys on first load."""
    defaults: dict = {
        "df_raw": None,
        "df_clean": None,
        "df_filtered": None,
        "file_name": None,
        "col_config": {},
        "filters": {},
        "ai_report": None,
        "ai_report_context": None,
        "clean_changes": [],
        "db_saved": False,
        "db_row_count": 0,
        "clean_result": None,
        "last_cleaning_options": None,
        # Widget state for column selector dropdowns (managed via key=)
        "sel_date": "— None —",
        "sel_category": "— None —",
        "sel_region": "— None —",
        "sel_sales": "— None —",
        "sel_profit": "— None —",
        "sel_quantity": "— None —",
        "sel_order_id": "— None —",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _get_cleaning_options() -> CleaningOptions:
    """Build CleaningOptions from current sidebar widget values."""
    fill_numeric_map = {
        "Keep as-is": "none",
        "Fill with Mean": "mean",
        "Fill with Median": "median",
        "Fill with Zero": "zero",
    }
    fill_text_map = {
        "Keep as-is": "none",
        "Fill with Most Common": "mode",
        'Fill with "Unknown"': "unknown",
    }
    return CleaningOptions(
        fill_numeric=fill_numeric_map.get(
            st.session_state.get("clean_fill_numeric", "Keep as-is"), "none"
        ),
        fill_text=fill_text_map.get(
            st.session_state.get("clean_fill_text", "Keep as-is"), "none"
        ),
        remove_outliers=st.session_state.get("clean_remove_outliers", False),
        outlier_threshold=st.session_state.get("clean_outlier_mult", 1.5),
    )


def ensure_cleaned_df() -> None:
    """Run cleaning when a new file is loaded or cleaning options change."""
    if st.session_state["df_raw"] is None:
        return

    options = _get_cleaning_options()
    last_options = st.session_state.get("last_cleaning_options")
    needs_clean = (
        st.session_state["df_clean"] is None
        or last_options != options.__dict__
    )

    if not needs_clean:
        return

    result = clean_dataframe(st.session_state["df_raw"], options)
    st.session_state["df_clean"] = result.cleaned_df
    st.session_state["clean_changes"] = result.changes
    st.session_state["clean_result"] = result
    st.session_state["last_cleaning_options"] = options.__dict__

    # Reset dependent state when data changes
    st.session_state["df_filtered"] = None
    st.session_state["ai_report"] = None
    st.session_state["ai_report_context"] = None

    # Persist cleaned data to SQLite
    try:
        db_save(result.cleaned_df)
        st.session_state["db_saved"] = True
        st.session_state["db_row_count"] = len(result.cleaned_df)
    except Exception:
        st.session_state["db_saved"] = False
        st.session_state["db_row_count"] = 0


# ── Header ────────────────────────────────────────────────────────────────────

def render_header() -> None:
    st.title(f"📊 {APP_TITLE}")
    st.caption(f"v{APP_VERSION}")
    st.markdown(APP_SUBTITLE)

    with st.expander("ℹ️ How to use this dashboard", expanded=False):
        st.markdown(
            """
            1. **Upload** a CSV file using the sidebar.
            2. **Configure** which columns represent dates, categories, sales, etc.
            3. **Apply filters** to focus on a subset of your data.
            4. **Explore** KPIs, charts, and data quality metrics.
            5. **Generate AI Insights** to get a business summary.
            6. **Download** the cleaned data or the AI report.

            > **Data Privacy Notice**
            > - Your data is processed locally within this app session.
            > - Do **not** upload highly sensitive or personally identifiable data.
            > - Only **aggregated statistics** (not raw rows) are sent to the AI service.
            > - This deployment may have its own logging; review your hosting provider's policy.
            """
        )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar_upload() -> None:
    """File upload widget — resets all state when a new file is loaded."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.subheader("📁 Upload Data")

        uploaded_file = st.file_uploader(
            label="Choose a CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help=f"Maximum file size: {MAX_FILE_SIZE_MB} MB",
        )

        if uploaded_file is not None:
            if st.session_state["file_name"] != uploaded_file.name:
                with st.spinner("Reading file…"):
                    result = load_csv(uploaded_file)

                if result.success:
                    st.session_state.update(
                        {
                            "df_raw": result.df,
                            "df_clean": None,
                            "df_filtered": None,
                            "file_name": uploaded_file.name,
                            "col_config": {},
                            "filters": {},
                            "ai_report": None,
                            "ai_report_context": None,
                            "clean_changes": [],
                            # Reset column selector widgets for the new file
                            "sel_date": "— None —",
                            "sel_category": "— None —",
                            "sel_region": "— None —",
                            "sel_sales": "— None —",
                            "sel_profit": "— None —",
                            "sel_quantity": "— None —",
                            "sel_order_id": "— None —",
                        }
                    )
                    for warning in result.warnings:
                        st.warning(warning)
                    st.success(f"✅ Loaded: **{uploaded_file.name}**")
                else:
                    st.error(f"❌ {result.error}")
                    st.session_state["df_raw"] = None
                    st.session_state["file_name"] = None

        elif st.session_state["df_raw"] is not None:
            for key in [
                "df_raw", "df_clean", "df_filtered", "file_name",
                "col_config", "filters", "ai_report", "ai_report_context",
            ]:
                st.session_state[key] = None
            st.session_state["clean_changes"] = []
            st.rerun()

        if st.session_state["df_raw"] is None:
            st.info("Upload a CSV file to begin.")
        elif st.session_state.get("db_saved"):
            st.caption(
                f"💾 Saved {st.session_state['db_row_count']:,} rows to SQLite "
                f"(`{DEFAULT_TABLE}`)"
            )
        else:
            st.caption("💾 Database: not saved yet.")


def render_sidebar_cleaning() -> None:
    """Sidebar section for user-configurable data cleaning options."""
    with st.sidebar:
        st.markdown("---")
        with st.expander("🧹 Cleaning Options", expanded=False):
            st.caption(
                "These steps run after the baseline cleaning "
                "(column rename, dedup, whitespace strip)."
            )

            st.selectbox(
                "Numeric missing values",
                options=["Keep as-is", "Fill with Mean", "Fill with Median", "Fill with Zero"],
                key="clean_fill_numeric",
                help="Applied to all numeric columns.",
            )

            st.selectbox(
                "Text missing values",
                options=["Keep as-is", "Fill with Most Common", 'Fill with "Unknown"'],
                key="clean_fill_text",
                help="Applied to all text/categorical columns.",
            )

            st.checkbox(
                "Remove outliers (IQR method)",
                key="clean_remove_outliers",
                help="Removes rows where a numeric value falls outside Q1 − k×IQR or Q3 + k×IQR.",
            )
            if st.session_state.get("clean_remove_outliers"):
                st.slider(
                    "IQR multiplier",
                    min_value=1.0,
                    max_value=3.0,
                    value=1.5,
                    step=0.5,
                    key="clean_outlier_mult",
                    help="1.5 = standard (more rows removed). 3.0 = loose (fewer rows removed).",
                )


def render_sidebar_col_config(df: pd.DataFrame) -> None:
    """
    Column mapping selector shown after a file is loaded.

    Uses key= on every selectbox so Streamlit stores widget state in
    st.session_state["sel_*"]. This prevents user selections from being
    reset on every rerender (the bug that occurs when using index= alone).

    Before rendering, each saved value is validated against the current
    file's columns. If the saved value no longer exists (e.g. a new file
    was uploaded), it is reset to '— None —'.
    """
    with st.sidebar:
        st.markdown("---")
        st.subheader("🗂 Column Configuration")
        st.caption("Map your columns to the roles used for KPIs and charts.")

        numeric_cols = get_numeric_columns(df)
        all_cols = get_all_columns(df)

        none_option = "— None —"
        numeric_options = [none_option] + numeric_cols
        all_options = [none_option] + all_cols

        # Validate saved widget values against the current file's columns.
        # Reset to none_option if the saved column no longer exists.
        for role, options in [
            ("date", all_options), ("category", all_options),
            ("region", all_options), ("order_id", all_options),
            ("sales", numeric_options), ("profit", numeric_options),
            ("quantity", numeric_options),
        ]:
            if st.session_state.get(f"sel_{role}", none_option) not in options:
                st.session_state[f"sel_{role}"] = none_option

        date_sel = st.selectbox(
            "📅 Date column", all_options,
            key="sel_date",
            help="Column containing order or transaction dates.",
        )
        category_sel = st.selectbox(
            "🏷 Category column", all_options,
            key="sel_category",
            help="Column for product category or segment.",
        )
        region_sel = st.selectbox(
            "🌍 Region column", all_options,
            key="sel_region",
            help="Column for geographic region or market.",
        )
        sales_sel = st.selectbox(
            "💰 Sales column", numeric_options,
            key="sel_sales",
            help="Numeric column for revenue / sales amount.",
        )
        profit_sel = st.selectbox(
            "📈 Profit column", numeric_options,
            key="sel_profit",
            help="Numeric column for profit (can be negative).",
        )
        quantity_sel = st.selectbox(
            "📦 Quantity column (optional)", numeric_options,
            key="sel_quantity",
            help="Numeric column for units sold.",
        )
        order_id_sel = st.selectbox(
            "🔢 Order ID column (optional)", all_options,
            key="sel_order_id",
            help="Column uniquely identifying each order.",
        )

        def _resolve(sel: str) -> str | None:
            return None if sel == none_option else sel

        new_config = {
            "date": _resolve(date_sel),
            "category": _resolve(category_sel),
            "region": _resolve(region_sel),
            "sales": _resolve(sales_sel),
            "profit": _resolve(profit_sel),
            "quantity": _resolve(quantity_sel),
            "order_id": _resolve(order_id_sel),
        }

        if new_config != st.session_state.get("col_config"):
            st.session_state["ai_report"] = None
            st.session_state["ai_report_context"] = None

        st.session_state["col_config"] = new_config


def render_sidebar_filters(df: pd.DataFrame) -> None:
    """
    Filter controls — only renders widgets for configured columns.

    Writes the filtered DataFrame to session_state["df_filtered"].
    """
    with st.sidebar:
        st.markdown("---")
        st.subheader("🔎 Filters")

        cfg = st.session_state.get("col_config", {})
        region_col = cfg.get("region")
        category_col = cfg.get("category")
        date_col = cfg.get("date")

        has_any_filter = any([region_col, category_col, date_col])

        if not has_any_filter:
            st.caption("Configure columns above to enable filters.")
            st.session_state["df_filtered"] = df.copy()
            return

        selected_regions: list[str] | None = None
        selected_categories: list[str] | None = None
        date_start = None
        date_end = None

        # ── Region multi-select ───────────────────────────────────────────────
        if region_col and region_col in df.columns:
            region_values = sorted(df[region_col].dropna().unique().tolist())
            selected_regions = st.multiselect(
                "Region",
                options=region_values,
                default=region_values,
                help="Select one or more regions to include.",
            )
            if set(selected_regions) == set(region_values):
                selected_regions = None

        # ── Category multi-select ─────────────────────────────────────────────
        if category_col and category_col in df.columns:
            cat_values = sorted(df[category_col].dropna().unique().tolist())
            selected_categories = st.multiselect(
                "Category",
                options=cat_values,
                default=cat_values,
                help="Select one or more categories to include.",
            )
            if set(selected_categories) == set(cat_values):
                selected_categories = None

        # ── Date range ────────────────────────────────────────────────────────
        if date_col and date_col in df.columns:
            parsed_dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
            if not parsed_dates.empty:
                min_date = parsed_dates.min().date()
                max_date = parsed_dates.max().date()

                col_a, col_b = st.columns(2)
                with col_a:
                    date_start = st.date_input(
                        "From", value=min_date, min_value=min_date, max_value=max_date
                    )
                with col_b:
                    date_end = st.date_input(
                        "To", value=max_date, min_value=min_date, max_value=max_date
                    )

                if date_start == min_date and date_end == max_date:
                    date_start = None
                    date_end = None
            else:
                st.caption(f"Column '{date_col}' has no parseable dates.")

        # ── Reset button ──────────────────────────────────────────────────────
        if st.button("↺ Reset Filters", use_container_width=True):
            st.rerun()

        # ── Apply filters ─────────────────────────────────────────────────────
        filtered = apply_filters(
            df=df,
            col_config=cfg,
            selected_regions=selected_regions,
            selected_categories=selected_categories,
            date_start=date_start,
            date_end=date_end,
        )

        new_filters = {
            "regions": str(selected_regions),
            "categories": str(selected_categories),
            "date_start": str(date_start),
            "date_end": str(date_end),
        }
        if new_filters != st.session_state.get("filters"):
            st.session_state["ai_report"] = None
            st.session_state["ai_report_context"] = None
            st.session_state["filters"] = new_filters

        st.session_state["df_filtered"] = filtered

        n_total = len(df)
        n_filtered = len(filtered)
        if n_filtered < n_total:
            st.caption(f"Showing **{n_filtered:,}** of {n_total:,} rows.")
        else:
            st.caption(f"All {n_total:,} rows shown (no active filters).")


# ── Main sections ─────────────────────────────────────────────────────────────

def render_no_file_state() -> None:
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            ### Get started

            Upload a CSV file using the **sidebar on the left** to begin your analysis.

            The dashboard will automatically:
            - Check data quality
            - Clean column names
            - Calculate KPIs
            - Generate interactive charts
            - Provide AI-powered business insights
            """
        )
        st.markdown("---")
        st.markdown("**Expected column format** — your CSV should have columns similar to:")
        st.code(
            "order_id, order_date, category, region, sales, profit, quantity",
            language="text",
        )


def render_data_preview() -> None:
    """Section 1: Data Preview."""
    df_raw = st.session_state["df_raw"]
    df_clean = st.session_state["df_clean"]
    df_filtered = st.session_state["df_filtered"]
    summary = get_file_summary(df_raw)

    st.markdown("## 📋 Section 1: Data Preview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows (raw)", f"{summary['row_count']:,}")
    c2.metric("Rows (cleaned)", f"{len(df_clean):,}" if df_clean is not None else "—")
    c3.metric("Rows (filtered)", f"{len(df_filtered):,}" if df_filtered is not None else "—")
    c4.metric("Columns", summary["col_count"])

    # ── Cleaning summary ──────────────────────────────────────────────────────
    clean_result = st.session_state.get("clean_result")
    if clean_result is not None:
        with st.expander("🧹 Cleaning Summary", expanded=False):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rows Before", f"{clean_result.rows_before:,}")
            m2.metric("Rows After", f"{clean_result.rows_after:,}",
                      delta=f"{clean_result.rows_after - clean_result.rows_before:+,}")
            m3.metric("Missing Before", f"{clean_result.missing_before:,}")
            m4.metric("Missing After", f"{clean_result.missing_after:,}",
                      delta=f"{clean_result.missing_after - clean_result.missing_before:+,}")

            if not clean_result.missing_by_col.empty:
                st.markdown("**Missing values by column**")
                st.dataframe(
                    clean_result.missing_by_col,
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("**Steps applied**")
            for change in clean_result.changes:
                st.markdown(f"- {change}")

    tab_raw, tab_clean, tab_filtered, tab_cols = st.tabs(
        ["Raw Data", "Cleaned Data", "Filtered Data", "Column Info"]
    )

    with tab_raw:
        st.caption("First 20 rows of the uploaded file (unmodified).")
        st.dataframe(df_raw.head(20), use_container_width=True)

    with tab_clean:
        if df_clean is not None:
            st.caption("First 20 rows after cleaning.")
            st.dataframe(df_clean.head(20), use_container_width=True)
        else:
            st.info("Cleaned data not available yet.")

    with tab_filtered:
        if df_filtered is not None and not df_filtered.empty:
            st.caption("First 20 rows after applying sidebar filters.")
            st.dataframe(df_filtered.head(20), use_container_width=True)
        elif df_filtered is not None and df_filtered.empty:
            st.warning("No rows match the current filter selection.")
        else:
            st.info("Configure columns in the sidebar to enable filters.")

    with tab_cols:
        st.caption("Data types from the raw file.")
        st.dataframe(summary["dtypes_df"], use_container_width=True, hide_index=True)


def render_data_quality() -> None:
    """Section 2: Data Quality."""
    df_raw = st.session_state["df_raw"]
    df_clean = st.session_state["df_clean"]
    changes = st.session_state["clean_changes"]

    st.markdown("## 🔍 Section 2: Data Quality")

    summary = get_quality_summary(df_raw)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", f"{summary['row_count']:,}")
    c2.metric("Duplicate Rows", f"{summary['duplicate_count']:,}")
    c3.metric("Total Missing Values", f"{summary['total_missing']:,}")
    c4.metric("Overall Missing %", f"{summary['total_missing_pct']}%")

    st.markdown("### Per-Column Quality Report")
    report = generate_quality_report(df_raw)

    if not report.empty:
        def highlight_missing(row):
            if row["Missing Count"] > 0:
                return ["background-color: #fff3cd"] * len(row)
            return [""] * len(row)

        st.dataframe(
            report.style.apply(highlight_missing, axis=1),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("🟡 Highlighted rows have at least one missing value.")

    st.markdown("### Cleaning Changes Applied")
    if changes:
        for change in changes:
            if "No" in change or "already" in change or "empty" in change.lower():
                st.success(f"✅ {change}")
            else:
                st.info(f"🔧 {change}")

    if df_clean is not None:
        raw_cols = df_raw.columns.tolist()
        clean_cols = df_clean.columns.tolist()
        renamed = [(o, n) for o, n in zip(raw_cols, clean_cols) if o != n]
        if renamed:
            st.markdown("### Column Name Mapping")
            mapping_df = pd.DataFrame(renamed, columns=["Original Name", "Cleaned Name"])
            st.dataframe(mapping_df, use_container_width=True, hide_index=True)


def render_kpi_section() -> None:
    """Section 3: Key Performance Indicators."""
    cfg = st.session_state.get("col_config", {})
    df_filtered = st.session_state.get("df_filtered")

    st.markdown("## 📊 Section 3: Key Performance Indicators")

    if not any(cfg.values()):
        st.info(
            "Configure columns in the **sidebar** to see KPIs. "
            "At minimum, select a Sales column."
        )
        return

    if df_filtered is None or df_filtered.empty:
        st.warning("No data matches the current filter selection — KPIs cannot be computed.")
        return

    kpi = _cached_kpis(df_filtered, tuple(sorted(cfg.items())))

    # ── Row 1: Financial metrics ──────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    if kpi.total_sales is not None:
        c1.metric("💰 Total Sales", f"${kpi.total_sales:,.2f}")
    else:
        c1.metric("💰 Total Sales", "N/A", help="Configure a Sales column.")

    if kpi.total_profit is not None:
        c2.metric("📈 Total Profit", f"${kpi.total_profit:,.2f}")
    else:
        c2.metric("📈 Total Profit", "N/A", help="Configure a Profit column.")

    if kpi.profit_margin_pct is not None:
        c3.metric("📉 Profit Margin", f"{kpi.profit_margin_pct:.1f}%")
    else:
        c3.metric("📉 Profit Margin", "N/A", help="Requires both Sales and Profit columns.")

    avg_label = "💳 Avg Order Value" if kpi.has_order_id else "💳 Avg Record Value"
    if kpi.avg_order_value is not None:
        c4.metric(avg_label, f"${kpi.avg_order_value:,.2f}")
    else:
        c4.metric(avg_label, "N/A")

    # ── Row 2: Volume metrics ─────────────────────────────────────────────────
    c5, c6, c7 = st.columns(3)

    c5.metric("📋 Total Records", f"{kpi.num_records:,}")

    if kpi.num_orders is not None:
        c6.metric("🛒 Unique Orders", f"{kpi.num_orders:,}")
    else:
        c6.metric("🛒 Unique Orders", "N/A", help="Configure an Order ID column.")

    if kpi.total_quantity is not None:
        c7.metric("📦 Total Quantity", f"{kpi.total_quantity:,.0f}")
    else:
        c7.metric("📦 Total Quantity", "N/A", help="Configure a Quantity column.")

    # ── Note on unconfigured roles ────────────────────────────────────────────
    not_set = [role for role, col in cfg.items() if col is None]
    if not_set:
        st.caption(
            f"Not yet configured: {', '.join(not_set)}. "
            "Set these in the sidebar to unlock all metrics."
        )


def render_visual_analysis() -> None:
    """Section 4: Visual Analysis — five Plotly charts."""
    cfg = st.session_state.get("col_config", {})
    df_filtered = st.session_state.get("df_filtered")

    st.markdown("## 📈 Section 4: Visual Analysis")

    if df_filtered is None or df_filtered.empty:
        st.info("No data to visualise — upload a file and configure columns.")
        return

    sales_col = cfg.get("sales")
    profit_col = cfg.get("profit")
    category_col = cfg.get("category")
    region_col = cfg.get("region")
    date_col = cfg.get("date")

    # ── Row 1: Category bar  |  Region bar ────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        fig1 = _cached_chart_sales_by_category(df_filtered, category_col, sales_col)
        if fig1:
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("Configure **Category** and **Sales** columns to see this chart.")

    with col_b:
        fig2 = _cached_chart_profit_by_region(df_filtered, region_col, profit_col)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Configure **Region** and **Profit** columns to see this chart.")

    # ── Row 2: Monthly trend (full width) ─────────────────────────────────────
    fig3 = _cached_chart_monthly_sales_trend(df_filtered, date_col, sales_col)
    if fig3:
        st.plotly_chart(fig3, use_container_width=True)
    else:
        if date_col and sales_col:
            st.warning(
                f"Could not build a monthly trend — column **{date_col}** "
                "may not contain parseable dates."
            )
        else:
            st.info("Configure **Date** and **Sales** columns to see the monthly trend.")

    # ── Row 3: Scatter  |  Top 10 ─────────────────────────────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        fig4 = _cached_chart_profit_vs_sales(df_filtered, sales_col, profit_col, category_col)
        if fig4:
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Configure **Sales** and **Profit** columns to see this scatter plot.")

    with col_d:
        fig5 = _cached_chart_top_categories(df_filtered, category_col, sales_col)
        if fig5:
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("Configure **Category** and **Sales** columns to see the top-10 chart.")


def render_ai_section() -> None:
    """Section 5: AI-Generated Insights."""
    st.markdown("## 🤖 Section 5: AI-Generated Insights")

    cfg        = st.session_state.get("col_config", {})
    df_filtered = st.session_state.get("df_filtered")
    df_raw     = st.session_state.get("df_raw")

    # ── Privacy notice ────────────────────────────────────────────────────────
    with st.expander("🔒 What data is sent to the AI?", expanded=False):
        st.markdown(
            """
            **Only aggregated statistics are sent — never your raw rows.**

            The AI receives a JSON summary containing:
            - Total row count, KPIs (sales, profit, margin)
            - Top categories by sales
            - Profit by region
            - Monthly sales trend (totals per month)
            - Min / max / mean for numeric columns
            - Active filter conditions

            Your individual records are **never** transmitted.
            """
        )

    # ── API key check ─────────────────────────────────────────────────────────
    api_key = get_api_key()
    if not api_key:
        st.warning(
            "⚠️ No OpenAI API key found. "
            "Add `OPENAI_API_KEY` to your `.env` file (local) "
            "or Streamlit Secrets (cloud deployment)."
        )
        return

    if df_filtered is None or df_filtered.empty:
        st.info("No data to analyse — upload a file and configure columns first.")
        return

    if not any(cfg.values()):
        st.info("Configure columns in the sidebar before generating insights.")
        return

    # ── Stale report warning ──────────────────────────────────────────────────
    existing_report = st.session_state.get("ai_report")
    if existing_report:
        st.info(
            "💡 The report below was generated with a previous filter or column "
            "configuration. Click **Generate AI Insights** to refresh it."
        )

    # ── Generate button ───────────────────────────────────────────────────────
    if st.button("✨ Generate AI Insights", type="primary", use_container_width=False):
        kpi = calculate_kpis(df_filtered, cfg)
        quality = get_quality_summary(df_raw) if df_raw is not None else {}
        active_filters = st.session_state.get("filters", {})

        context = build_analysis_context(
            df=df_filtered,
            col_config=cfg,
            kpi=kpi,
            quality_summary=quality,
            active_filters=active_filters,
        )

        with st.spinner("Analysing data and generating insights…"):
            try:
                report = generate_ai_summary(
                    context=context,
                    api_key=api_key,
                    model=DEFAULT_MODEL,
                    max_tokens=DEFAULT_MAX_TOKENS,
                )
                st.session_state["ai_report"] = report
                st.session_state["ai_report_context"] = context
                st.rerun()
            except RuntimeError as exc:
                st.error(f"❌ {exc}")
            except Exception:
                st.error(
                    "❌ An unexpected error occurred. "
                    "Please try again or check your API key."
                )

    # ── Display report ────────────────────────────────────────────────────────
    report = st.session_state.get("ai_report")
    if report:
        st.markdown("---")
        st.markdown(report.replace("$", r"\$"))
        generated_at = st.session_state.get("ai_report_context", {}).get(
            "generated_at", ""
        )
        if generated_at:
            st.caption(f"Report generated at: {generated_at}")

        st.caption(
            "⚠️ **Data Limitation Reminder:** This report is based solely on the "
            "uploaded dataset. Verify all figures before making business decisions."
        )


def render_download_section() -> None:
    """Section 6: Downloads — cleaned CSV and AI report."""
    st.markdown("## ⬇️ Section 6: Downloads")

    df_filtered = st.session_state.get("df_filtered")
    ai_report   = st.session_state.get("ai_report")
    ai_context  = st.session_state.get("ai_report_context", {})

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Cleaned Data")
        if df_filtered is not None and not df_filtered.empty:
            csv_bytes = df_filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Cleaned CSV",
                data=csv_bytes,
                file_name="cleaned_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption(f"{len(df_filtered):,} rows · filtered view")
        else:
            st.info("Upload and configure data to enable download.")

    with col_b:
        st.markdown("### AI Report")
        if ai_report:
            report_text = format_report_for_download(ai_report, ai_context)
            st.download_button(
                label="📄 Download AI Report",
                data=report_text.encode("utf-8"),
                file_name="ai_analysis_report.txt",
                mime="text/plain",
                use_container_width=True,
            )
            st.caption("Plain-text report including header and data context.")
        else:
            st.info("Generate AI Insights (Section 5) to enable report download.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    init_session_state()
    render_header()
    render_sidebar_upload()

    if st.session_state["df_raw"] is None:
        render_no_file_state()
        return

    render_sidebar_cleaning()
    ensure_cleaned_df()

    df_clean = st.session_state["df_clean"]
    if df_clean is None:
        st.error("Cleaning step failed — please re-upload your file.")
        return

    render_sidebar_col_config(df_clean)
    render_sidebar_filters(df_clean)

    # Main page sections
    render_data_preview()
    render_data_quality()
    render_kpi_section()
    render_visual_analysis()
    render_ai_section()
    render_download_section()


if __name__ == "__main__":
    main()
