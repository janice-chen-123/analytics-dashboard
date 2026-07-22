# AI Data Analyst Dashboard

An interactive data analytics platform built with Python and Streamlit. Upload any CSV file to get automated data cleaning, KPI tracking, interactive visualizations, and AI-generated business insights — without sending your raw data to any external service.

> **Live Demo:** *(add Streamlit Cloud link after deployment)*
>
> **Screenshot:** *(add after deployment)*

---

## Core Features

- **CSV Upload & Validation** — file type, size, encoding, and empty-file checks
- **Automated Data Cleaning** — column name normalization, duplicate removal, whitespace stripping
- **Data Quality Report** — per-column missing values, unique counts, and data types
- **Flexible Column Mapping** — map your column names to roles (Sales, Profit, Date, etc.) via the sidebar
- **KPI Dashboard** — Total Sales, Total Profit, Profit Margin, Avg Order Value, Unique Orders, Total Quantity
- **Interactive Charts** (Plotly) — Sales by Category, Profit by Region, Monthly Trend, Profit vs Sales scatter, Top 10 Categories
- **Sidebar Filters** — Region, Category, and Date range; all KPIs and charts update instantly
- **SQLite Persistence** — cleaned data is automatically saved to a local database
- **AI Business Insights** — one-click GPT analysis of aggregated statistics (never raw rows)
- **Downloads** — cleaned CSV and AI report as plain text

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| UI framework | Streamlit |
| Data processing | pandas |
| Visualization | Plotly |
| Database | SQLite (built-in `sqlite3`) |
| AI integration | OpenAI Python SDK |
| Environment config | python-dotenv |
| Testing | pytest |
| Deployment | Streamlit Community Cloud |

---

## Project Structure

```
ai-data-analyst-dashboard/
│
├── app.py                    # Streamlit entry point — layout, session state, UI
├── requirements.txt
├── .gitignore
├── .env.example
│
├── data/
│   └── sample_sales.csv      # Demo dataset (83 rows, multiple categories & regions)
│
├── database/
│   └── .gitkeep              # analytics.db is created at runtime, not committed
│
├── scripts/
│   └── generate_sample_data.py   # Reproducible sample data generator (seed=42)
│
├── src/
│   ├── config.py             # Constants, env var loading, get_api_key()
│   ├── data_loader.py        # CSV validation and loading
│   ├── data_cleaning.py      # Conservative cleaning (no rows dropped)
│   ├── data_quality.py       # Per-column quality report
│   ├── filters.py            # Column helpers, col_has_data(), apply_filters()
│   ├── analytics.py          # KPI calculation (KPIResult dataclass)
│   ├── visualizations.py     # Five Plotly chart builders
│   ├── database.py           # SQLite save_dataframe() and run_query()
│   └── ai_summary.py         # Context builder, prompt, OpenAI API call
│
└── tests/
    ├── test_data_cleaning.py
    ├── test_data_quality.py
    ├── test_analytics.py
    ├── test_database.py
    └── test_ai_summary.py
```

---

## System Architecture

```
CSV Upload
    │
    ▼
Data Validation (data_loader.py)
    │
    ▼
pandas Cleaning (data_cleaning.py)
    │
    ├──► SQLite Storage (database.py)
    │
    ▼
Column Mapping  ←──  Sidebar Config
    │
    ▼
Sidebar Filters (filters.py)
    │
    ▼
KPI Calculation (analytics.py)
    │
    ├──► Plotly Visualizations (visualizations.py)
    │
    ▼
Aggregated JSON Context (ai_summary.py)
    │
    ▼
OpenAI API  →  Structured Business Report
    │
    ▼
Download: Cleaned CSV  |  AI Report (.txt)
```

---

## AI Design Principle

> **Python calculates the numbers. AI explains the numbers.**

The AI (GPT) never receives your raw data. Instead, `build_analysis_context()` computes a compact JSON summary containing:

- Total row count and data quality metrics
- All KPI values
- Top N categories by sales
- Profit by region (with loss regions flagged)
- Monthly sales trend (last 12 months)
- Min / max / mean / median for numeric columns
- Active filter conditions

The system prompt explicitly instructs the model to:
- Only analyse the provided statistics
- Never fabricate numbers, company names, or industry facts
- Never claim causation from correlation
- State uncertainty explicitly
- Always include a Data Limitations section

This approach reduces hallucination, protects data privacy, and keeps AI costs predictable.

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-data-analyst-dashboard.git
cd ai-data-analyst-dashboard
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your OpenAI API key:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_MAX_TOKENS=1500
```

> ⚠️ Never commit `.env` to GitHub. It is already listed in `.gitignore`.

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Sample Data

A demo dataset is included at `data/sample_sales.csv` (83 rows, 9 columns). It includes:

- 5 product categories across 4 regions
- 18 months of order dates (Jan 2023 – Jun 2024)
- Realistic missing values, duplicate rows, and loss-making records

To regenerate it:

```bash
python scripts/generate_sample_data.py
```

---

## Running Tests

```bash
pytest
```

Expected output: **120 tests passed**

| Test file | Tests | What it covers |
|-----------|-------|---------------|
| `test_data_cleaning.py` | 16 | Column name normalization, deduplication, whitespace, immutability |
| `test_data_quality.py` | 18 | Missing value counts, unique counts, empty DataFrame handling |
| `test_analytics.py` | 27 | KPI calculation, division-by-zero, missing columns, empty data |
| `test_database.py` | 26 | SQLite read/write, SELECT-only enforcement, injection prevention |
| `test_ai_summary.py` | 31 | Context building, prompt structure, None columns, download formatting |

---

## Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub (ensure `.env` is **not** committed)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select your repository → set main file to `app.py`
4. Under **Advanced settings → Secrets**, add:

```toml
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_MAX_TOKENS = "1500"
```

5. Click **Deploy**

The app reads secrets via `st.secrets` in production and `.env` locally — handled automatically by `get_api_key()` in `src/config.py`.

---

## Data Privacy

- Uploaded files are processed **in-memory within your session only**
- Only **aggregated statistics** (not individual rows) are sent to the OpenAI API
- The SQLite database (`database/analytics.db`) is created locally and is excluded from version control
- Do not upload datasets containing sensitive personal information (PII, health records, financial credentials)
- When deploying, review your hosting provider's data retention and logging policies

---

## Known Limitations

- SQL interface (`run_query`) is first-version only: keyword blocking is not a complete SQL injection defence and should not be exposed to arbitrary user input in production
- AI analysis quality depends on the OpenAI model and the quality of uploaded data
- No user authentication — unsuitable for multi-tenant or sensitive production environments
- Large files (>50 MB) are rejected; very wide datasets may produce less useful AI summaries
- Date parsing relies on pandas `to_datetime` with `errors="coerce"`; unusual date formats may be silently dropped

---

## Future Improvements

**Version 2**
- Excel / multi-sheet support
- Auto-detection of date and numeric columns
- Anomaly detection and data quality scoring
- PDF report export and chart downloads
- User-defined KPIs

**Version 3**
- Natural language → SQL with user review before execution
- Query history and SQL explanation
- Stricter read-only database sandbox

**Version 4**
- RAG over data dictionary
- Multi-table schema understanding
- AI question-answering with citations
- Role-based access control

---

## Resume Description

> Built an interactive AI-assisted analytics dashboard using Python, Streamlit, pandas, Plotly, and SQLite, enabling users to upload datasets, configure column schemas, apply multi-dimensional filters, and explore KPIs and business trends through five interactive charts.
>
> Designed a controlled LLM analysis workflow that sends only aggregated JSON statistics to the OpenAI API — never raw rows — generating structured five-section business reports while reducing hallucination risk and protecting data privacy.
>
> Developed modular, fully tested Python components (120 pytest cases) covering data cleaning, quality reporting, KPI calculation, SQLite persistence, and AI context generation, and deployed the application on Streamlit Community Cloud.

---

## License

MIT License — see `LICENSE` for details.
