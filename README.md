# AI Data Analyst Dashboard

An interactive data analytics platform built with Python and Streamlit. Upload any CSV file to get automated data cleaning, KPI tracking, interactive visualizations, and AI-generated business insights — without sending your raw data to any external service.

**Live Demo:** *(coming soon)*

---

## Features

- **CSV Upload & Validation** — file type, size, encoding, and empty-file checks
- **Automated Data Cleaning** — column name normalization, duplicate removal, whitespace stripping
- **Data Quality Report** — per-column missing values, unique counts, and data types
- **Flexible Column Mapping** — map your column names to roles (Sales, Profit, Date, etc.) via the sidebar
- **KPI Dashboard** — Total Sales, Total Profit, Profit Margin, Avg Order Value, Unique Orders, Total Quantity
- **Interactive Charts** — Sales by Category, Profit by Region, Monthly Trend, Profit vs Sales scatter, Top 10 Categories
- **Sidebar Filters** — Region, Category, and Date range; all KPIs and charts update instantly
- **SQLite Persistence** — cleaned data is automatically saved to a local database
- **AI Business Insights** — one-click GPT analysis of aggregated statistics (never raw rows)
- **Downloads** — cleaned CSV and AI report as plain text

---

## Tech Stack

Python · Streamlit · pandas · Plotly · SQLite · OpenAI API · pytest

---

## Project Structure

```
ai-data-analyst-dashboard/
├── app.py                        # Streamlit entry point
├── src/
│   ├── config.py                 # Constants and API key management
│   ├── data_loader.py            # CSV validation and loading
│   ├── data_cleaning.py          # Conservative cleaning pipeline
│   ├── data_quality.py           # Per-column quality report
│   ├── filters.py                # Column helpers and filter logic
│   ├── analytics.py              # KPI calculation
│   ├── visualizations.py         # Plotly chart builders
│   ├── database.py               # SQLite persistence layer
│   └── ai_summary.py             # Context builder and OpenAI integration
├── tests/                        # 120 pytest cases
├── data/sample_sales.csv         # Demo dataset
└── scripts/generate_sample_data.py
```

---

## AI Design Principle

> **Python calculates the numbers. AI explains the numbers.**

The AI never receives raw data. `build_analysis_context()` computes a compact JSON summary — KPIs, top categories, regional profit, monthly trend, and descriptive stats — which is sent to the model instead. The system prompt explicitly prohibits fabricating numbers, inventing causal explanations, or making claims beyond the provided data.

---

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
streamlit run app.py
```

```bash
pytest   # 120 tests
```

---

## Data Privacy

Only aggregated statistics are sent to the OpenAI API — never individual rows. Uploaded data is processed in-memory within your session. The local SQLite database is excluded from version control. Do not upload datasets containing sensitive personal information.
