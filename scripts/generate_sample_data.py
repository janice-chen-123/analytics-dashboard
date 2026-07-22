"""
generate_sample_data.py — Create data/sample_sales.csv for testing and demos.

Run from the project root:
    python scripts/generate_sample_data.py

The random seed is fixed so the output is reproducible across runs.
"""

import random
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

# ── Constants ─────────────────────────────────────────────────────────────────
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "sample_sales.csv"

CATEGORIES = ["Technology", "Office Supplies", "Furniture", "Clothing", "Sports"]
SUB_CATEGORIES = {
    "Technology":      ["Phones", "Laptops", "Accessories", "Tablets"],
    "Office Supplies": ["Paper", "Binders", "Pens", "Labels"],
    "Furniture":       ["Chairs", "Tables", "Bookcases", "Frames"],
    "Clothing":        ["Shirts", "Shoes", "Outerwear", "Bags"],
    "Sports":          ["Equipment", "Apparel", "Footwear", "Accessories"],
}
REGIONS  = ["East", "West", "Central", "South"]
DISCOUNTS = [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.5]  # mostly no discount


def _base_profit_margin(category: str, discount: float) -> float:
    """Return a realistic margin before discount erosion."""
    base = {
        "Technology":      0.18,
        "Office Supplies": 0.22,
        "Furniture":       0.12,
        "Clothing":        0.25,
        "Sports":          0.15,
    }.get(category, 0.15)
    return base - discount * 0.8   # heavy discounts push margin negative


def generate_rows(n: int = 80) -> list[dict]:
    rows = []
    start_date = pd.Timestamp("2023-01-01")
    end_date   = pd.Timestamp("2024-06-30")
    date_range = (end_date - start_date).days

    for i in range(n):
        category    = random.choice(CATEGORIES)
        sub_cat     = random.choice(SUB_CATEGORIES[category])
        region      = random.choice(REGIONS)
        discount    = random.choice(DISCOUNTS)
        sales       = round(float(rng.uniform(20, 2000)), 2)
        margin      = _base_profit_margin(category, discount)
        noise       = float(rng.normal(0, 0.05))
        profit      = round(sales * (margin + noise), 2)
        quantity    = int(rng.integers(1, 20))
        order_date  = start_date + pd.Timedelta(days=int(rng.integers(0, date_range)))
        order_id    = f"ORD-{10000 + i}"

        rows.append({
            "order_id":    order_id,
            "order_date":  order_date.strftime("%Y-%m-%d"),
            "category":    category,
            "sub_category": sub_cat,
            "region":      region,
            "sales":       sales,
            "profit":      profit,
            "quantity":    quantity,
            "discount":    discount,
        })

    return rows


def add_data_quality_issues(rows: list[dict]) -> list[dict]:
    """Inject realistic data quality issues for demo purposes."""
    result = list(rows)

    # 1. Add 3 exact duplicate rows (copies of early records)
    for i in [2, 7, 15]:
        result.append(dict(rows[i]))

    # 2. Introduce missing values in ~8% of cells across key columns
    for row in result:
        if rng.random() < 0.05:
            row["profit"] = None
        if rng.random() < 0.04:
            row["quantity"] = None
        if rng.random() < 0.03:
            row["discount"] = None

    # 3. Force at least 2 clearly loss-making rows (negative profit, high discount)
    result[5]["discount"] = 0.5
    result[5]["profit"]   = round(-abs(result[5]["sales"]) * 0.15, 2)
    result[12]["discount"] = 0.5
    result[12]["profit"]   = round(-abs(result[12]["sales"]) * 0.20, 2)

    return result


def main() -> None:
    rows  = generate_rows(80)
    rows  = add_data_quality_issues(rows)
    df    = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Generated {len(df)} rows → {OUTPUT_PATH}")
    print(f"  Duplicates injected : 3")
    print(f"  Missing profit      : {df['profit'].isna().sum()}")
    print(f"  Missing quantity    : {df['quantity'].isna().sum()}")
    print(f"  Loss-making rows    : {(df['profit'].dropna() < 0).sum()}")
    print(f"  Date range          : {df['order_date'].min()} → {df['order_date'].max()}")
    print(f"  Categories          : {sorted(df['category'].unique().tolist())}")
    print(f"  Regions             : {sorted(df['region'].unique().tolist())}")


if __name__ == "__main__":
    main()
