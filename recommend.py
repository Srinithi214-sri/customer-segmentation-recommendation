"""
recommend.py
Product recommendation engine.

Products are NOT hardcoded per persona in Python. They live in
data/products.csv, each tagged with an income_level, spend_level, and
price_range. This module:

  1. Classifies a live customer's actual income/spending into the same
     low/mid/high buckets used during training (persona_config thresholds).
  2. Scores every product by how close it is to the customer's ACTUAL
     income/spending value (not just "same bucket, random pick") -- so
     recommendations are ranked, not shuffled.
  3. Attaches a short human-readable reason for each recommendation.
  4. Splits results into a "top" tier (best matches, diversified across
     categories so it doesn't feel like a random dump of one category)
     and a "more" tier (everything else, still relevant).

Income scale: Annual_Income_k (USD thousands)
  low  =  $1k –  $40k  (≈ ₹1L  – ₹33L)
  mid  = $40k – $120k  (≈ ₹33L – ₹1Cr)
  high = $120k – $240k (≈ ₹1Cr – ₹2Cr)
"""

import pandas as pd

from persona_config import INCOME_LOW_MAX, INCOME_HIGH_MIN, SPEND_LOW_MAX, SPEND_HIGH_MIN

CATALOG_PATH = "data/products.csv"

_catalog = None  # loaded lazily, cached after first read

# Representative numeric "center" for each level, derived from the
# same thresholds used to name personas -- so a product tagged
# income_level="mid" is treated as if it's aimed at a customer roughly
# at the midpoint of the "mid" income band, etc.
INCOME_CENTERS = {
    "low": INCOME_LOW_MAX * 0.6,
    "mid": (INCOME_LOW_MAX + INCOME_HIGH_MIN) / 2,
    "high": INCOME_HIGH_MIN * 1.3,
}
SPEND_CENTERS = {
    "low": SPEND_LOW_MAX * 0.6,
    "mid": (SPEND_LOW_MAX + SPEND_HIGH_MIN) / 2,
    "high": SPEND_HIGH_MIN + (100 - SPEND_HIGH_MIN) * 0.4,
}

# Normalize income/spend differences onto a comparable 0-1-ish scale
# before combining into one distance number. Income now spans 1.2–240,
# spending 1–100, so without normalization income differences would
# completely dominate the score.
INCOME_RANGE = 240.0
SPEND_RANGE = 100.0


def _load_catalog() -> pd.DataFrame:
    global _catalog
    if _catalog is None:
        df = pd.read_csv(CATALOG_PATH)
        df["target_income"] = df["income_level"].map(INCOME_CENTERS)
        df["target_spend"] = df["spend_level"].map(SPEND_CENTERS)
        _catalog = df
    return _catalog


def _level(value: float, low_max: float, high_min: float) -> str:
    if value <= low_max:
        return "low"
    if value >= high_min:
        return "high"
    return "mid"


def classify_customer(income: float, spending_score: float) -> tuple[str, str]:
    """Returns (income_level, spend_level) using the SAME thresholds
    train_model.py used to name personas, so recommendations and
    persona labels stay consistent with each other."""
    income_level = _level(income, INCOME_LOW_MAX, INCOME_HIGH_MIN)
    spend_level = _level(spending_score, SPEND_LOW_MAX, SPEND_HIGH_MIN)
    return income_level, spend_level


def _reason(product_income_level: str, product_spend_level: str,
            customer_income_level: str, customer_spend_level: str) -> str:
    exact = (product_income_level == customer_income_level and
             product_spend_level == customer_spend_level)
    if exact:
        return f"Matches shoppers with {customer_income_level} income and {customer_spend_level} spending, like you."
    return "A popular alternative that fits close to your profile."


def get_recommendations(income: float, spending_score: float,
                         top_n: int = 50, top_picks: int = 20,
                         max_per_category_in_top: int = 3) -> dict:
    """Returns {"top": [...], "more": [...]} -- ranked, price-ranged,
    reasoned product recommendations. 'top' is diversified across
    categories; 'more' fills out the rest of top_n by rank.

    Returns up to top_picks items in 'top' and (top_n - top_picks)
    items in 'more', for a combined 50 recommendations by default.
    """
    catalog = _load_catalog().copy()
    income_level, spend_level = classify_customer(income, spending_score)

    # Distance from the customer's ACTUAL numbers to each product's
    # target profile -- smaller distance = better match. This is what
    # makes ranking real instead of a random shuffle within a bucket.
    catalog["distance"] = (
        ((income - catalog["target_income"]) / INCOME_RANGE) ** 2 +
        ((spending_score - catalog["target_spend"]) / SPEND_RANGE) ** 2
    ) ** 0.5
    catalog = catalog.sort_values("distance").reset_index(drop=True)

    def to_result(row) -> dict:
        return {
            "name": row["name"],
            "category": row["category"],
            "price_range": str(row["price_range"]),
            "reason": _reason(row["income_level"], row["spend_level"], income_level, spend_level),
        }

    top: list[dict] = []
    category_counts: dict[str, int] = {}
    leftover_idx: list[int] = []

    # Build the diversified "top" tier: walk the ranked list, take the
    # best match per category up to a cap, so the top picks don't all
    # come from a single category just because it dominates the catalog.
    for idx, row in catalog.iterrows():
        if len(top) >= top_picks:
            leftover_idx.append(idx)
            continue
        cat = row["category"]
        if category_counts.get(cat, 0) >= max_per_category_in_top:
            leftover_idx.append(idx)
            continue
        top.append(to_result(row))
        category_counts[cat] = category_counts.get(cat, 0) + 1

    # Fill "more" with the next-best remaining items by rank.
    remaining_slots = max(top_n - len(top), 0)
    more = [to_result(catalog.loc[idx]) for idx in leftover_idx[:remaining_slots]]

    return {"top": top, "more": more}
