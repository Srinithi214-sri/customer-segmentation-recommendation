"""
verify_model.py
Post-training accuracy verification for the K-Means customer segmentation model.

Tests a set of hand-crafted (income, spending_score) pairs against the trained
model and asserts that each prediction is consistent with the expected income
and spending bucket — i.e., the cluster's persona_config entry must report
the correct income_level and spend_level.

Run AFTER train_model.py:
    python verify_model.py

Income scale: Annual_Income_k (USD thousands)
  low   :   1.2 –  40  ($k)  ≈  ₹1L – ₹33L
  mid   :  40   – 120  ($k)  ≈  ₹33L – ₹1Cr
  high  : 120   – 240  ($k)  ≈  ₹1Cr – ₹2Cr
"""

import sys
import joblib
import numpy as np

try:
    from persona_config import PERSONA_MAP, INCOME_LOW_MAX, INCOME_HIGH_MIN, SPEND_LOW_MAX, SPEND_HIGH_MIN
except ImportError:
    print("ERROR: persona_config.py not found. Run train_model.py first.")
    sys.exit(1)

# ── Load model ────────────────────────────────────────────────────────────────
try:
    scaler = joblib.load("model/scaler.pkl")
    kmeans = joblib.load("model/kmeans_model.pkl")
except FileNotFoundError:
    print("ERROR: Model files not found. Run train_model.py first.")
    sys.exit(1)


def level(value, low_max, high_min):
    if value <= low_max:
        return "low"
    if value >= high_min:
        return "high"
    return "mid"


def predict(income, spending_score):
    features = np.array([[income, spending_score]])
    scaled = scaler.transform(features)
    cluster = int(kmeans.predict(scaled)[0])
    info = PERSONA_MAP.get(cluster, {})
    return cluster, info.get("income_level", "?"), info.get("spend_level", "?"), info.get("persona", "?")


# ── Test cases ────────────────────────────────────────────────────────────────
# Format: (income_k, spending_score, expected_income_level, expected_spend_level, label)
test_cases = [
    # LOW INCOME ──────────────────────────────────────
    (5.0,   10, "low", "low",  "₹4L income, score=10  → Budget Customer"),
    (15.0,  8,  "low", "low",  "₹12L income, score=8  → Budget Customer"),
    (30.0,  20, "low", "low",  "₹25L income, score=20 → Budget Customer"),
    (10.0,  50, "low", "mid",  "₹8L income,  score=50 → Value Seeker"),
    (20.0,  48, "low", "mid",  "₹17L income, score=48 → Value Seeker"),
    (8.0,   80, "low", "high", "₹7L income,  score=80 → Young Trend Shopper"),
    (25.0,  88, "low", "high", "₹21L income, score=88 → Young Trend Shopper"),
    # MID INCOME ──────────────────────────────────────
    (60.0,  12, "mid", "low",  "₹50L income, score=12 → Cautious Saver"),
    (90.0,  20, "mid", "low",  "₹75L income, score=20 → Cautious Saver"),
    (70.0,  50, "mid", "mid",  "₹58L income, score=50 → Average Customer"),
    (100.0, 55, "mid", "mid",  "₹83L income, score=55 → Average Customer"),
    (55.0,  82, "mid", "high", "₹46L income, score=82 → Active Spender"),
    (105.0, 78, "mid", "high", "₹87L income, score=78 → Active Spender"),
    # HIGH INCOME ─────────────────────────────────────
    (150.0, 15, "high", "low",  "₹1.25Cr income, score=15 → Frugal High-Earner"),
    (220.0, 25, "high", "low",  "₹1.83Cr income, score=25 → Frugal High-Earner"),
    (160.0, 55, "high", "mid",  "₹1.33Cr income, score=55 → Steady Professional"),
    (200.0, 48, "high", "mid",  "₹1.66Cr income, score=48 → Steady Professional"),
    (135.0, 85, "high", "high", "₹1.12Cr income, score=85 → Luxury Buyer"),
    (230.0, 92, "high", "high", "₹1.91Cr income, score=92 → Luxury Buyer"),
    # EDGE / BOUNDARY ─────────────────────────────────
    (1.2,   5,  "low",  "low",  "₹1L   income (minimum), score=5  → Budget Customer"),
    (240.0, 98, "high", "high", "₹2Cr  income (maximum), score=98 → Luxury Buyer"),
]


# ── Run tests ─────────────────────────────────────────────────────────────────
print(f"\nModel thresholds: income_low_max={INCOME_LOW_MAX}, income_high_min={INCOME_HIGH_MIN}")
print(f"                  spend_low_max={SPEND_LOW_MAX},  spend_high_min={SPEND_HIGH_MIN}\n")
print(f"{'Test Case':<52} {'Expected':>14} {'Got':>14}  Status")
print("─" * 100)

passes = 0
failures = 0
fail_list = []

for income, score, exp_inc, exp_spend, label in test_cases:
    cluster, got_inc, got_spend, persona = predict(income, score)
    ok = (got_inc == exp_inc and got_spend == exp_spend)
    status = "✓ PASS" if ok else "✗ FAIL"
    expected_str = f"{exp_inc}/{exp_spend}"
    got_str = f"{got_inc}/{got_spend}"
    print(f"{label:<52} {expected_str:>14} {got_str:>14}  {status}  [{persona}]")
    if ok:
        passes += 1
    else:
        failures += 1
        fail_list.append((label, exp_inc, exp_spend, got_inc, got_spend, persona, cluster))

print("─" * 100)
print(f"\nResults: {passes} PASS, {failures} FAIL out of {len(test_cases)} tests\n")

if failures > 0:
    print("FAILED CASES:")
    for label, exp_inc, exp_spend, got_inc, got_spend, persona, cluster in fail_list:
        print(f"  [{label}]")
        print(f"    Expected income_level={exp_inc}, spend_level={exp_spend}")
        print(f"    Got     income_level={got_inc}, spend_level={got_spend} (cluster={cluster}, persona={persona})")
    print("\nFix: Add more rows to the failing buckets in train_model.py and retrain.")
    sys.exit(1)
else:
    print("All predictions are consistent with their income/spending bucket. ✓")
    sys.exit(0)
