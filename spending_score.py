"""
spending_score.py
Single source of truth for turning a customer's purchase behaviour into
the 1-100 Spending Score used by K-Means.

Both train_model.py (synthetic training customers) and server.py (live
customers) call calculate_spending_score(), so training and prediction
always use exactly the same normalization ranges and weights.

Units:
  total_spent      -- total amount spent, in ₹ (rupees)
  purchase_count   -- number of purchases made in the period
  purchase_period  -- length of the period, in months
"""

import math

# Monthly spending (₹/month) is normalized on a LOG scale between these
# bounds, because spending amounts vary over orders of magnitude.
MONTHLY_SPEND_MIN = 500.0        # at/below -> intensity score 1
MONTHLY_SPEND_MAX = 100_000.0    # at/above -> intensity score 100

# Monthly purchase frequency (purchases/month) is normalized LINEARLY.
MONTHLY_FREQ_MIN = 0.0           # at/below -> frequency score 1
MONTHLY_FREQ_MAX = 10.0          # at/above -> frequency score 100

INTENSITY_WEIGHT = 0.7
FREQUENCY_WEIGHT = 0.3


def _clamp(value: float, lo: float = 1.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def spending_intensity_score(monthly_spending: float) -> float:
    """₹/month -> 1-100 (log-scaled between MONTHLY_SPEND_MIN and MAX)."""
    x = max(monthly_spending, MONTHLY_SPEND_MIN)
    frac = (math.log(x) - math.log(MONTHLY_SPEND_MIN)) / (
        math.log(MONTHLY_SPEND_MAX) - math.log(MONTHLY_SPEND_MIN))
    return _clamp(1 + 99 * frac)


def purchase_frequency_score(monthly_frequency: float) -> float:
    """purchases/month -> 1-100 (linear between MONTHLY_FREQ_MIN and MAX)."""
    frac = (monthly_frequency - MONTHLY_FREQ_MIN) / (MONTHLY_FREQ_MAX - MONTHLY_FREQ_MIN)
    return _clamp(1 + 99 * frac)


def calculate_spending_score(total_spent: float, purchase_count: float,
                             purchase_period: float) -> float:
    """Final Spending Score (1-100) = 70% intensity + 30% frequency."""
    if purchase_period <= 0:
        raise ValueError("purchase_period must be greater than 0")
    monthly_spending = total_spent / purchase_period
    monthly_frequency = purchase_count / purchase_period
    score = (INTENSITY_WEIGHT * spending_intensity_score(monthly_spending) +
             FREQUENCY_WEIGHT * purchase_frequency_score(monthly_frequency))
    return round(_clamp(score), 1)
