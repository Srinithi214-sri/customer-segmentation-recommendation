import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

from spending_score import calculate_spending_score

np.random.seed(42)

os.makedirs("model", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ---------------------------------------------------------
# 1. Generate synthetic "Mall Customer" style dataset
#    (mirrors the well-known 5-segment structure of the
#    classic Mall_Customers.csv dataset used in tutorials),
#    with Spending_Score derived from purchase behaviour
#
#    Income bands (Annual_Income_k = USD thousands):
#      low   : 1.2 –  40   ($1.2k–$40k   ≈ ₹1L–₹33L)
#      mid   :  40 – 120   ($40k–$120k   ≈ ₹33L–₹1Cr)
#      high  : 120 – 240   ($120k–$240k  ≈ ₹1Cr–₹2Cr)
# ---------------------------------------------------------

# Purchase-behaviour bands used to generate realistic raw purchase data.
# The Spending_Score is NOT sampled directly -- it is computed from these
# raw values with calculate_spending_score(), the exact same function
# server.py uses for live customers.
#   monthly spend in ₹/month, purchase frequency in purchases/month
SPEND_BANDS = {
    "low":  {"monthly_spend": (600, 3_000),     "monthly_freq": (0.3, 2.5)},
    "mid":  {"monthly_spend": (3_000, 15_000),  "monthly_freq": (2.5, 5.5)},
    "high": {"monthly_spend": (12_000, 75_000), "monthly_freq": (5.0, 9.5)},
}


def make_segment(n, age_range, income_range, spend_band):
    age = np.random.randint(age_range[0], age_range[1], n)
    income = np.random.normal((income_range[0]+income_range[1])/2,
                               (income_range[1]-income_range[0])/8, n)
    income = np.clip(income, 1.0, 250.0)

    band = SPEND_BANDS[spend_band]
    # Monthly spend is sampled in log space (amounts span orders of magnitude).
    log_lo, log_hi = np.log(band["monthly_spend"][0]), np.log(band["monthly_spend"][1])
    monthly_spend = np.exp(np.random.normal((log_lo + log_hi)/2, (log_hi - log_lo)/8, n))
    freq_lo, freq_hi = band["monthly_freq"]
    monthly_freq = np.clip(np.random.normal((freq_lo + freq_hi)/2, (freq_hi - freq_lo)/8, n), 0.05, None)

    period = np.random.randint(1, 25, n)                       # 1-24 months
    purchase_count = np.maximum(1, np.round(monthly_freq * period)).astype(int)
    total_spent = np.round(monthly_spend * period, 2)
    return age, income, total_spent, purchase_count, period

segments = [
    # age_range, income_range($k), spend band
    ((18, 35), (1.2, 40),   "high"),   # low income,  high spend  -> Enthusiastic Spender
    ((25, 45), (120, 240),  "high"),   # high income, high spend  -> Premium High-Spender
    ((30, 60), (120, 240),  "low"),    # high income, low spend   -> Frugal High-Earner
    ((20, 40), (1.2, 40),   "low"),    # low income,  low spend   -> Budget-Conscious Customer
    ((25, 55), (40, 120),   "mid"),    # mid income,  mid spend   -> Balanced Spender
    ((22, 45), (1.2, 40),   "mid"),    # low income,  mid spend   -> Value-Focused Spender
    ((25, 50), (40, 120),   "low"),    # mid income,  low spend   -> Cautious Saver
    ((22, 50), (40, 120),   "high"),   # mid income,  high spend  -> Active Spender
    ((30, 60), (120, 240),  "mid"),    # high income, mid spend   -> Steady High-Income Spender
]

rows = []
n_per_segment = 120          # 9 × 120 = 1080 rows for good coverage
genders = ["Male", "Female"]

for age_r, income_r, spend_band in segments:
    age, income, total_spent, purchase_count, period = make_segment(
        n_per_segment, age_r, income_r, spend_band)
    for a, inc, spent, cnt, months in zip(age, income, total_spent, purchase_count, period):
        rows.append({
            "Gender": np.random.choice(genders),
            "Age": int(a),
            "Annual_Income_k": round(float(inc), 1),
            "Total_Amount_Spent": float(spent),
            "Purchase_Count": int(cnt),
            "Purchase_Period_Months": int(months),
            "Spending_Score": calculate_spending_score(float(spent), int(cnt), int(months)),
        })

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
df.insert(0, "CustomerID", range(1, len(df) + 1))
df.to_csv("data/mall_customers.csv", index=False)
print(f"Synthetic dataset created: {df.shape[0]} rows -> data/mall_customers.csv")

# ---------------------------------------------------------
# 2. Preprocess: encode gender, scale features
# ---------------------------------------------------------
df["Gender_enc"] = df["Gender"].map({"Male": 1, "Female": 0})

# NOTE: Clustering uses ONLY Annual_Income_k and Spending_Score.
# This is the classic/standard approach for this dataset, and it matters
# here specifically: our persona names are defined purely by income level
# x spending level (see PERSONA_NAMES below). Including Age or Gender as
# clustering features let them pull a customer into a cluster whose
# *income* didn't match their own -- e.g. an older customer with low
# income could get grouped with a "mid income" cluster just because
# their age was a close match, producing a persona label that
# contradicted the customer's own numbers. Clustering on exactly the
# two features that define the personas keeps cluster assignment and
# persona labeling always consistent.
X = df[["Annual_Income_k", "Spending_Score"]].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------
# 3. Train K-Means (10 clusters)
# ---------------------------------------------------------
N_CLUSTERS = 10
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
kmeans.fit(X_scaled)

df["Cluster"] = kmeans.labels_

# ---------------------------------------------------------
# 4. Save model + scaler
# ---------------------------------------------------------
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(kmeans, "model/kmeans_model.pkl")

print("\nSaved model/scaler.pkl and model/kmeans_model.pkl")

# ---------------------------------------------------------
# 5. Print cluster centroids (unscaled) to help map
#    cluster number -> persona in persona_config.py
# ---------------------------------------------------------
print("\nCluster profiles (mean values per cluster):")
profile = df.groupby("Cluster")[["Age", "Annual_Income_k", "Spending_Score"]].mean().round(1)
profile["Count"] = df.groupby("Cluster").size()
print(profile)

# ---------------------------------------------------------
# 6. Compute income/spending thresholds from the FULL
#    training dataset (not just cluster centroids). These
#    same thresholds are reused by recommend.py to classify
#    a *live* customer's income/spending level when picking
#    products from the catalog -- so persona naming and
#    product recommendation both use one consistent
#    definition of "low/mid/high", not two separate ones.
# ---------------------------------------------------------
INCOME_LOW_MAX = round(float(df["Annual_Income_k"].quantile(0.33)), 1)
INCOME_HIGH_MIN = round(float(df["Annual_Income_k"].quantile(0.67)), 1)
SPEND_LOW_MAX = round(float(df["Spending_Score"].quantile(0.33)), 1)
SPEND_HIGH_MIN = round(float(df["Spending_Score"].quantile(0.67)), 1)

print(f"\nThresholds: income_low_max={INCOME_LOW_MAX}, income_high_min={INCOME_HIGH_MIN}")
print(f"            spend_low_max={SPEND_LOW_MAX}, spend_high_min={SPEND_HIGH_MIN}")


def level(value, low_max, high_min):
    if value <= low_max:
        return "low"
    if value >= high_min:
        return "high"
    return "mid"


# ---------------------------------------------------------
# 7. Auto-generate persona_config.py from the ACTUAL centroids.
#    Persona name is a combination of income level x spending
#    level (9 possible combinations), so 10 clusters map to
#    meaningful, mostly-distinct names instead of repeating a
#    small fixed set.
# ---------------------------------------------------------
PERSONA_NAMES = {
    ("low", "low"): "Budget-Conscious Customer",
    ("low", "mid"): "Value-Focused Spender",
    ("low", "high"): "Enthusiastic Spender",
    ("mid", "low"): "Cautious Saver",
    ("mid", "mid"): "Balanced Spender",
    ("mid", "high"): "Active Spender",
    ("high", "low"): "Frugal High-Earner",
    ("high", "mid"): "Steady High-Income Spender",
    ("high", "high"): "Premium High-Spender",
}

PERSONA_DESCRIPTIONS = {
    ("low", "low"): "Low income and low spending, with a preference for keeping purchases within a budget.",
    ("low", "mid"): "Limited income with moderate spending focused on selected purchases.",
    ("low", "high"): "Modest income, but spends enthusiastically and often.",
    ("mid", "low"): "Comfortable income, but spends carefully.",
    ("mid", "mid"): "Moderate income and moderate spending; a balanced shopper.",
    ("mid", "high"): "Comfortable income and enjoys spending freely.",
    ("high", "low"): "High income but cautious with spending; buys selectively.",
    ("high", "mid"): "High income with steady, considered spending habits.",
    ("high", "high"): "High income and high spending on premium products.",
}


def classify(row):
    income_lvl = level(row["Annual_Income_k"], INCOME_LOW_MAX, INCOME_HIGH_MIN)
    spend_lvl = level(row["Spending_Score"], SPEND_LOW_MAX, SPEND_HIGH_MIN)
    key = (income_lvl, spend_lvl)
    return PERSONA_NAMES[key], PERSONA_DESCRIPTIONS[key], income_lvl, spend_lvl


persona_map = {}
used_names = set()
for cluster_id, row in profile.iterrows():
    persona, description, income_lvl, spend_lvl = classify(row)
    base_persona = persona
    suffix = 2
    while persona in used_names:
        persona = f"{base_persona} #{suffix}"
        suffix += 1
    used_names.add(persona)
    persona_map[int(cluster_id)] = {
        "persona": persona,
        "description": description,
        "income_level": income_lvl,
        "spend_level": spend_lvl,
    }

# ---------------------------------------------------------
# 8. Assign a color + a normalized (x, y) map position to each
#    cluster, based on its REAL income/spending centroid, so
#    the frontend's "cluster map" visualization reflects the
#    actual trained model rather than made-up coordinates.
#    x = income (scaled 10-90), y = 90-10 inverted spending
#    (so high spenders plot near the top).
# ---------------------------------------------------------
COLOR_PALETTE = [
    "#FF6B6B", "#4ECDC4", "#FFD166", "#A78BFA", "#06D6A0",
    "#F473B9", "#4D96FF", "#FFA94D", "#82C91E", "#E599F7",
]

income_min, income_max = profile["Annual_Income_k"].min(), profile["Annual_Income_k"].max()
score_min, score_max = profile["Spending_Score"].min(), profile["Spending_Score"].max()


def scale(v, lo, hi, out_lo=10, out_hi=90):
    if hi == lo:
        return (out_lo + out_hi) / 2
    return out_lo + (out_hi - out_lo) * (v - lo) / (hi - lo)


for i, (cluster_id, row) in enumerate(profile.iterrows()):
    x = round(float(scale(row["Annual_Income_k"], income_min, income_max)), 1)
    y = round(float(scale(row["Spending_Score"], score_min, score_max, out_lo=90, out_hi=10)), 1)
    persona_map[int(cluster_id)]["color"] = COLOR_PALETTE[i % len(COLOR_PALETTE)]
    persona_map[int(cluster_id)]["centroid"] = {"x": x, "y": y}

with open("persona_config.py", "w") as f:
    f.write('"""\n')
    f.write("persona_config.py\n")
    f.write("Auto-generated by train_model.py from the trained K-Means centroids.\n")
    f.write("Maps each cluster ID -> persona name, description, a display color,\n")
    f.write("and a normalized (x, y) position for the cluster map.\n")
    f.write("Product recommendations are NOT stored here -- see recommend.py,\n")
    f.write("which scores products from data/products.csv against the customer's\n")
    f.write("actual income/spending level using the thresholds below.\n")
    f.write('"""\n\n')
    f.write(f"INCOME_LOW_MAX = {INCOME_LOW_MAX!r}\n")
    f.write(f"INCOME_HIGH_MIN = {INCOME_HIGH_MIN!r}\n")
    f.write(f"SPEND_LOW_MAX = {SPEND_LOW_MAX!r}\n")
    f.write(f"SPEND_HIGH_MIN = {SPEND_HIGH_MIN!r}\n\n")
    f.write("PERSONA_MAP = {\n")
    for cid, info in persona_map.items():
        f.write(f"    {cid}: {{\n")
        f.write(f'        "persona": {info["persona"]!r},\n')
        f.write(f'        "description": {info["description"]!r},\n')
        f.write(f'        "income_level": {info["income_level"]!r},\n')
        f.write(f'        "spend_level": {info["spend_level"]!r},\n')
        f.write(f'        "color": {info["color"]!r},\n')
        f.write(f'        "centroid": {info["centroid"]!r},\n')
        f.write("    },\n")
    f.write("}\n")

print("\nGenerated persona_config.py from actual cluster centroids:")
for cid, info in persona_map.items():
    print(f"  Cluster {cid} -> {info['persona']} | {info['income_level']}/{info['spend_level']} | color={info['color']}")
