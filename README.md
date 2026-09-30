# Persona — Customer Analysis System

A Flask backend + a custom animated dark-themed frontend. Users register/login,
enter a customer's Age / Gender / Income / Spending Score, and the app predicts
their segment using a trained K-Means model, then shows product recommendations
and an animated "cluster map" of where that customer landed. Every prediction
is saved to a per-user history.

## Project structure

```
customer_analysis_system/
├── server.py              # Flask app: serves the frontend + REST API
├── db.py                  # SQLite + bcrypt auth, prediction history
├── train_model.py         # Trains K-Means (10 clusters), generates persona_config.py
├── recommend.py           # Product recommendation engine (reads data/products.csv)
├── persona_config.py      # AUTO-GENERATED: cluster -> persona/color/centroid + thresholds
├── requirements.txt
├── static/
│   └── index.html         # The frontend (single file: HTML+CSS+JS)
├── model/
│   ├── scaler.pkl
│   └── kmeans_model.pkl
├── data/
│   ├── mall_customers.csv # Synthetic training data (see note below)
│   └── products.csv       # Product catalog (46 products, tagged by income/spend level)
└── customer_system.db     # Created automatically on first run (SQLite)
```

## How to run it

```bash
pip install -r requirements.txt
python server.py
```

Open **http://localhost:5000** in your browser.

1. Click **Log in** → switch to the **Register** tab, create an account.
2. Log in.
3. Go to **Predict**, set Age / Gender / Income / Spending Score, click **Predict my cluster**.
4. Watch the processing animation, then see your persona, recommended products,
   and where you landed on the cluster map.
5. Check **History** to see every past prediction for your account.

Login state is stored server-side in a Flask session cookie, so refreshing the
page keeps you logged in (no `localStorage` used, matching browser-storage best
practice).

## How the frontend talks to the backend

The frontend (`static/index.html`) is a single-page app. Instead of the fake
in-browser clustering logic a plain demo would use, its JavaScript calls real
API routes:

| Frontend action | API call | What actually happens |
|---|---|---|
| Register | `POST /api/register` | bcrypt-hashes the password, inserts into SQLite |
| Login | `POST /api/login` | Verifies bcrypt hash, starts a session |
| Predict | `POST /api/predict` | Scales input → real `KMeans.predict()` → persona lookup → saved to history |
| History | `GET /api/history` | Reads this user's past predictions from SQLite |
| Landing page legend / cluster map | `GET /api/personas` | Real cluster centroids + colors, not hardcoded |

## About the training data

Synthetic data (`data/mall_customers.csv`, 360 rows) shaped like the classic
"Mall Customer Segmentation" dataset, generated because this environment had
no internet access to download the real one. To use the real dataset instead:
download `Mall_Customers.csv` from Kaggle, place it in `data/`, and point
`train_model.py` at it — the rest of the pipeline is unchanged.

## How cluster → persona mapping works

`train_model.py` trains **10 clusters**. Each cluster's centroid income and
spending score are classified into "low / mid / high" using thresholds
computed from the full training set (33rd/67th percentiles) — the same
thresholds are reused by `recommend.py`, so persona labels and product
recommendations always agree with each other. Income level x spend level
gives 9 possible persona names (Budget Customer, Value Seeker, Young Trend
Shopper, Cautious Saver, Average Customer, Active Spender, Frugal
High-Earner, Steady Professional, Luxury Buyer); with 10 clusters, one name
repeats (shown as e.g. "Luxury Buyer #2").

## How product recommendations work (not hardcoded)

Products are **not** a fixed Python list per persona. They live in
`data/products.csv` — 46 products, each independently tagged with an
`income_level` and `spend_level`. At prediction time, `recommend.py`:

1. Classifies the customer's actual income/spending into low/mid/high
   (same thresholds as above).
2. Filters the catalog for products tagged with that exact combination.
3. Randomly samples up to 6 of them (so recommendations vary between
   predictions, even for the same customer profile).
4. If the exact bucket is short on products, broadens the search
   (same income level → same spend level → whole catalog) so a
   result is always returned.

To add or change products, just edit `data/products.csv` — no code changes
needed. Columns: `product_id, name, category, income_level, spend_level`
(`income_level`/`spend_level` must each be `low`, `mid`, or `high`).

## Retraining the model

```bash
python3 train_model.py
```

Regenerates `model/scaler.pkl`, `model/kmeans_model.pkl`, and
`persona_config.py`. Existing users/history in `customer_system.db` are
untouched.

## Notes on this rebuild

This project was originally built with Streamlit, then rebuilt on Flask to
support a fully custom animated frontend (canvas particle background, animated
cluster map, custom sliders) that Streamlit's component model can't produce.
All backend logic (auth, model, database) carried over unchanged — only the
UI layer and how it's served changed.
