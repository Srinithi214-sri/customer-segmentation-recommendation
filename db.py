"""
db.py
Handles all SQLite database logic:
  - users table (username, bcrypt password hash, role)
  - history table (every prediction ever made)
"""

from collections import Counter

import sqlite3
import bcrypt
import json
from datetime import datetime, timedelta

DB_PATH = "customer_system.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT,
            password_hash TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            income REAL,
            spending_score REAL,
            cluster INTEGER,
            persona TEXT,
            products TEXT,
            date TEXT
        )
    """)

    # Add 'role' column to users if it doesn't exist yet (safe migration).
    # Existing rows will automatically receive the default value 'user'.
    try:
        cur.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
    except Exception:
        pass  # Column already exists — nothing to do.

    conn.commit()
    conn.close()


# ---------------------------------------------------------
# User auth
# ---------------------------------------------------------

def username_exists(username: str) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM users WHERE username = ?", (username,))
    exists = cur.fetchone() is not None
    conn.close()
    return exists


def register_user(username: str, password: str, email: str = "") -> tuple[bool, str]:
    """Register a normal user.  Role is always 'user' — never admin."""
    if not username or not password:
        return False, "Username and password cannot be empty."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."
    if username_exists(username):
        return False, "Username already exists."

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, 'user')",
        (username, email or None, hashed.decode("utf-8")),
    )
    conn.commit()
    conn.close()
    return True, "Registration successful. Please log in."


def register_admin(username: str, password: str) -> tuple[bool, str]:
    """Create an admin account.  Only called by create_admin.py, never via the web API."""
    if not username or not password:
        return False, "Username and password cannot be empty."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."
    if username_exists(username):
        return False, "Username already exists."

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (username, email, password_hash, role) VALUES (?, NULL, ?, 'admin')",
        (username, hashed.decode("utf-8")),
    )
    conn.commit()
    conn.close()
    return True, f"Admin account '{username}' created successfully."


def verify_user(username: str, password: str) -> tuple[bool, str]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()

    if row is None:
        return False, "Username not found."

    stored_hash = row["password_hash"].encode("utf-8")
    if bcrypt.checkpw(password.encode("utf-8"), stored_hash):
        return True, "Login successful."
    return False, "Incorrect password."


def get_user_role(username: str) -> str:
    """Return 'admin' or 'user' for the given username."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT role FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return "user"
    return row["role"] or "user"


# ---------------------------------------------------------
# Prediction history
# ---------------------------------------------------------

def save_prediction(username, age, gender, income, spending_score,
                     cluster, persona, products):
    """products: list of dicts, e.g. {"name":..., "price":..., "reason":..., "tier":...}"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO history
            (username, age, gender, income, spending_score, cluster, persona, products, date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        username, age, gender, income, spending_score,
        cluster, persona, json.dumps(products),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ))
    conn.commit()
    conn.close()


def get_history(username: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT age, gender, income, spending_score, cluster, persona, products, date
        FROM history
        WHERE username = ?
        ORDER BY id DESC
    """, (username,))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_persona_counts(username: str):
    """Returns {persona: count} for this user's prediction history."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT persona, COUNT(*) as cnt
        FROM history
        WHERE username = ?
        GROUP BY persona
        ORDER BY cnt DESC
    """, (username,))
    rows = cur.fetchall()
    conn.close()
    return {row["persona"]: row["cnt"] for row in rows}


# ---------------------------------------------------------
# Admin analytics queries
# ---------------------------------------------------------

def get_total_predictions() -> int:
    """Total number of prediction rows across all users."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as cnt FROM history")
    row = cur.fetchone()
    conn.close()
    return row["cnt"] if row else 0


def get_persona_distribution() -> list:
    """
    Returns a list of dicts sorted by count descending:
      [{"persona": "Luxury Buyer", "count": 42}, ...]
    Only personas that appear in history are included.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT persona, COUNT(*) as cnt
        FROM history
        GROUP BY persona
        ORDER BY cnt DESC
    """)
    rows = cur.fetchall()
    conn.close()
    return [{"persona": row["persona"], "count": row["cnt"]} for row in rows]


def get_customer_stats() -> dict:
    """
    Returns aggregate statistics across all history rows.
    Fields: avg_age, avg_income, avg_spending_score, total,
            min_age, max_age, min_income, max_income,
            male_count, female_count.
    Only includes non-null values in each average.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT
            COUNT(*) as total,
            AVG(CASE WHEN age IS NOT NULL THEN age END) as avg_age,
            MIN(CASE WHEN age IS NOT NULL THEN age END) as min_age,
            MAX(CASE WHEN age IS NOT NULL THEN age END) as max_age,
            AVG(CASE WHEN income IS NOT NULL THEN income END) as avg_income,
            MIN(CASE WHEN income IS NOT NULL THEN income END) as min_income,
            MAX(CASE WHEN income IS NOT NULL THEN income END) as max_income,
            AVG(CASE WHEN spending_score IS NOT NULL THEN spending_score END) as avg_spending_score,
            SUM(CASE WHEN LOWER(gender) = 'male' THEN 1 ELSE 0 END) as male_count,
            SUM(CASE WHEN LOWER(gender) = 'female' THEN 1 ELSE 0 END) as female_count
        FROM history
    """)
    row = cur.fetchone()
    conn.close()
    if row is None:
        return {
            "total": 0, "avg_age": None, "avg_income": None, "avg_spending_score": None,
            "min_age": None, "max_age": None,
            "min_income": None, "max_income": None,
            "male_count": 0, "female_count": 0,
        }
    return {
        "total": row["total"] or 0,
        "avg_age": round(row["avg_age"], 1) if row["avg_age"] is not None else None,
        "min_age": int(row["min_age"]) if row["min_age"] is not None else None,
        "max_age": int(row["max_age"]) if row["max_age"] is not None else None,
        "avg_income": round(row["avg_income"], 1) if row["avg_income"] is not None else None,
        "min_income": round(row["min_income"], 1) if row["min_income"] is not None else None,
        "max_income": round(row["max_income"], 1) if row["max_income"] is not None else None,
        "avg_spending_score": round(row["avg_spending_score"], 1) if row["avg_spending_score"] is not None else None,
        "male_count": int(row["male_count"] or 0),
        "female_count": int(row["female_count"] or 0),
    }


def get_most_recommended_products(top_n: int = 10) -> list:
    """
    Parses the JSON 'products' column in every history row.
    Each row stores a list of dicts with a 'name' key.
    Counts how many times each product name appears across ALL
    prediction rows and returns the top_n most frequent.
    Returns: [{"name": ..., "count": ...}, ...] sorted desc by count.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT products FROM history WHERE products IS NOT NULL")
    rows = cur.fetchall()
    conn.close()

    counter = Counter()
    for row in rows:
        try:
            products = json.loads(row["products"])
            if isinstance(products, list):
                for p in products:
                    if isinstance(p, dict) and p.get("name"):
                        counter[p["name"]] += 1
        except (json.JSONDecodeError, TypeError, KeyError):
            pass  # Skip malformed rows

    return [
        {"name": name, "count": cnt}
        for name, cnt in counter.most_common(top_n)
    ]


def get_persona_analysis() -> list:
    """
    Returns per-persona aggregated stats from the history table.
    For each persona: count, avg_income, avg_spending_score.
    Sorted descending by count.
    Returns: [{"persona": ..., "count": ..., "avg_income": ..., "avg_spending_score": ...}, ...]
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT
            persona,
            COUNT(*) as cnt,
            AVG(CASE WHEN income IS NOT NULL THEN income END) as avg_income,
            AVG(CASE WHEN spending_score IS NOT NULL THEN spending_score END) as avg_spending_score
        FROM history
        WHERE persona IS NOT NULL
        GROUP BY persona
        ORDER BY cnt DESC
    """)
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "persona": row["persona"],
            "count": row["cnt"],
            "avg_income": round(row["avg_income"], 1) if row["avg_income"] is not None else None,
            "avg_spending_score": round(row["avg_spending_score"], 1) if row["avg_spending_score"] is not None else None,
        }
        for row in rows
    ]


def get_product_category_stats() -> list:
    """
    Parses the JSON 'products' column in every history row and tallies
    recommendation counts per category.
    Each stored product dict has a 'category' key (set by recommend.py).
    Returns: [{"category": ..., "count": ...}, ...] sorted desc by count.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT products FROM history WHERE products IS NOT NULL")
    rows = cur.fetchall()
    conn.close()

    counter = Counter()
    for row in rows:
        try:
            products = json.loads(row["products"])
            if isinstance(products, list):
                for p in products:
                    if isinstance(p, dict) and p.get("category"):
                        counter[p["category"]] += 1
        except (json.JSONDecodeError, TypeError, KeyError):
            pass  # Skip malformed rows

    return [
        {"category": cat, "count": cnt}
        for cat, cnt in counter.most_common()
    ]



def get_segment_product_affinity() -> list:
    """
    Segment-Product Affinity: for each persona, tallies which products and
    product categories were recommended to it, from the JSON 'products'
    column of every history row (same parsing as the functions above).

    share = (recommendations for item / total recommendations for that persona) * 100

    Returns a list sorted by total_recommendations desc:
      [{"persona": ..., "predictions": ..., "total_recommendations": ...,
        "products":   [{"name": ..., "category": ..., "count": ..., "share": ...}, ...],
        "categories": [{"category": ..., "count": ..., "share": ...}, ...]}, ...]
    Products and categories are each sorted desc by count.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT persona, products FROM history
        WHERE persona IS NOT NULL AND products IS NOT NULL
    """)
    rows = cur.fetchall()
    conn.close()

    by_persona = {}
    for row in rows:
        try:
            products = json.loads(row["products"])
        except (json.JSONDecodeError, TypeError):
            continue  # Skip malformed rows
        if not isinstance(products, list):
            continue
        entry = by_persona.setdefault(row["persona"], {
            "predictions": 0, "products": Counter(),
            "categories": Counter(), "product_category": {},
        })
        entry["predictions"] += 1
        for p in products:
            if isinstance(p, dict) and p.get("name"):
                entry["products"][p["name"]] += 1
                if p.get("category"):
                    entry["categories"][p["category"]] += 1
                    entry["product_category"].setdefault(p["name"], p["category"])

    result = []
    for persona, entry in by_persona.items():
        total = sum(entry["products"].values())
        if total == 0:
            continue
        result.append({
            "persona": persona,
            "predictions": entry["predictions"],
            "total_recommendations": total,
            "products": [
                {"name": name, "category": entry["product_category"].get(name),
                 "count": cnt, "share": round(cnt / total * 100, 1)}
                for name, cnt in entry["products"].most_common()
            ],
            "categories": [
                {"category": cat, "count": cnt, "share": round(cnt / total * 100, 1)}
                for cat, cnt in entry["categories"].most_common()
            ],
        })
    result.sort(key=lambda x: x["total_recommendations"], reverse=True)
    return result


# Thresholds for Opportunity / Mismatch Detection. A gap is only reported
# when it is large in absolute terms AND the persona's share is at most
# half of the reference share, so small fluctuations are ignored.
CATEGORY_MIN_GAP_PP = 5.0          # category: gap of >= 5 percentage points
PRODUCT_MIN_REFERENCE_PCT = 20.0   # product: in >= 20% of other personas' predictions on average
PRODUCT_MIN_GAP_PP = 20.0          # product: gap of >= 20 percentage points
MAX_RELATIVE_SHARE = 0.5           # current share <= 50% of reference share


def get_opportunity_gaps() -> dict:
    """
    Opportunity / Mismatch Detection, built on get_segment_product_affinity().

    Category share for persona =
        (category recommendations for persona / total recommendations for persona) * 100
    Product rate for persona =
        (predictions for persona that included the product / predictions for persona) * 100
        (a product appears at most once per prediction, so this is the share
         of that persona's predictions in which it was recommended)

    For every persona x category (and persona x product), the reference value
    is the average of the same share across ALL OTHER personas (0 where the
    item was never recommended to them). gap = reference - current, in
    percentage points. Gaps that pass the thresholds above are returned,
    largest first.

    Product gaps are only reported inside categories already flagged as a
    category gap for that persona, i.e. they show which specific products
    drive the category gap. (Products are tiered by income/spend level, so
    persona-vs-everyone product differences outside that context mostly
    reflect intended segmentation rather than opportunities.)
    """
    affinity = get_segment_product_affinity()

    volume = [
        {
            "persona": a["persona"],
            "predictions": a["predictions"],
            "total_recommendations": a["total_recommendations"],
            "unique_products": len(a["products"]),
            "unique_categories": len(a["categories"]),
        }
        for a in affinity
    ]

    result = {
        "persona_count": len(affinity),
        "volume": volume,
        "category_gaps": [],
        "product_gaps": [],
    }
    if len(affinity) < 2:
        return result  # nothing to compare against

    cat_share = {a["persona"]: {c["category"]: c["share"] for c in a["categories"]} for a in affinity}
    prod_rate = {
        a["persona"]: {p["name"]: p["count"] / a["predictions"] * 100 for p in a["products"]}
        for a in affinity
    }
    prod_category = {}
    for a in affinity:
        for p in a["products"]:
            if p.get("category"):
                prod_category.setdefault(p["name"], p["category"])

    all_categories = sorted({c for shares in cat_share.values() for c in shares})
    all_products = sorted({p for rates in prod_rate.values() for p in rates})

    def find_gaps(values_by_persona, items, min_reference, min_gap):
        gaps = []
        for persona, values in values_by_persona.items():
            others = [v for p, v in values_by_persona.items() if p != persona]
            for item in items:
                current = values.get(item, 0.0)
                other_vals = [o.get(item, 0.0) for o in others]
                reference = sum(other_vals) / len(other_vals)
                gap = reference - current
                if (reference >= min_reference and gap >= min_gap
                        and current <= reference * MAX_RELATIVE_SHARE):
                    gaps.append({
                        "persona": persona,
                        "item": item,
                        "current_share": round(current, 1),
                        "reference_share": round(reference, 1),
                        "gap": round(gap, 1),
                        # other personas at or above the reference share
                        "strong_in": sum(1 for v in other_vals if v >= reference),
                        "compared_with": len(other_vals),
                    })
        gaps.sort(key=lambda g: (g["gap"], g["reference_share"]), reverse=True)
        return gaps

    predictions = {a["persona"]: a["predictions"] for a in affinity}

    for g in find_gaps(cat_share, all_categories, CATEGORY_MIN_GAP_PP, CATEGORY_MIN_GAP_PP):
        presence = "no" if g["current_share"] == 0 else "low"
        g["category"] = g.pop("item")
        g["predictions"] = predictions[g["persona"]]
        g["insight"] = (
            f"{g['category'].capitalize()} averages {g['reference_share']}% of recommendations "
            f"across the other {g['compared_with']} personas (at or above that level for "
            f"{g['strong_in']} of them), but has {presence} recommendation presence for "
            f"{g['persona']} at {g['current_share']}% — a gap of {g['gap']} percentage points."
        )
        result["category_gaps"].append(g)

    flagged = {(g["persona"], g["category"]) for g in result["category_gaps"]}
    for g in find_gaps(prod_rate, all_products, PRODUCT_MIN_REFERENCE_PCT, PRODUCT_MIN_GAP_PP):
        if (g["persona"], prod_category.get(g["item"])) not in flagged:
            continue
        g["product"] = g.pop("item")
        g["category"] = prod_category.get(g["product"])
        g["predictions"] = predictions[g["persona"]]
        when = ("is never recommended" if g["current_share"] == 0
                else f"is recommended in only {g['current_share']}% of predictions")
        g["insight"] = (
            f"{g['product']} is recommended in {g['reference_share']}% of predictions on average "
            f"for the other {g['compared_with']} personas, but {when} for {g['persona']}."
        )
        result["product_gaps"].append(g)

    return result


# ---------------------------------------------------------
# Period-over-Period Analysis
# ---------------------------------------------------------

PERIOD_MIN_PERIODS = 2
PERIOD_TOP_PRODUCTS = 10


def _pct_change(current, previous):
    """((current - previous) / previous) * 100; None when previous is 0/None."""
    if previous in (0, None) or current is None:
        return None
    return round((current - previous) / previous * 100, 1)


def _period_of(dt: datetime, granularity: str):
    """Returns (key, label, start, end) of the calendar period containing dt."""
    if granularity == "week":
        start = (dt - timedelta(days=dt.weekday())).date()
        end = start + timedelta(days=6)
        return start.isoformat(), f"Week of {start.strftime('%d %b %Y')}", start, end
    start = dt.date().replace(day=1)
    end = (start.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    return start.strftime("%Y-%m"), start.strftime("%B %Y"), start, end


def get_period_over_period(granularity: str = "month", period_key: str = None) -> dict:
    """
    Groups history rows into calendar periods (month or week) using ONLY the
    stored 'date' column, then compares a selected period (default: latest)
    with the previous period that has data.

    Percentage change = ((current - previous) / previous) * 100, None if previous is 0.
    Persona share     = persona predictions / period predictions * 100
    Category share    = category recommendations / period recommendations * 100
    Product rate      = period predictions that included the product / period predictions * 100
    Share changes are reported in percentage points (pp).
    """
    if granularity not in ("month", "week"):
        granularity = "month"

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT date, persona, spending_score, products FROM history WHERE date IS NOT NULL")
    rows = cur.fetchall()
    conn.close()

    periods = {}
    for row in rows:
        try:
            dt = datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S")
        except (TypeError, ValueError):
            continue  # Skip rows without a valid stored date
        key, label, start, end = _period_of(dt, granularity)
        p = periods.setdefault(key, {
            "key": key, "label": label, "start": start.isoformat(), "end": end.isoformat(),
            "predictions": 0, "score_sum": 0.0, "score_n": 0, "last_date": row["date"],
            "personas": Counter(), "products": Counter(), "categories": Counter(),
            "recommendations": 0,
        })
        p["predictions"] += 1
        p["last_date"] = max(p["last_date"], row["date"])
        if row["spending_score"] is not None:
            p["score_sum"] += row["spending_score"]
            p["score_n"] += 1
        if row["persona"]:
            p["personas"][row["persona"]] += 1
        try:
            products = json.loads(row["products"]) if row["products"] else []
        except (json.JSONDecodeError, TypeError):
            products = []
        if isinstance(products, list):
            seen = set()
            for prod in products:
                if isinstance(prod, dict) and prod.get("name"):
                    p["recommendations"] += 1
                    if prod["name"] not in seen:
                        p["products"][prod["name"]] += 1
                        seen.add(prod["name"])
                    if prod.get("category"):
                        p["categories"][prod["category"]] += 1

    ordered = [periods[k] for k in sorted(periods)]
    for p in ordered:
        p["avg_spending_score"] = round(p["score_sum"] / p["score_n"], 1) if p["score_n"] else None

    series = [
        {"key": p["key"], "label": p["label"], "predictions": p["predictions"],
         "avg_spending_score": p["avg_spending_score"]}
        for p in ordered
    ]
    result = {
        "granularity": granularity,
        "periods": series,
        "sufficient": len(ordered) >= PERIOD_MIN_PERIODS,
    }
    if not result["sufficient"]:
        result["message"] = ("Period-over-period analysis requires historical data "
                             "from at least two different periods.")
        return result

    keys = [p["key"] for p in ordered]
    idx = keys.index(period_key) if period_key in keys[1:] else len(ordered) - 1
    cur_p, prev_p = ordered[idx], ordered[idx - 1]
    cur_lbl, prev_lbl = cur_p["label"], prev_p["label"]

    # Is the selected period still in progress (today falls inside it)?
    today = datetime.now().date().isoformat()
    in_progress = cur_p["start"] <= today <= cur_p["end"]
    # Are the two periods back-to-back, or were there empty periods between?
    next_after_prev = (datetime.strptime(prev_p["end"], "%Y-%m-%d") + timedelta(days=1)).date().isoformat()
    adjacent = next_after_prev == cur_p["start"]

    volume = {
        "current": cur_p["predictions"], "previous": prev_p["predictions"],
        "change": cur_p["predictions"] - prev_p["predictions"],
        "pct_change": _pct_change(cur_p["predictions"], prev_p["predictions"]),
    }
    cs, ps = cur_p["avg_spending_score"], prev_p["avg_spending_score"]
    spending = {
        "current": cs, "previous": ps,
        "change": round(cs - ps, 1) if cs is not None and ps is not None else None,
        "pct_change": _pct_change(cs, ps),
    }

    def share_rows(cur_counter, prev_counter, cur_total, prev_total, name_key):
        out = []
        for name in set(cur_counter) | set(prev_counter):
            c, pv = cur_counter.get(name, 0), prev_counter.get(name, 0)
            c_sh = round(c / cur_total * 100, 1) if cur_total else 0.0
            p_sh = round(pv / prev_total * 100, 1) if prev_total else 0.0
            out.append({
                name_key: name, "current_count": c, "previous_count": pv,
                "count_pct_change": _pct_change(c, pv),
                "current_share": c_sh, "previous_share": p_sh,
                "share_change": round(c_sh - p_sh, 1),
            })
        out.sort(key=lambda r: (abs(r["share_change"]), r["current_count"]), reverse=True)
        return out

    persona_changes = share_rows(cur_p["personas"], prev_p["personas"],
                                 cur_p["predictions"], prev_p["predictions"], "persona")
    category_changes = share_rows(cur_p["categories"], prev_p["categories"],
                                  cur_p["recommendations"], prev_p["recommendations"], "category")
    product_changes = share_rows(cur_p["products"], prev_p["products"],
                                 cur_p["predictions"], prev_p["predictions"], "product")[:PERIOD_TOP_PRODUCTS]

    def top(counter, key, n=5):
        return [{key: name, "count": cnt} for name, cnt in counter.most_common(n)]

    # ---- Dynamically generated insights (from the numbers above only) ----
    insights = []
    if in_progress:
        insights.append(f"{cur_lbl} is still in progress (data up to {cur_p['last_date'][:10]}), "
                        f"so its totals may still change.")
    if not adjacent:
        insights.append(f"There is no stored history between {prev_lbl} and {cur_lbl}, "
                        f"so {cur_lbl} is compared with {prev_lbl}.")

    if volume["change"] == 0:
        insights.append(f"Customer analysis volume was unchanged compared with {prev_lbl} "
                        f"({volume['current']} analyses).")
    else:
        direction = "increased" if volume["change"] > 0 else "decreased"
        insights.append(f"Customer analysis volume {direction} by {abs(volume['pct_change'])}% "
                        f"compared with {prev_lbl} ({volume['previous']} → {volume['current']} analyses).")

    if spending["change"] is not None:
        if spending["change"] == 0:
            insights.append(f"Average spending score was unchanged at {cs} compared with {prev_lbl}.")
        else:
            direction = "increased" if spending["change"] > 0 else "decreased"
            insights.append(f"Average spending score {direction} by {abs(spending['change'])} points "
                            f"compared with {prev_lbl} ({ps} → {cs}).")

    rising = max(persona_changes, key=lambda r: r["share_change"], default=None)
    falling = min(persona_changes, key=lambda r: r["share_change"], default=None)
    if rising and rising["share_change"] > 0:
        insights.append(f"{rising['persona']} increased from {rising['previous_share']}% to "
                        f"{rising['current_share']}% of analyzed customers.")
    if falling and falling["share_change"] < 0:
        insights.append(f"{falling['persona']} decreased from {falling['previous_share']}% to "
                        f"{falling['current_share']}% of analyzed customers.")

    cat_up = max(category_changes, key=lambda r: r["share_change"], default=None)
    cat_down = min(category_changes, key=lambda r: r["share_change"], default=None)
    for r, word in ((cat_up, "rose"), (cat_down, "fell")):
        if r and r["share_change"] != 0 and (word == "rose") == (r["share_change"] > 0):
            count_part = (f"recommendations {'increased' if r['count_pct_change'] > 0 else 'decreased'} "
                          f"by {abs(r['count_pct_change'])}% ({r['previous_count']} → {r['current_count']})"
                          if r["count_pct_change"] not in (None, 0)
                          else f"recommendations went from {r['previous_count']} to {r['current_count']}")
            insights.append(f"{r['category'].capitalize()} {count_part}, and its share of all "
                            f"recommendations {word} from {r['previous_share']}% to {r['current_share']}%.")

    cur_top, prev_top = top(cur_p["products"], "name", 1), top(prev_p["products"], "name", 1)
    if cur_top and prev_top and cur_top[0]["name"] != prev_top[0]["name"]:
        insights.append(f"The most recommended product changed from {prev_top[0]['name']} in {prev_lbl} "
                        f"to {cur_top[0]['name']} in {cur_lbl}.")

    result.update({
        "current": {"key": cur_p["key"], "label": cur_lbl, "in_progress": in_progress,
                    "last_date": cur_p["last_date"]},
        "previous": {"key": prev_p["key"], "label": prev_lbl},
        "adjacent": adjacent,
        "volume": volume,
        "spending": spending,
        "persona_changes": persona_changes,
        "category_changes": category_changes,
        "product_changes": product_changes,
        "top_products": {"current": top(cur_p["products"], "name"), "previous": top(prev_p["products"], "name")},
        "top_categories": {"current": top(cur_p["categories"], "category"),
                           "previous": top(prev_p["categories"], "category")},
        "insights": insights,
    })
    return result


# ---------------------------------------------------------
# What-If Simulation (read-only lookups; nothing is written)
# ---------------------------------------------------------

def get_history_customers(limit: int = 200) -> list:
    """Most recent history rows (all users) for the What-If customer picker."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, username, age, gender, income, spending_score, persona, date
        FROM history
        WHERE income IS NOT NULL AND spending_score IS NOT NULL
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_history_record(record_id: int):
    """One history row by id (or None)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, username, age, gender, income, spending_score, cluster, persona, products, date
        FROM history WHERE id = ?
    """, (record_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None
