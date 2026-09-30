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
from datetime import datetime

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
    Fields: avg_age, avg_income, avg_spending_score, total.
    Only includes non-null values in each average.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT
            COUNT(*) as total,
            AVG(CASE WHEN age IS NOT NULL THEN age END) as avg_age,
            AVG(CASE WHEN income IS NOT NULL THEN income END) as avg_income,
            AVG(CASE WHEN spending_score IS NOT NULL THEN spending_score END) as avg_spending_score
        FROM history
    """)
    row = cur.fetchone()
    conn.close()
    if row is None:
        return {"total": 0, "avg_age": None, "avg_income": None, "avg_spending_score": None}
    return {
        "total": row["total"] or 0,
        "avg_age": round(row["avg_age"], 1) if row["avg_age"] is not None else None,
        "avg_income": round(row["avg_income"], 1) if row["avg_income"] is not None else None,
        "avg_spending_score": round(row["avg_spending_score"], 1) if row["avg_spending_score"] is not None else None,
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
