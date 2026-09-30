"""
server.py
Flask backend for the Customer Analysis System.

Serves the "Persona" frontend (static/index.html) and exposes a small
JSON API that the frontend's JavaScript calls instead of the fake
in-browser demo logic:

  POST /api/register    -> create account (bcrypt + SQLite)
  POST /api/login        -> log in, starts a server-side session
  POST /api/logout       -> clears the session
  GET  /api/session      -> check if the browser is currently logged in
  GET  /api/personas     -> all cluster personas (name, color, centroid) for
                             the landing-page legend + cluster map background
  POST /api/predict      -> real K-Means prediction + persona + products,
                             saved to that user's history
  GET  /api/history       -> this user's past predictions
  GET  /api/admin/stats  -> admin-only: total predictions + persona distribution

Run with:
    python server.py
Then open http://localhost:5000
"""

import os
import secrets
import json

import joblib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, session

from db import (
    init_db,
    register_user,
    verify_user,
    get_user_role,
    save_prediction,
    get_history,
    get_total_predictions,
    get_persona_distribution,
    get_customer_stats,
    get_most_recommended_products,
    get_persona_analysis,
)
from persona_config import PERSONA_MAP
from recommend import get_recommendations

app = Flask(__name__, static_folder="static", static_url_path="")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))

init_db()

scaler = joblib.load("model/scaler.pkl")
kmeans = joblib.load("model/kmeans_model.pkl")


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def persona_payload(cluster_id: int) -> dict:
    info = PERSONA_MAP.get(cluster_id)
    if info is None:
        return {
            "cluster": cluster_id,
            "persona": "Unknown",
            "description": "No persona mapped for this cluster.",
            "color": "#999999",
            "centroid": {"x": 50, "y": 50},
        }
    return {"cluster": cluster_id, **info}


def login_required():
    return "username" in session


def admin_required():
    """True only if the current session belongs to an authenticated admin user."""
    return "username" in session and session.get("role") == "admin"


# ---------------------------------------------------------
# Static frontend
# ---------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ---------------------------------------------------------
# Auth API
# ---------------------------------------------------------

@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    email = (data.get("email") or "").strip()
    password = data.get("password") or ""

    # Role is never accepted from the client — always 'user'.
    ok, message = register_user(username, password, email)
    status = 200 if ok else 400
    return jsonify({"success": ok, "message": message}), status


@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    ok, message = verify_user(username, password)
    if ok:
        role = get_user_role(username)
        session["username"] = username
        session["role"] = role
        return jsonify({
            "success": True,
            "message": message,
            "username": username,
            "role": role,
        })
    return jsonify({"success": False, "message": message}), 401


@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"success": True})


@app.route("/api/session", methods=["GET"])
def api_session():
    if login_required():
        return jsonify({
            "logged_in": True,
            "username": session["username"],
            "role": session.get("role", "user"),
        })
    return jsonify({"logged_in": False})


# ---------------------------------------------------------
# Persona / legend data (public, no login needed)
# ---------------------------------------------------------

@app.route("/api/personas", methods=["GET"])
def api_personas():
    return jsonify([persona_payload(cid) for cid in sorted(PERSONA_MAP.keys())])


# ---------------------------------------------------------
# Prediction API
# ---------------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not login_required():
        return jsonify({"success": False, "message": "Please log in first."}), 401

    data = request.get_json(silent=True) or {}
    try:
        age = int(data.get("age"))
        gender = str(data.get("gender"))
        income = float(data.get("income"))
        spending_score = float(data.get("spending_score"))
    except (TypeError, ValueError):
        return jsonify({"success": False, "message": "Invalid input values."}), 400

    if age < 10 or age > 100:
        return jsonify({"success": False, "message": "Age must be between 10 and 100."}), 400
    # Income is in Annual_Income_k (USD thousands). Range: $1.2k–$240k ≈ ₹1L–₹2Cr
    if income < 1.2 or income > 240:
        return jsonify({"success": False, "message": "Income must be between ₹1 lakh and ₹2 crore."}), 400
    if not (1 <= spending_score <= 100):
        return jsonify({"success": False, "message": "Spending score must be between 1 and 100."}), 400

    # Age and Gender are still collected and saved to history for
    # context, but are NOT fed into the clustering model -- see
    # train_model.py for why (clustering uses Income + Spending only,
    # the two features that actually define our personas).
    raw_features = np.array([[income, spending_score]])
    scaled_features = scaler.transform(raw_features)
    cluster = int(kmeans.predict(scaled_features)[0])

    info = persona_payload(cluster)
    recs = get_recommendations(income, spending_score, top_n=50, top_picks=20)

    # Flatten to one list for storage, tagging each item's tier so
    # history can still show "top pick" vs "more" later.
    products_for_storage = (
        [{**p, "tier": "top"} for p in recs["top"]] +
        [{**p, "tier": "more"} for p in recs["more"]]
    )

    save_prediction(
        session["username"], age, gender, income, spending_score,
        cluster, info["persona"], products_for_storage,
    )

    return jsonify({
        "success": True,
        "input": {"age": age, "gender": gender, "income": income, "spending_score": spending_score},
        "products": recs["top"] + recs["more"],
        "top_picks": recs["top"],
        "more_options": recs["more"],
        **info,
    })


# ---------------------------------------------------------
# History API
# ---------------------------------------------------------

@app.route("/api/history", methods=["GET"])
def api_history():
    if not login_required():
        return jsonify({"success": False, "message": "Please log in first."}), 401

    rows = get_history(session["username"])
    result = []
    for r in rows:
        info = PERSONA_MAP.get(r["cluster"], {})
        try:
            products = json.loads(r["products"]) if r["products"] else []
        except (json.JSONDecodeError, TypeError):
            products = []
        result.append({
            "age": r["age"],
            "gender": r["gender"],
            "income": r["income"],
            "spending_score": r["spending_score"],
            "cluster": r["cluster"],
            "persona": r["persona"],
            "products": products,
            "top_picks": [p for p in products if p.get("tier") == "top"],
            "more_options": [p for p in products if p.get("tier") == "more"],
            "color": info.get("color", "#999999"),
            "date": r["date"],
        })
    return jsonify(result)


# ---------------------------------------------------------
# Admin Analytics API
# ---------------------------------------------------------

@app.route("/api/admin/stats", methods=["GET"])
def api_admin_stats():
    if not login_required():
        return jsonify({"success": False, "message": "Please log in first."}), 401
    if not admin_required():
        return jsonify({"success": False, "message": "Access denied. Admin only."}), 403

    total = get_total_predictions()
    distribution = get_persona_distribution()

    # Enrich distribution with color from PERSONA_MAP (look up by name).
    # Build a name→color lookup from PERSONA_MAP so we never duplicate the mapping.
    name_to_color = {info["persona"]: info["color"] for info in PERSONA_MAP.values()}

    enriched = []
    for item in distribution:
        pct = round((item["count"] / total * 100), 1) if total > 0 else 0.0
        enriched.append({
            "persona": item["persona"],
            "count": item["count"],
            "pct": pct,
            "color": name_to_color.get(item["persona"], "#7C83FD"),
        })

    most_common = enriched[0] if enriched else None

    # --- New analytics ---
    customer_stats = get_customer_stats()
    top_products = get_most_recommended_products(top_n=10)
    persona_analysis = get_persona_analysis()

    # Enrich persona_analysis with persona color.
    for pa in persona_analysis:
        pa["color"] = name_to_color.get(pa["persona"], "#7C83FD")

    # Highest spending persona (max avg_spending_score, ignoring None).
    spending_ranked = [
        pa for pa in persona_analysis if pa["avg_spending_score"] is not None
    ]
    highest_spending = (
        max(spending_ranked, key=lambda x: x["avg_spending_score"])
        if spending_ranked else None
    )

    # Highest income persona (max avg_income, ignoring None).
    income_ranked = [
        pa for pa in persona_analysis if pa["avg_income"] is not None
    ]
    highest_income = (
        max(income_ranked, key=lambda x: x["avg_income"])
        if income_ranked else None
    )

    # --- Business Insights ---------------------------------------------
    # Simple, automatically generated business-level insights. Every value
    # below is derived from the analytics already computed above — no new
    # calculations, no hardcoded persona/product names, no external AI.

    # 1. Persona insights: `enriched` is already sorted by count desc.
    least_common = enriched[-1] if enriched else None

    # 2. Product insights: reuse get_most_recommended_products but ask for
    #    the full ranked list so we can also read the least-recommended
    #    end. The Counter inside it only contains products that actually
    #    appear in stored recommendations, so a never-recommended product
    #    can never be picked as "least".
    all_recommended_products = get_most_recommended_products(top_n=10_000_000)
    most_recommended_product = all_recommended_products[0] if all_recommended_products else None
    least_recommended_product = all_recommended_products[-1] if all_recommended_products else None

    # 3. Spending insights: reuse the spending-ranked personas from above.
    lowest_spending = (
        min(spending_ranked, key=lambda x: x["avg_spending_score"])
        if spending_ranked else None
    )

    # 4. Income insights: reuse the income-ranked personas from above.
    lowest_income = (
        min(income_ranked, key=lambda x: x["avg_income"])
        if income_ranked else None
    )

    business_insights = {
        "persona": {"most_common": most_common, "least_common": least_common},
        "products": {
            "most_recommended": most_recommended_product,
            "least_recommended": least_recommended_product,
        },
        "spending": {"highest": highest_spending, "lowest": lowest_spending},
        "income": {"highest": highest_income, "lowest": lowest_income},
    }

    return jsonify({
        "success": True,
        "total_predictions": total,
        "most_common_persona": most_common,
        "distribution": enriched,
        "customer_stats": customer_stats,
        "top_products": top_products,
        "persona_analysis": persona_analysis,
        "highest_spending_persona": highest_spending,
        "highest_income_persona": highest_income,
        "business_insights": business_insights,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
