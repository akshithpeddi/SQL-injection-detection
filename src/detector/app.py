"""
app.py  --  Real-time SQL injection detector

A small Flask web service that loads the trained model and checks every
input in real time. Safe input is allowed through; anything flagged as an
attack is BLOCKED and written to a log file. It also reports how long each
check took, which is the latency measurement your project cares about.

Run from your project root:
    python src/detector/app.py
Then open http://localhost:5000 in your browser.
"""

import os
import time
from datetime import datetime

import joblib
from flask import Flask, render_template_string, request, jsonify

# --- Paths (work no matter where you run from) ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "blocked.log")

# --- Load the model once at startup ---
# Using Logistic Regression because it predicts fastest (best for real time).
# To compare, swap "logreg.joblib" for "randomforest.joblib".
vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.joblib"))
model = joblib.load(os.path.join(MODEL_DIR, "logreg.joblib"))

app = Flask(__name__)


def check_query(text):
    """Classify one input. Returns (is_attack, confidence, latency_ms)."""
    start = time.time()
    vec = vectorizer.transform([text])
    is_attack = int(model.predict(vec)[0]) == 1
    confidence = float(model.predict_proba(vec)[0].max())
    latency_ms = (time.time() - start) * 1000
    return is_attack, confidence, latency_ms


def log_blocked(text, confidence):
    """Write a blocked attack to the log file with a timestamp."""
    line = f"{datetime.now().isoformat()} | BLOCKED | conf={confidence:.2f} | {text}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)


PAGE = """
<!doctype html>
<title>Real-Time SQLi Detector</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 640px; margin: 60px auto; padding: 0 20px; }
  h1 { font-size: 22px; }
  p.sub { color: #555; }
  input[type=text] { width: 100%; padding: 12px; font-size: 16px; box-sizing: border-box; }
  button { margin-top: 12px; padding: 10px 20px; font-size: 16px; cursor: pointer; }
  .result { margin-top: 24px; padding: 16px; border-radius: 8px; }
  .blocked { background: #fde8e8; border: 1px solid #f5a3a3; }
  .allowed { background: #e6f6ea; border: 1px solid #9ad4ab; }
  .verdict { font-weight: bold; font-size: 18px; }
  .meta { color: #444; margin-top: 6px; font-size: 14px; }
  code { background: #f1f1f1; padding: 2px 5px; border-radius: 4px; }
</style>
<h1>Real-Time SQL Injection Detector</h1>
<p class="sub">Type an input as if it were a login or search field. The detector checks it before it would reach the database.</p>
<form method="post">
  <input type="text" name="query" placeholder="e.g. 1' OR '1'='1" value="{{ query|e }}" autofocus>
  <button type="submit">Check</button>
</form>
{% if verdict %}
<div class="result {{ 'blocked' if is_attack else 'allowed' }}">
  <div class="verdict">{{ '🚫 BLOCKED — looks like an attack' if is_attack else '✅ ALLOWED — looks safe' }}</div>
  <div class="meta">Input: <code>{{ query|e }}</code></div>
  <div class="meta">Confidence: {{ '%.0f' % (confidence * 100) }}% &nbsp;|&nbsp; Checked in {{ '%.2f' % latency }} ms</div>
</div>
{% endif %}
"""


@app.route("/", methods=["GET", "POST"])
def home():
    ctx = {"verdict": False, "query": ""}
    if request.method == "POST":
        query = request.form.get("query", "")
        is_attack, confidence, latency = check_query(query)
        if is_attack:
            log_blocked(query, confidence)
        ctx = {
            "verdict": True, "query": query, "is_attack": is_attack,
            "confidence": confidence, "latency": latency,
        }
    return render_template_string(PAGE, **ctx)


@app.route("/api/check", methods=["POST"])
def api_check():
    """JSON endpoint: POST {"query": "..."} -> verdict. Handy for testing."""
    data = request.get_json(force=True, silent=True) or {}
    query = data.get("query", "")
    is_attack, confidence, latency = check_query(query)
    if is_attack:
        log_blocked(query, confidence)
    return jsonify({
        "query": query,
        "result": "attack" if is_attack else "safe",
        "blocked": is_attack,
        "confidence": round(confidence, 4),
        "latency_ms": round(latency, 3),
    })


if __name__ == "__main__":
    print("Detector running at http://localhost:5000")
    app.run(debug=True, port=5000)
