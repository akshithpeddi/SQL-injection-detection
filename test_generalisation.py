"""
test_generalisation.py
Tests the trained detector on attacks it was NOT trained on, to see how well
it generalises. Two tests:

  PART A - attacks I generated myself with sqlmap (parsed from the traffic log).
  PART B - 'evasion-style' attacks written as clean SQL, to probe weaknesses.

This gives a fairer picture than the benchmark alone, because these attacks
come from a different source than the training data.

Run from project root:  python test_generalisation.py
"""

import os
import re
import urllib.parse

import joblib

# The sqlmap traffic log (created when you ran sqlmap). Adjust if needed.
LOG_CANDIDATES = [
    "../DVWA/sqli_traffic.log",
    "sqli_traffic.log",
    "data/sqli_traffic.log",
]

vec = joblib.load("models/vectorizer.joblib")
model = joblib.load("models/logreg.joblib")


def predict(text):
    return int(model.predict(vec.transform([text]))[0]) == 1  # True = attack/blocked


def load_sqlmap_payloads():
    path = next((p for p in LOG_CANDIDATES if os.path.exists(p)), None)
    if not path:
        return None
    text = open(path, encoding="utf-8", errors="ignore").read()
    raw = re.findall(r'[?&]id=([^&\s]+)', text)
    seen, payloads = set(), []
    for r in raw:
        d = urllib.parse.unquote(r).strip()
        # skip the plain non-attack value and duplicates
        if d and d != "1" and d not in seen:
            seen.add(d)
            payloads.append(d)
    return payloads


def run_test(name, payloads):
    if not payloads:
        print(f"\n{name}: no payloads found.")
        return
    caught = sum(predict(p) for p in payloads)
    rate = caught / len(payloads)
    print(f"\n{'=' * 55}\n{name}")
    print(f"  detected: {caught}/{len(payloads)} = {rate:.1%}")
    missed = [p for p in payloads if not predict(p)]
    if missed:
        print(f"  missed ({len(missed)}):")
        for m in missed[:8]:
            print("   -", m[:65])
    else:
        print("  missed: none")


# PART A: your own sqlmap attacks
sqlmap_payloads = load_sqlmap_payloads()
if sqlmap_payloads is None:
    print("Could not find sqli_traffic.log. Set its path in LOG_CANDIDATES.")
else:
    run_test("PART A - self-generated sqlmap attacks", sqlmap_payloads)

# PART B: evasion-style attacks written as clean SQL
evasion = [
    "UNION SELECT username, password FROM users",
    "SELECT * FROM accounts WHERE id = 1 OR 1 = 1",
    "DROP TABLE users",
    "admin' --",
    "1 OR 1=1",
    "'; EXEC xp_cmdshell('dir'); --",
    "SeLeCt * FrOm users WhErE 1=1",
    "/*!50000UNION*/ SELECT 1,2,3",
]
run_test("PART B - evasion-style / clean SQL attacks", evasion)

print(f"\n{'=' * 55}")
print("Compare these to ~98.5% on the benchmark test set.")
