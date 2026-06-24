"""
benchmark.py
Head-to-head comparison of the ML detector vs the ModSecurity firewall.

It takes a balanced sample of safe and attack queries, sends each one to
BOTH systems, and reports accuracy, false positives, false negatives and
average latency for each. This produces the core results of the project.

Before running, make sure BOTH are up:
  1. ML detector:   python src/detector/app.py        (http://localhost:5000)
  2. ModSecurity:   docker compose up -d   (in waf/)   (http://localhost:8080)

Then run from project root:  python benchmark.py
"""

import time
import urllib.parse

import pandas as pd
import requests

DATA = "data/clean_dataset.csv"
ML_URL = "http://localhost:5000/api/check"
WAF_URL = "http://localhost:8080/get"
SAMPLE_PER_CLASS = 200   # how many safe + how many attack queries to test


def get_test_set():
    """Take an equal number of safe and attack queries at random."""
    df = pd.read_csv(DATA).dropna(subset=["query", "label"])
    df["query"] = df["query"].astype(str)
    attacks = df[df["label"] == 1].sample(SAMPLE_PER_CLASS, random_state=1)
    safe = df[df["label"] == 0].sample(SAMPLE_PER_CLASS, random_state=1)
    test = pd.concat([attacks, safe]).sample(frac=1, random_state=1)
    return test.reset_index(drop=True)


def ask_ml(query):
    """Returns (blocked: bool, latency_ms: float). blocked=True means 'attack'."""
    start = time.time()
    r = requests.post(ML_URL, json={"query": query}, timeout=10)
    latency = (time.time() - start) * 1000
    return r.json().get("blocked", False), latency


def ask_waf(query):
    """Returns (blocked: bool, latency_ms: float). 403 means the WAF blocked it."""
    url = WAF_URL + "?id=" + urllib.parse.quote(query)
    start = time.time()
    r = requests.get(url, timeout=10)
    latency = (time.time() - start) * 1000
    return r.status_code == 403, latency


def evaluate(name, ask_fn, test):
    tp = fp = tn = fn = 0
    latencies = []
    for _, row in test.iterrows():
        is_attack = row["label"] == 1
        try:
            blocked, latency = ask_fn(row["query"])
        except Exception as e:
            print(f"  request failed: {str(e)[:60]}")
            continue
        latencies.append(latency)
        if is_attack and blocked:
            tp += 1
        elif is_attack and not blocked:
            fn += 1            # missed attack
        elif not is_attack and blocked:
            fp += 1            # false alarm
        else:
            tn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0
    detection = tp / (tp + fn) if (tp + fn) else 0     # recall on attacks
    fp_rate = fp / (fp + tn) if (fp + tn) else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    print(f"\n{'=' * 50}\n{name}")
    print(f"  attacks caught (detection rate): {detection:.1%}  ({tp}/{tp + fn})")
    print(f"  attacks MISSED                 : {fn}")
    print(f"  false alarms on safe input     : {fp}  ({fp_rate:.1%})")
    print(f"  overall accuracy               : {accuracy:.1%}")
    print(f"  average latency                : {avg_latency:.2f} ms")
    return {
        "system": name, "detection_rate": detection, "missed": fn,
        "false_positives": fp, "accuracy": accuracy, "avg_latency_ms": avg_latency,
    }


def main():
    test = get_test_set()
    print(f"Testing {len(test)} queries "
          f"({SAMPLE_PER_CLASS} attacks + {SAMPLE_PER_CLASS} safe) on each system...")

    results = []
    results.append(evaluate("ML Detector", ask_ml, test))
    results.append(evaluate("ModSecurity WAF", ask_waf, test))

    # Save the results so you can use them in your write-up
    pd.DataFrame(results).to_csv("data/benchmark_results.csv", index=False)
    print(f"\n{'=' * 50}\nSaved results to data/benchmark_results.csv")


if __name__ == "__main__":
    main()
