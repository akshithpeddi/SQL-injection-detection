import time
import urllib.parse
import joblib
import pandas as pd
import requests

DATA = "data/clean_dataset.csv"
ML_URL = "http://127.0.0.1:5000/api/check"
WAF_URL = "http://127.0.0.1:8080/get"
SAMPLE_PER_CLASS = 200
session = requests.Session()

def get_test_set():
    df = pd.read_csv(DATA).dropna(subset=["query", "label"])
    df["query"] = df["query"].astype(str)
    attacks = df[df["label"] == 1].sample(SAMPLE_PER_CLASS, random_state=1)
    safe = df[df["label"] == 0].sample(SAMPLE_PER_CLASS, random_state=1)
    return pd.concat([attacks, safe]).sample(frac=1, random_state=1).reset_index(drop=True)

def measure_core_model_latency(test):
    vec = joblib.load("models/vectorizer.joblib")
    model = joblib.load("models/logreg.joblib")
    times = []
    for q in test["query"]:
        start = time.perf_counter()
        model.predict(vec.transform([q]))
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return sum(times) / len(times), times[len(times) // 2]

def ask_ml(query):
    start = time.perf_counter()
    r = session.post(ML_URL, json={"query": query}, timeout=10)
    return r.json().get("blocked", False), (time.perf_counter() - start) * 1000

def ask_waf(query):
    url = WAF_URL + "?id=" + urllib.parse.quote(query)
    start = time.perf_counter()
    r = session.get(url, timeout=10)
    return r.status_code == 403, (time.perf_counter() - start) * 1000

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
        if is_attack and blocked: tp += 1
        elif is_attack and not blocked: fn += 1
        elif not is_attack and blocked: fp += 1
        else: tn += 1
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0
    detection = tp / (tp + fn) if (tp + fn) else 0
    fp_rate = fp / (fp + tn) if (fp + tn) else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    print(f"\n{'=' * 55}\n{name}")
    print(f"  attacks caught (detection rate): {detection:.1%}  ({tp}/{tp + fn})")
    print(f"  attacks MISSED                 : {fn}")
    print(f"  false alarms on safe input     : {fp}  ({fp_rate:.1%})")
    print(f"  overall accuracy               : {accuracy:.1%}")
    print(f"  end-to-end latency (HTTP)      : {avg_latency:.2f} ms")
    return {"system": name, "detection_rate": round(detection, 4), "missed": fn,
            "false_positives": fp, "accuracy": round(accuracy, 4),
            "end_to_end_latency_ms": round(avg_latency, 3)}

def main():
    test = get_test_set()
    print(f"Testing {len(test)} queries ({SAMPLE_PER_CLASS} attacks + {SAMPLE_PER_CLASS} safe) on each system...")
    core_avg, core_median = measure_core_model_latency(test)
    print(f"\n{'=' * 55}\nML model core prediction time (no network):")
    print(f"  average: {core_avg:.3f} ms   median: {core_median:.3f} ms")
    results = []
    results.append(evaluate("ML Detector", ask_ml, test))
    results.append(evaluate("ModSecurity WAF", ask_waf, test))
    results[0]["core_model_latency_ms"] = round(core_avg, 3)
    pd.DataFrame(results).to_csv("data/benchmark_results.csv", index=False)
    print(f"\n{'=' * 55}\nSaved results to data/benchmark_results.csv")

if __name__ == "__main__":
    main()
