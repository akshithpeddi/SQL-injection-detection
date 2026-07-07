"""
make_charts.py
Turns benchmark_results.csv into comparison charts for the dissertation.
Saves a combined figure and individual charts into a 'figures' folder.
Run from project root:  python make_charts.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS = "data/benchmark_results.csv"
OUT = "figures"
SAFE_N = 200   # number of safe queries tested (from benchmark SAMPLE_PER_CLASS)

COLORS = ["#2E86C1", "#C0392B"]   # ML = blue, WAF = red


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(RESULTS)
    systems = df["system"].tolist()

    # Prepare the four metrics (as percentages / ms)
    accuracy = (df["accuracy"] * 100).tolist()
    detection = (df["detection_rate"] * 100).tolist()
    fp_rate = (df["false_positives"] / SAFE_N * 100).tolist()
    latency = df["end_to_end_latency_ms"].tolist()

    charts = [
        ("Overall Accuracy (%)", accuracy, "%.1f%%", 100),
        ("Attack Detection Rate (%)", detection, "%.1f%%", 100),
        ("False Alarm Rate on Safe Input (%)", fp_rate, "%.1f%%", 100),
        ("Average Latency (ms)", latency, "%.1f ms", None),
    ]

    # Combined 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (title, vals, fmt, ymax) in zip(axes.flat, charts):
        bars = ax.bar(systems, vals, color=COLORS, width=0.55)
        ax.set_title(title, fontsize=12, fontweight="bold")
        if ymax:
            ax.set_ylim(0, ymax * 1.12)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt % v, ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("ML Detector vs ModSecurity WAF", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{OUT}/comparison_all.png", dpi=150)
    plt.close(fig)

    # Individual charts too
    names = ["accuracy", "detection_rate", "false_alarm_rate", "latency"]
    for (title, vals, fmt, ymax), name in zip(charts, names):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        bars = ax.bar(systems, vals, color=COLORS, width=0.55)
        ax.set_title(title, fontsize=12, fontweight="bold")
        if ymax:
            ax.set_ylim(0, ymax * 1.12)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt % v, ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(f"{OUT}/{name}.png", dpi=150)
        plt.close(fig)

    print(f"Saved charts to '{OUT}/': comparison_all.png + 4 individual charts")


if __name__ == "__main__":
    main()
