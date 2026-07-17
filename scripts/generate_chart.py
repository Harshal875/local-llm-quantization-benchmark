"""
Generates results/quantization_chart.png: file size and perplexity vs. quant
level, from results/benchmark_results.csv.

Usage:
    python scripts/generate_chart.py
"""
import csv
import os

import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(REPO_ROOT, "results", "benchmark_results.csv")
OUT_PATH = os.path.join(REPO_ROOT, "results", "quantization_chart.png")

# Display order (smallest/most-quantized to largest), independent of CSV row order
QUANT_ORDER = ["Q3_K_M", "Q4_K_M", "Q5_K_M", "Q8_0", "f16"]


def main():
    with open(CSV_PATH, encoding="utf-8") as f:
        rows = {row["quant"]: row for row in csv.DictReader(f)}

    quants = [q for q in QUANT_ORDER if q in rows]
    sizes = [float(rows[q]["file_size_mib"]) for q in quants]
    ppl = [float(rows[q]["perplexity"]) for q in quants]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = "tab:blue"
    ax1.set_xlabel("Quantization level")
    ax1.set_ylabel("File size (MiB)", color=color1)
    bars = ax1.bar(quants, sizes, color=color1, alpha=0.7, label="File size")
    ax1.tick_params(axis="y", labelcolor=color1)
    for bar, size in zip(bars, sizes):
        ax1.text(bar.get_x() + bar.get_width() / 2, size + 15, f"{size:.0f}", ha="center", fontsize=9)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Perplexity (WikiText-2 sample, lower = better)", color=color2)
    ax2.plot(quants, ppl, color=color2, marker="o", linewidth=2, label="Perplexity")
    ax2.tick_params(axis="y", labelcolor=color2)
    for x, y in zip(quants, ppl):
        ax2.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color=color2)

    plt.title("Qwen3-0.6B: file size vs. perplexity by quantization level")
    fig.tight_layout()
    plt.savefig(OUT_PATH, dpi=150)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
