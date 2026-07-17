"""
Benchmarks llama.cpp's native KV-cache quantization (--cache-type-k /
--cache-type-v) at a fixed model weight quant level (Q4_K_M), measuring its
effect on peak RAM and throughput.

Uses a longer prompt/generation than the weight-quant benchmark specifically
so the KV cache is large enough for cache-type differences to show up in
peak RSS (KV cache size scales with context length, and at short contexts
the fixed model weights dominate RAM, masking the effect).

Quantized KV cache types require Flash Attention, hence `-fa on` throughout
(including the f16 baseline, so the comparison isolates cache-type effects
rather than mixing in a flash-attention-on/off variable).

Usage:
    python scripts/benchmark_kv_cache.py
"""
import json
import os

from benchmark import run_with_rss_tracking, REPO_ROOT, LLAMA_BENCH

MODEL_PATH = os.path.join(REPO_ROOT, "models", "gguf", "qwen3-0.6b-Q4_K_M.gguf")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

THREADS = 8
PP_TOKENS = 1024
GEN_TOKENS = 256
BENCH_REPETITIONS = 3

# (label, cache-type-k, cache-type-v)
CACHE_CONFIGS = [
    ("f16 (baseline)", "f16", "f16"),
    ("q8_0", "q8_0", "q8_0"),
    ("q4_0", "q4_0", "q4_0"),
]


def bench_kv_config(ctk, ctv):
    cmd = [
        LLAMA_BENCH, "-m", MODEL_PATH,
        "-p", str(PP_TOKENS), "-n", str(GEN_TOKENS),
        "-r", str(BENCH_REPETITIONS), "-t", str(THREADS),
        "-ctk", ctk, "-ctv", ctv, "-fa", "on",
        "-o", "json",
    ]
    stdout, peak_bytes = run_with_rss_tracking(cmd, REPO_ROOT)
    json_start = stdout.index("[")
    data = json.loads(stdout[json_start:])
    pp_ts = next(d["avg_ts"] for d in data if d["n_prompt"] > 0)
    tg_ts = next(d["avg_ts"] for d in data if d["n_gen"] > 0)
    return pp_ts, tg_ts, peak_bytes


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []

    for label, ctk, ctv in CACHE_CONFIGS:
        print(f"=== KV cache: {label} ===")
        pp_ts, tg_ts, peak_bytes = bench_kv_config(ctk, ctv)
        peak_mib = peak_bytes / (1024 * 1024)
        print(f"  pp={pp_ts:.1f} tok/s  tg={tg_ts:.1f} tok/s  peak_rss={peak_mib:.1f} MiB")
        rows.append({
            "cache_type": label,
            "pp_tok_s": round(pp_ts, 2),
            "tg_tok_s": round(tg_ts, 2),
            "peak_rss_mib": round(peak_mib, 1),
        })

    csv_path = os.path.join(RESULTS_DIR, "kv_cache_results.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        header = list(rows[0].keys())
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row[h]) for h in header) + "\n")
    print(f"\nWrote {csv_path}")

    baseline_rss = rows[0]["peak_rss_mib"]
    print("\n| KV cache type | PP tok/s | TG tok/s | Peak RSS (MiB) | RSS vs f16 |")
    print("|---------------|---------:|---------:|---------------:|-----------:|")
    for row in rows:
        delta = row["peak_rss_mib"] - baseline_rss
        print(f"| {row['cache_type']} | {row['pp_tok_s']} | {row['tg_tok_s']} | {row['peak_rss_mib']} | {delta:+.1f} |")


if __name__ == "__main__":
    main()
