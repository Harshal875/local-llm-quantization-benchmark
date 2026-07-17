"""
Benchmarks each quantized GGUF model: file size, peak RAM during inference,
prompt-processing / generation tokens-per-second, and perplexity on the
WikiText-2 sample.

Usage:
    python scripts/benchmark.py

Writes results/benchmark_results.csv and prints a markdown table.
"""
import json
import os
import re
import subprocess
import threading
import time

import psutil

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLAMA_BENCH = os.path.join(REPO_ROOT, "llama.cpp", "build", "bin", "llama-bench.exe")
LLAMA_PERPLEXITY = os.path.join(REPO_ROOT, "llama.cpp", "build", "bin", "llama-perplexity.exe")
GGUF_DIR = os.path.join(REPO_ROOT, "models", "gguf")
PPL_DATA = os.path.join(REPO_ROOT, "data", "wikitext2_sample.txt")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

QUANT_LEVELS = ["f16", "Q8_0", "Q5_K_M", "Q4_K_M", "Q3_K_M"]
THREADS = 8
PP_TOKENS = 256
GEN_TOKENS = 64
BENCH_REPETITIONS = 3
PPL_CTX = 512
PPL_CHUNKS = 5


class PeakRSSMonitor:
    """Polls a subprocess's (and its children's) RSS in a background thread
    to find the peak working-set size during its run."""

    def __init__(self, pid, interval=0.1):
        self.pid = pid
        self.interval = interval
        self.peak_bytes = 0
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            proc = psutil.Process(self.pid)
        except psutil.NoSuchProcess:
            return
        while not self._stop:
            try:
                total = proc.memory_info().rss
                for child in proc.children(recursive=True):
                    total += child.memory_info().rss
                self.peak_bytes = max(self.peak_bytes, total)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            time.sleep(self.interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop = True
        self._thread.join(timeout=2)


def run_with_rss_tracking(cmd, cwd):
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    monitor = PeakRSSMonitor(proc.pid)
    monitor.start()
    stdout, _ = proc.communicate()
    monitor.stop()
    return stdout, monitor.peak_bytes


def bench_throughput(model_path):
    cmd = [
        LLAMA_BENCH, "-m", model_path,
        "-p", str(PP_TOKENS), "-n", str(GEN_TOKENS),
        "-r", str(BENCH_REPETITIONS), "-t", str(THREADS),
        "-o", "json",
    ]
    stdout, peak_bytes = run_with_rss_tracking(cmd, REPO_ROOT)
    json_start = stdout.index("[")
    data = json.loads(stdout[json_start:])
    pp_ts = next(d["avg_ts"] for d in data if d["n_prompt"] > 0)
    tg_ts = next(d["avg_ts"] for d in data if d["n_gen"] > 0)
    return pp_ts, tg_ts, peak_bytes


def bench_perplexity(model_path):
    cmd = [
        LLAMA_PERPLEXITY, "-m", model_path,
        "-f", PPL_DATA, "-c", str(PPL_CTX), "--chunks", str(PPL_CHUNKS),
        "-t", str(THREADS),
    ]
    stdout, peak_bytes = run_with_rss_tracking(cmd, REPO_ROOT)
    match = re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", stdout)
    if not match:
        raise RuntimeError(f"Could not parse perplexity output:\n{stdout}")
    ppl, ppl_stderr = float(match.group(1)), float(match.group(2))
    return ppl, ppl_stderr, peak_bytes


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []

    for quant in QUANT_LEVELS:
        model_path = os.path.join(GGUF_DIR, f"qwen3-0.6b-{quant}.gguf")
        if not os.path.exists(model_path):
            print(f"Skipping {quant}: {model_path} not found")
            continue

        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"=== {quant} ({file_size_mb:.1f} MiB) ===")

        print("  Running throughput benchmark...")
        pp_ts, tg_ts, bench_peak = bench_throughput(model_path)
        print(f"    pp={pp_ts:.1f} tok/s  tg={tg_ts:.1f} tok/s  peak_rss={bench_peak / (1024*1024):.1f} MiB")

        print("  Running perplexity eval...")
        ppl, ppl_stderr, ppl_peak = bench_perplexity(model_path)
        print(f"    PPL={ppl:.3f} +/- {ppl_stderr:.3f}  peak_rss={ppl_peak / (1024*1024):.1f} MiB")

        rows.append({
            "quant": quant,
            "file_size_mib": round(file_size_mb, 1),
            "pp_tok_s": round(pp_ts, 2),
            "tg_tok_s": round(tg_ts, 2),
            "peak_rss_mib_bench": round(bench_peak / (1024 * 1024), 1),
            "perplexity": round(ppl, 3),
            "perplexity_stderr": round(ppl_stderr, 3),
            "peak_rss_mib_ppl": round(ppl_peak / (1024 * 1024), 1),
        })

    csv_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        header = list(rows[0].keys())
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row[h]) for h in header) + "\n")
    print(f"\nWrote {csv_path}")

    print("\n| Quant | Size (MiB) | PP tok/s | TG tok/s | Peak RSS (MiB) | Perplexity |")
    print("|-------|-----------:|---------:|---------:|---------------:|-----------:|")
    for row in rows:
        peak = max(row["peak_rss_mib_bench"], row["peak_rss_mib_ppl"])
        print(f"| {row['quant']} | {row['file_size_mib']} | {row['pp_tok_s']} | {row['tg_tok_s']} | {peak} | {row['perplexity']} ± {row['perplexity_stderr']} |")


if __name__ == "__main__":
    main()
