"""
Compares Ollama against raw llama.cpp using the *same* GGUF weights
(qwen3-0.6b-Q4_K_M.gguf, imported into Ollama via ollama/Modelfile) so the
comparison isolates serving-stack overhead rather than differences in the
underlying quantized weights.

Prerequisites:
- Ollama installed and running (the background service starts automatically
  after install on Windows; `ollama serve` if you need to start it manually)
- Model imported: `cd ollama && ollama create qwen3-0.6b-q4km -f Modelfile`

An initial run of this comparison surfaced a bigger finding than "Ollama vs
llama.cpp": on this laptop's hybrid P-core/E-core CPU (13th Gen Intel
i5-13500H), llama.cpp's generation speed is NOT monotonically improved by
more threads — it peaks around 2 threads and gets *worse* at 8+ threads,
because single-token decode is memory-bandwidth/latency-bound, and more
threads on a hybrid core layout adds scheduling contention without adding
usable bandwidth. So this script benchmarks llama.cpp at both a naive
default (8 threads) and the empirically-found sweet spot (2 threads) to
give Ollama a fair comparison rather than a strawman.

Usage:
    python scripts/benchmark_ollama.py
"""
import json
import os
import threading
import time
import urllib.request
import uuid

import psutil

from benchmark import run_with_rss_tracking, REPO_ROOT, LLAMA_BENCH

MODEL_PATH = os.path.join(REPO_ROOT, "models", "gguf", "qwen3-0.6b-Q4_K_M.gguf")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
WIKITEXT_SAMPLE = os.path.join(REPO_ROOT, "data", "wikitext2_sample.txt")

PP_TOKENS_TARGET = 256  # approximate; actual depends on tokenizer
GEN_TOKENS = 64
BENCH_REPETITIONS = 3
OLLAMA_MODEL = "qwen3-0.6b-q4km"
OLLAMA_URL = "http://localhost:11434/api/generate"


def load_prompt(char_budget=1000):
    # A random nonce prefix guarantees this prompt has never been seen by the
    # server before, defeating llama.cpp server's prompt-prefix cache (which
    # persists across separate script runs, not just within one run — a
    # fixed prompt read from a file makes every re-run after the first look
    # artificially instant).
    with open(WIKITEXT_SAMPLE, "r", encoding="utf-8") as f:
        text = f.read()[:char_budget]
    return f"[run-{uuid.uuid4()}] {text}"


def bench_llamacpp(threads):
    cmd = [
        LLAMA_BENCH, "-m", MODEL_PATH,
        "-p", str(PP_TOKENS_TARGET), "-n", str(GEN_TOKENS),
        "-r", str(BENCH_REPETITIONS), "-t", str(threads),
        "-o", "json",
    ]
    stdout, peak_bytes = run_with_rss_tracking(cmd, REPO_ROOT)
    json_start = stdout.index("[")
    data = json.loads(stdout[json_start:])
    pp_ts = next(d["avg_ts"] for d in data if d["n_prompt"] > 0)
    tg_ts = next(d["avg_ts"] for d in data if d["n_gen"] > 0)
    return pp_ts, tg_ts, peak_bytes / (1024 * 1024)


class ExternalProcessRSSMonitor:
    """Tracks peak combined RSS of all processes matching a name substring
    (for tracking Ollama's externally-managed llama-server.exe runner)."""

    def __init__(self, name_substring, interval=0.1):
        self.name_substring = name_substring.lower()
        self.interval = interval
        self.peak_bytes = 0
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop:
            total = 0
            for proc in psutil.process_iter(["name"]):
                try:
                    if self.name_substring in (proc.info["name"] or "").lower():
                        total += proc.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            self.peak_bytes = max(self.peak_bytes, total)
            time.sleep(self.interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop = True
        self._thread.join(timeout=2)


def bench_ollama(prompt):
    # Warmup with a *different* prompt than the timed run: llama.cpp's server
    # (which Ollama wraps) caches matching prompt prefixes per-slot, so
    # warming up with the same prompt we then time would make prompt
    # processing look artificially near-instant on the timed call.
    _ollama_request("Describe your favorite hobby in a few sentences.", num_predict=8)

    monitor = ExternalProcessRSSMonitor("llama-server")
    monitor.start()
    response = _ollama_request(prompt, num_predict=GEN_TOKENS)
    monitor.stop()

    pp_ts = response["prompt_eval_count"] / (response["prompt_eval_duration"] / 1e9)
    tg_ts = response["eval_count"] / (response["eval_duration"] / 1e9)
    return pp_ts, tg_ts, monitor.peak_bytes / (1024 * 1024)


def _ollama_request(prompt, num_predict):
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": num_predict, "num_ctx": 1024},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    prompt = load_prompt()
    rows = []

    print("=== llama.cpp, 8 threads (naive default) ===")
    pp, tg, rss = bench_llamacpp(threads=8)
    print(f"  pp={pp:.1f} tok/s  tg={tg:.1f} tok/s  peak_rss={rss:.1f} MiB")
    rows.append({"engine": "llama.cpp (-t 8)", "pp_tok_s": round(pp, 2), "tg_tok_s": round(tg, 2), "peak_rss_mib": round(rss, 1)})

    print("=== llama.cpp, 2 threads (empirical sweet spot on this CPU) ===")
    pp, tg, rss = bench_llamacpp(threads=2)
    print(f"  pp={pp:.1f} tok/s  tg={tg:.1f} tok/s  peak_rss={rss:.1f} MiB")
    rows.append({"engine": "llama.cpp (-t 2)", "pp_tok_s": round(pp, 2), "tg_tok_s": round(tg, 2), "peak_rss_mib": round(rss, 1)})

    print("=== Ollama (same Q4_K_M weights, default thread config) ===")
    pp, tg, rss = bench_ollama(prompt)
    print(f"  pp={pp:.1f} tok/s  tg={tg:.1f} tok/s  peak_rss={rss:.1f} MiB")
    rows.append({"engine": "Ollama", "pp_tok_s": round(pp, 2), "tg_tok_s": round(tg, 2), "peak_rss_mib": round(rss, 1)})

    csv_path = os.path.join(RESULTS_DIR, "ollama_comparison.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        header = list(rows[0].keys())
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row[h]) for h in header) + "\n")
    print(f"\nWrote {csv_path}")

    print("\n| Engine | PP tok/s | TG tok/s | Peak RSS (MiB) |")
    print("|--------|---------:|---------:|----------------:|")
    for row in rows:
        print(f"| {row['engine']} | {row['pp_tok_s']} | {row['tg_tok_s']} | {row['peak_rss_mib']} |")


if __name__ == "__main__":
    main()
