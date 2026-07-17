# Local LLM Quantization Benchmark

Hands-on exploration of local LLM inference and quantization on CPU-only
hardware — built as a portfolio project for AI/ML placement interviews.
This is a learning/benchmarking project, not a production system.

## Hardware used

- CPU: Intel Core i5-13500H (12 cores / 16 threads)
- RAM: 16 GB
- GPU: Intel Iris Xe (integrated, no CUDA) — all inference is CPU-only
- OS: Windows 11

## Plan

1. Build `llama.cpp` locally (CPU-only build)
2. Download a small open-weight model from Hugging Face, convert to GGUF,
   quantize at multiple levels (Q8_0, Q5_K_M, Q4_K_M, Q3_K_M)
3. Benchmark each quant level: file size, RAM usage, tokens/sec
   (prompt processing + generation), perplexity on a WikiText-2 subset
4. Test llama.cpp's native KV-cache quantization (`--cache-type-k` /
   `--cache-type-v`) and measure RAM/speed impact
5. Compare against Ollama running the same model
6. Write up results with a table and chart
7. (Stretch) Sketch a LoRA fine-tuning script for Colab (not run locally —
   fine-tuning needs a GPU this laptop doesn't have)

Status: project scaffolding in progress. Results will be filled in below as
each step completes.

## Setup

Model weights, GGUF files, and the `llama.cpp` source tree are **not**
committed to this repo (see `.gitignore`) — they're either huge or trivially
re-fetchable. Below are the exact commands to reproduce the environment.

### 1. Build llama.cpp (CPU-only)

Tested on Windows with a MSYS2 UCRT64 toolchain (g++ 13.2, CMake 3.28, Ninja).
If you don't have MSYS2, install it from https://www.msys2.org/, then from
an MSYS2 UCRT64 shell:

```bash
pacman -S mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-toolchain ninja
```

Then, from this repo's root:

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# -D_WIN32_WINNT=0x0A00 works around a cpp-httplib build error on MinGW
# (it can't detect the Windows version and refuses to build without this)
cmake -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
  -DCMAKE_CXX_FLAGS="-D_WIN32_WINNT=0x0A00" -DCMAKE_C_FLAGS="-D_WIN32_WINNT=0x0A00"

cmake --build build --config Release -j 16
```

`-DGGML_NATIVE=ON` lets GGML auto-detect and use CPU instruction set
extensions available on this machine (AVX2 etc.) for faster inference.

The resulting binaries dynamically link against MSYS2's UCRT64 runtime DLLs
(`libstdc++-6.dll`, `libgomp-1.dll`, `libwinpthread-1.dll`, etc.). Copy them
next to the executables so they run without needing MSYS2 on `PATH`:

```bash
cp /c/msys64/ucrt64/bin/{libwinpthread-1,libgcc_s_seh-1,libcrypto-3-x64,libstdc++-6,libssl-3-x64,libgomp-1}.dll build/bin/
```

Verify: `./llama.cpp/build/bin/llama-cli.exe --version`

Key binaries produced: `llama-cli` (interactive/single-prompt inference),
`llama-bench` (throughput benchmarking), `llama-perplexity` (perplexity
eval), `llama-quantize` (GGUF quantization), `llama-server` (OpenAI-compatible
HTTP server).

### 2. Model download & quantization

**Model:** [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) — Apache-2.0,
instruct/chat model, 0.6B params, BF16 safetensors (~1.5GB on disk). Picked
over Gemma 3n E2B because Qwen's small dense models have mature, well-tested
GGUF/llama.cpp support, while Gemma 3n's "effective 2B" MatFormer architecture
has a larger real on-disk/RAM footprint than the name suggests and newer,
less battle-tested llama.cpp support. At 0.6B params this comfortably fits
this laptop's 16GB RAM even before quantization, leaving headroom to scale up
to Qwen3-1.7B/4B later if useful.

Set up the Python environment (from repo root):

```bash
python -m venv .venv
source .venv/Scripts/activate   # Git Bash / MSYS2; use .venv\Scripts\activate on cmd
pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
pip install "huggingface_hub[cli]"
```

Then download, convert, and quantize in one go:

```bash
bash scripts/download_and_convert.sh
```

This downloads the model to `models/qwen3-0.6b-hf/`, converts it to FP16
GGUF (`models/gguf/qwen3-0.6b-f16.gguf`), then produces 4 quantized copies:

| Quant   | Size (MiB) | Bits/weight | Notes |
|---------|-----------:|-------------|-------|
| Q8_0    | 761.8      | 8.50        | Near-lossless, largest of the 4 |
| Q5_K_M  | 520.2      | 5.81        | Good quality/size balance |
| Q4_K_M  | 456.1      | 5.09        | Common "sweet spot" default |
| Q3_K_M  | 389.1      | 4.34        | Smallest, most quality loss |

(FP16 baseline is 1433.75 MiB / 16 bits-per-weight for comparison.)

Note: `hf download` is the current CLI (the older `huggingface-cli download`
is deprecated and, on Windows, crashes on a Unicode deprecation-warning
glyph under the default `cp1252` console encoding — hence
`PYTHONIOENCODING=utf-8` in the script).

### 3. Benchmarking

Install the extra script dependencies (on top of the conversion deps from
step 2):

```bash
pip install -r scripts/requirements.txt
```

The perplexity eval needs a small held-out text sample. Fetch a ~20KB
WikiText-2 subset (test split, from the `Salesforce/wikitext` dataset on
Hugging Face) once:

```bash
python scripts/fetch_wikitext_sample.py
```

Then run the benchmark suite:

```bash
python scripts/benchmark.py
```

For each quant level, this sequentially (one model in memory at a time):
- Reads file size directly from disk
- Runs `llama-bench` (256 prompt tokens / 64 generation tokens, 3
  repetitions, 8 threads) for prompt-processing and generation throughput
- Runs `llama-perplexity` over 5 chunks of 512 tokens from the WikiText-2
  sample
- Polls the subprocess's resident set size (RSS) every 100ms in a background
  thread (via `psutil`) to record peak RAM used during each run

Results are written to `results/benchmark_results.csv` and printed as a
markdown table.

**What the metrics mean:**
- **File size**: on-disk size of the GGUF weights — the main lever
  quantization pulls.
- **PP tok/s (prompt processing)**: how fast the model ingests/encodes input
  tokens (relevant for long prompts, e.g. RAG context).
- **TG tok/s (token generation)**: how fast the model produces output tokens
  one at a time — this is usually what "feels slow" to a user chatting with
  the model, and is memory-bandwidth-bound on CPU (not compute-bound), which
  is why it barely changes across quant levels here — the bottleneck is
  moving weights through RAM, not the matrix-multiply itself.
- **Peak RSS**: actual physical RAM used during the run — the practical
  ceiling on what model+quant combination fits on a given machine.
- **Perplexity (PPL)**: how "surprised" the model is by the held-out text
  (lower = the model assigns higher probability to what actually comes
  next = better fit to that text distribution). It's a relative quality
  proxy, not a benchmark of "intelligence" — useful for seeing how much a
  quantization level degrades the model's underlying language modeling
  vs. the FP16 baseline.

**Results (Qwen3-0.6B, 8 threads, this laptop):**

| Quant  | Size (MiB) | PP tok/s | TG tok/s | Peak RSS (MiB) | Perplexity |
|--------|-----------:|---------:|---------:|----------------:|-----------:|
| F16    |     1439.4 |    181.5 |      7.1 |          2331.2 | 18.56 ± 1.76 |
| Q8_0   |      767.5 |    228.6 |      8.7 |          1796.4 | 18.46 ± 1.75 |
| Q5_K_M |      525.8 |    143.1 |      8.5 |          1609.8 | 18.65 ± 1.79 |
| Q4_K_M |      461.8 |    249.3 |      8.5 |          1769.3 | 20.17 ± 1.96 |
| Q3_K_M |      394.8 |    168.9 |      8.7 |          1608.6 | 25.90 ± 2.46 |

**Takeaways:**
- File size shrinks ~3.6x from F16 to Q3_K_M (1439 MiB → 395 MiB), as
  expected from ~16 bits/weight down to ~4.3 bits/weight.
- Generation speed (TG tok/s) is nearly flat (7-9 tok/s) across all quant
  levels — on CPU, single-token generation is memory-bandwidth bound, so
  a smaller model mostly helps by needing less RAM traffic per token, but
  the gain is modest at this model size. Prompt-processing (PP) numbers are
  noisier run-to-run (background CPU contention on a laptop with only 3
  repetitions per test) and shouldn't be read as a clean scaling trend
  here — this would benefit from more repetitions in a future pass.
- Perplexity is essentially unchanged through Q8_0 and Q5_K_M (within noise
  of the F16 baseline), starts drifting at Q4_K_M, and degrades noticeably
  at Q3_K_M (+~40% relative to F16) — consistent with the general rule of
  thumb that Q4_K_M is often the "sweet spot" and quality loss accelerates
  below it.
- Peak RAM roughly tracks file size plus a fairly constant ~1.1-1.2GB of
  overhead from the perplexity run's larger batch/context buffers (batch
  size 2048 even at ctx 512) — the throughput benchmark's peak RSS is
  consistently lower than the perplexity run's for the same quant.

## Results

Weight-quantization benchmark results are in the "Benchmarking" section
above. KV-cache quantization and Ollama comparison results will be added
here as those steps complete.
