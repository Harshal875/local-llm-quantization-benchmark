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

## Results

_TBD — filled in as benchmarks complete._
