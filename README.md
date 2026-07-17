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

_TBD — added in the next step._

## Results

_TBD — filled in as benchmarks complete._
