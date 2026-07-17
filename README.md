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

Model weights and GGUF files are **not** committed to this repo (see
`.gitignore`) — they're multiple GB and don't belong in git. Instructions for
re-downloading/re-converting will be added here once step 2 is implemented.

## Results

_TBD — filled in as benchmarks complete._
