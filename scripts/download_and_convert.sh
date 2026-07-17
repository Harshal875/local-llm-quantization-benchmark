#!/usr/bin/env bash
# Downloads Qwen3-0.6B from Hugging Face, converts to GGUF (FP16), and
# quantizes to the 4 levels used in this project's benchmarks.
#
# Prerequisites: llama.cpp built at ./llama.cpp/build (see README "Setup"
# section), and the Python venv at ./.venv with
# llama.cpp/requirements/requirements-convert_hf_to_gguf.txt + huggingface_hub installed.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_ID="Qwen/Qwen3-0.6B"
HF_DIR="models/qwen3-0.6b-hf"
GGUF_DIR="models/gguf"
BASE_NAME="qwen3-0.6b"

source .venv/Scripts/activate
export PYTHONIOENCODING=utf-8

mkdir -p "$HF_DIR" "$GGUF_DIR"

echo "=== Downloading $MODEL_ID ==="
hf download "$MODEL_ID" --local-dir "$HF_DIR"

echo "=== Converting to GGUF (FP16) ==="
python llama.cpp/convert_hf_to_gguf.py "$HF_DIR" \
  --outfile "$GGUF_DIR/${BASE_NAME}-f16.gguf" --outtype f16

echo "=== Quantizing ==="
for q in Q8_0 Q5_K_M Q4_K_M Q3_K_M; do
  echo "--- $q ---"
  ./llama.cpp/build/bin/llama-quantize.exe \
    "$GGUF_DIR/${BASE_NAME}-f16.gguf" \
    "$GGUF_DIR/${BASE_NAME}-${q}.gguf" \
    "$q"
done

echo "=== Done. Files in $GGUF_DIR: ==="
ls -la "$GGUF_DIR"
