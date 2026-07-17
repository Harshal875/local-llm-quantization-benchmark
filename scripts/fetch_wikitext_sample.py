"""
Fetches a small subset of WikiText-2 (raw, test split) from Hugging Face and
saves it as a plain text file for use as a perplexity held-out sample.

Kept deliberately small (~aiming for a few thousand tokens) since perplexity
evaluation is run repeatedly across quant levels on CPU.
"""
import os
from datasets import load_dataset

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "wikitext2_sample.txt")
TARGET_CHARS = 20_000  # roughly enough for a handful of 512-token ppl chunks

def main():
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", streaming=True)

    lines = []
    total_chars = 0
    for row in ds:
        text = row["text"]
        stripped = text.strip()
        # skip empty lines and wikitext section headers (e.g. "= Title =")
        if not stripped or stripped.startswith("="):
            continue
        lines.append(text.rstrip("\n"))
        total_chars += len(text)
        if total_chars >= TARGET_CHARS:
            break

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Wrote {total_chars} chars ({len(lines)} lines) to {OUT_PATH}")


if __name__ == "__main__":
    main()
