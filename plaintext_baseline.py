#!/usr/bin/env python3

# PYTHON BASELINE & QUANTISATION HELPER
# -------------------------------------
# Provides two modes:
# 1. `--baseline`              → run GPT-2 forward pass and print timing
# 2. `--dump-embedding --text` → dump quantised embedding (Q8.8) to .npy for Rust

import time
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def quantise_to_q88(tensor: torch.Tensor) -> np.ndarray:
    """Quantise a float32 tensor to signed 16-bit Q8.8 fixed-point."""
    scaled = torch.round(tensor * 256).to(torch.int16)
    return scaled.cpu().numpy().astype(np.int16)


def baseline_inference(
    text: str = "The quick brown fox", model_name: str = "gpt2"
) -> None:
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt")
    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    dur_ms = (time.time() - start) * 1e3

    next_id = int(outputs.logits[0, -1].argmax())
    next_tok = tokenizer.decode([next_id])
    print(f"[python] Baseline next token: '{next_tok.strip()}' in {dur_ms:.2f} ms")


def dump_embedding(text: str, output: str, model_name: str = "gpt2") -> None:
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    with torch.no_grad():
        ids = tokenizer(text, return_tensors="pt")["input_ids"]
        emb = model.transformer.wte(ids)[0, 0]
        np.save(output, quantise_to_q88(emb))
    print(f"[python] wrote quantised embedding to {output} (shape={emb.shape})")


def main():
    parser = argparse.ArgumentParser(
        description="GPT-2 plaintext baseline & embedding dumper"
    )
    parser.add_argument(
        "--dump-embedding",
        action="store_true",
        help="Dump first-token embedding to .npy",
    )
    parser.add_argument(
        "--text", type=str, default="The quick brown fox", help="Input text"
    )
    parser.add_argument(
        "--out", type=str, default="embedding.npy", help="Output .npy filename"
    )
    args = parser.parse_args()

    if args.dump_embedding:
        dump_embedding(args.text, args.out)
    else:
        baseline_inference(args.text)


if __name__ == "__main__":
    main()
