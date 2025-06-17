"""Plaintext baseline + quantisation helper.

Usage examples
--------------
Run timing baseline:
    python plaintext_baseline.py --text "The quick brown fox"

Dump quantised embedding (Q8.8) for the first token of a prompt:
    python plaintext_baseline.py --dump-embedding --text "The quick brown fox" --out embedding.npy
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def quantise_to_q88(tensor: torch.Tensor) -> np.ndarray:
    """Quantise a float32 tensor to signed-16-bit Q8.8 fixed-point."""
    return torch.round(tensor * 256).to(torch.int16).cpu().numpy()


def dump_embedding(text: str, output: Path | str) -> None:
    """Save the first-token embedding (quantised) to *output* (.npy)."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    with torch.no_grad():
        ids = tokenizer(text, return_tensors="pt")["input_ids"]
        emb = model.transformer.wte(ids)[0, 0]  # first token
        np.save(output, quantise_to_q88(emb))

    print(f"[python] wrote quantised embedding to {output}")


def dump_embeddings(text: str, output: Path | str) -> None:
    """Save quantised embeddings for *all* tokens in the prompt to *output* (.npy).

    Output shape: (n_tokens, hidden_size).
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    with torch.no_grad():
        ids = tokenizer(text, return_tensors="pt")["input_ids"]
        emb = model.transformer.wte(ids)[0]  # shape (seq_len, hidden)
        np.save(output, quantise_to_q88(emb))

    print(f"[python] wrote quantised embeddings for {emb.shape[0]} tokens to {output}")


def run_baseline(text: str) -> None:
    """Plaintext GPT-2 forward & log timing."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    inputs = tokenizer(text, return_tensors="pt")
    start = time.time()
    with torch.no_grad():
        _ = model(**inputs)
    elapsed = (time.time() - start) * 1e3
    print(f"[python] baseline inference: {elapsed:.1f} ms")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--text", type=str, required=True, help="Prompt text")
    p.add_argument(
        "--dump-embedding",
        action="store_true",
        help="Write Q8.8 embedding .npy and exit",
    )
    p.add_argument("--out", type=Path, default=Path("embedding.npy"))
    p.add_argument(
        "--dump-embeddings",
        action="store_true",
        help="Write Q8.8 embeddings for all tokens (.npy) and exit",
    )
    args = p.parse_args()

    if args.dump_embedding:
        dump_embedding(args.text, args.out)
    elif args.dump_embeddings:
        dump_embeddings(args.text, args.out)
    else:
        run_baseline(args.text)


if __name__ == "__main__":
    main()
