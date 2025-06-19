"""Plaintext baseline + quantisation helper.

Usage examples
--------------
Run timing baseline:
    python plaintext_baseline.py --text "The quick brown fox"

Dump quantised embedding (Q8.8) for the first token of a prompt:
    python plaintext_baseline.py --dump-embedding --text "The quick brown fox" --out embedding.npy

Dump transformer block weights:
    python plaintext_baseline.py --dump-block-weights --block-id 0 --out-dir weights/
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


def dump_transformer_block_weights(block_id: int, out_dir: Path | str) -> None:
    """Extract and quantize weights for Q, K, V, O, FC1, FC2 from a transformer block.

    Args:
        block_id: Which transformer block to extract (0-11 for GPT-2 small)
        out_dir: Directory to save the .npy weight files

    Saves files:
        {out_dir}/block_{block_id}_q.npy - Query projection weights
        {out_dir}/block_{block_id}_k.npy - Key projection weights
        {out_dir}/block_{block_id}_v.npy - Value projection weights
        {out_dir}/block_{block_id}_o.npy - Output projection weights
        {out_dir}/block_{block_id}_fc1.npy - First FFN layer weights
        {out_dir}/block_{block_id}_fc2.npy - Second FFN layer weights
    """
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    block = model.transformer.h[block_id]

    # Extract attention weights (Q, K, V are packed in c_attn)
    # c_attn contains [Q, K, V] concatenated along the output dimension
    c_attn_weight = block.attn.c_attn.weight  # shape: [hidden_size, 3 * hidden_size]
    hidden_size = c_attn_weight.shape[0]

    # Split into Q, K, V
    q_weight = c_attn_weight[:, :hidden_size]  # Query weights
    k_weight = c_attn_weight[:, hidden_size : 2 * hidden_size]  # Key weights
    v_weight = c_attn_weight[:, 2 * hidden_size :]  # Value weights

    # Output projection weights
    o_weight = block.attn.c_proj.weight  # shape: [hidden_size, hidden_size]

    # FFN weights
    fc1_weight = block.mlp.c_fc.weight  # shape: [hidden_size, 4 * hidden_size]
    fc2_weight = block.mlp.c_proj.weight  # shape: [4 * hidden_size, hidden_size]

    # Quantize and save each matrix
    weights = {
        "q": q_weight,
        "k": k_weight,
        "v": v_weight,
        "o": o_weight,
        "fc1": fc1_weight,
        "fc2": fc2_weight,
    }

    for name, weight in weights.items():
        filename = out_dir / f"block_{block_id}_{name}.npy"
        quantized = quantise_to_q88(weight)
        np.save(filename, quantized)
        print(f"[python] wrote {name} weights {weight.shape} -> {filename}")

    print(f"[python] saved all transformer block {block_id} weights to {out_dir}")


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
    p.add_argument("--text", type=str, help="Prompt text")
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
    p.add_argument(
        "--dump-block-weights",
        action="store_true",
        help="Extract transformer block weights (Q,K,V,O,FC1,FC2) and exit",
    )
    p.add_argument(
        "--block-id",
        type=int,
        default=0,
        help="Which transformer block to extract (0-11)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("weights"),
        help="Directory for weight files",
    )
    args = p.parse_args()

    if args.dump_embedding:
        if not args.text:
            p.error("--dump-embedding requires --text")
        dump_embedding(args.text, args.out)
    elif args.dump_embeddings:
        if not args.text:
            p.error("--dump-embeddings requires --text")
        dump_embeddings(args.text, args.out)
    elif args.dump_block_weights:
        dump_transformer_block_weights(args.block_id, args.out_dir)
    else:
        if not args.text:
            p.error("--text is required for baseline run")
        run_baseline(args.text)


if __name__ == "__main__":
    main()
