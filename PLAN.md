# Garbled-GPT-2 • Implementation Plan

A living document tracking the **minimum-viable** steps to get GPT-2 inference running under Fancy-Garbling and to compare it against ordinary plaintext evaluation.

---

## 0. Scope & Assumptions

- Two-party semi-honest setting (Twopac/BMR16 from `fancy-garbling`).
- Public model weights, private user input.
- LAN connection; we use `TcpChannel` (`swanky::transports::TcpChannel`) so Garbler and Evaluator can run on separate machines.
- Rust 1.74+, nightly **not** required.

---

## 1. Milestones

| ID     | Deliverable                                                                                                                                                                                                                        | What we learn                                                                       |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **M1** | Skeleton workspace with two binaries: `garbler` and `evaluator` that open a socket and exchange a "ping" message.                                                                                                                  | Proves channel plumbing across machines. ✅                                         |
| **M2** | Plaintext baseline **in Python**: PyTorch script loads GPT-2, runs single-token inference, **exports quantised (Q8.8) weights/embeddings** to JSON/NumPy for Rust.                                                                 | Establishes latency/throughput baseline **and produces re-usable weight dumps**. ✅ |
| **M3** | GC demo with _one_ linear layer (matmul + bias) lifted from CNN example in `garbled-neural-network-experiments`. Input vector length = hidden size (768).                                                                          | Validates encode/reveal workflow, obtains first GC timing. ✅                       |
| **M4** | **(in-progress)** Dynamic-length embedding support, Python helper dumps full sequence (`--dump-embeddings`), garbler streams length-prefixed data, evaluator auto-sizes. _Next_: load real quantised GPT-2 weights & per-token GC. | End-to-end flow for any prompt length; foundation for true model weights.           |
| **M5** | Benchmark harness for **fresh garbling** every run – report wall-time, ciphertext bytes; OT should remain zero (weights are public).                                                                                               | Baseline GC cost.                                                                   |

These are intentionally shallow; once M5 is solid we iterate toward full transformer blocks (see full roadmap in `docs/roadmap.md`).

---

## 2. Implementation Notes

- **Channel**: both binaries accept `--listen <addr:port>` OR `--connect <addr:port>`. Rookie-friendly: garbler listens, evaluator connects.
- **RNG & OT**: weights are public ⇒ evaluator has _no private inputs_, so **no OT is performed**. We still supply `AlszSender/Receiver` placeholders to satisfy Twopac's type signatures, but their cost is zero.
- **Linear Layer Gadget**: copy `linear_layer_gc()` from `garbled-neural-network-experiments` (it already builds a Fancy-Arithmetic matmul). Adjust modulus to 2¹⁶ and quantise inputs with scale 256.
- **Plaintext Path (Python)**: all baseline and quantisation steps live in `plaintext_baseline.py` (PyTorch). Export helper writes `{model/*.npy}` for all weights.
- **GC Path (Rust)** : only secure computation lives in Rust using `fancy-garbling`; consumes the NumPy/JSON dumps produced by Python.

---

## 3. Benchmark Script Outline

```bash
# build everything
cargo build --release

# run garbler in one terminal
cargo run --bin garbler --listen 0.0.0.0:7000 --input "The quick brown fox"

# run evaluator in another (can be remote)
cargo run --bin evaluator --connect garbler.host:7000
```

The garbler prints plaintext logits; evaluator prints GC logits + stats, plus both sides log wall-clock duration.

---

## 4. Next Steps

1. Replace linear layer with full Transformer block (attention + FFN).
2. Add streaming evaluation for every block.
3. GPU offload for evaluator-side matmuls.

_(Edit this file as milestones are completed.)_
