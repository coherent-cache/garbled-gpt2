# Garbled-GPT-2 • Implementation Plan

A living document tracking the **minimum-viable** steps to get GPT-2 inference running under Fancy-Garbling and to compare it against ordinary plaintext evaluation.

---

## 0. Scope & Assumptions

- Two-party semi-honest setting (Twopac/BMR16 from `fancy-garbling`).
- Public model weights, private user input.
- LAN connection; we use `TcpChannel` (`swanky::transports::TcpChannel`) so Garbler and Evaluator can run on separate machines.
- Rust 1.74+, nightly **not** required.

---

## 1. Milestones (Quick Wins)

| ID     | Deliverable                                                                                                                                               | What we learn                                                                             |
| ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **M1** | Skeleton workspace with two binaries: `garbler` and `evaluator` that open a socket and exchange a "ping" message.                                         | Proves channel plumbing across machines. ✅                                               |
| **M2** | Plaintext baseline: load tiny GPT-2 (Hugging-Face `ggml` or `rust-bert`), run single-token inference & time it.                                           | Establishes latency/throughput baseline. ✅                                               |
| **M3** | GC demo with _one_ linear layer (matmul + bias) lifted from CNN example in `garbled-neural-network-experiments`. Input vector length = hidden size (768). | Validates encode/ reveal workflow, obtains first GC timing. ✅                            |
| **M4** | Integrate local tokenizer + embedding (runs in the Garbler) and stream encoded embedding to Evaluator where same linear layer runs in GC.                 | Converts CNN code-path to transformer shape; measures extra overhead of network transfer. |
| **M5** | Side-by-side benchmark harness printing: plaintext µs, GC µs, cipher size, OT count.                                                                      | Hard numbers for first blogpost / discussion.                                             |

These are intentionally shallow; once M5 is solid we iterate toward full transformer blocks (see full roadmap in `docs/roadmap.md`).

---

## 2. Implementation Notes

- **Channel**: both binaries accept `--listen <addr:port>` OR `--connect <addr:port>`. Rookie-friendly: garbler listens, evaluator connects.
- **RNG & OT**: use default `rand_chacha` RNG; start with `dummy` OT (local testing), then swap to IKNP (`swanky_ot::iknp`). Same code signature.
- **Linear Layer Gadget**: copy `linear_layer_gc()` from `garbled-neural-network-experiments` (it already builds a Fancy-Arithmetic matmul). Adjust modulus to 2¹⁶ and quantise inputs with scale 256.
- **Plaintext Path**: call identical Rust function but on `i16` values without encoding.

---

## 3. Benchmark Script Outline

```bash
# build everything
cargo build --release

# run garbler in one terminal
./target/release/garbler --listen 0.0.0.0:7000 --input "The quick brown fox"

# run evaluator in another (can be remote)
./target/release/evaluator --connect garbler.host:7000
```

The garbler prints plaintext logits; evaluator prints GC logits + stats, plus both sides log wall-clock duration.

---

## 4. Next Steps (after quick wins)

1. Replace linear layer with full Transformer block (attention + FFN).
2. Add streaming re-randomisation every block.
3. GPU offload for evaluator-side matmuls.
4. Security hardening (malicious check, transcript audit).

_(Edit this file as milestones are completed.)_
