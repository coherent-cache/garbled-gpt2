# Project Task List

This document tracks high-level milestones and common recurring tasks for the **garbled-gpt2** project. Each task is assigned a difficulty rating (all leaf tasks must remain _Easy_ by project policy) and a subjective confidence estimate.

---

## Milestone Tasks

| Step                                                   | Description                                                                                                                                        | Difficulty | Confidence |
| ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ---------- |
| **1 Finish Milestone M4 – Dynamic Embedding Path**     |                                                                                                                                                    |            |            |
| 1.1                                                    | Extend `plaintext_baseline.py` with `--dump-embeddings` flag that writes length-prefixed NumPy file(s) for a given prompt                          | Easy       | 90 %       |
| 1.2                                                    | Add `--embeddings <file>` CLI option to Rust `garbler` to mmap or load the `.npy` produced in 1.1 (no Python invocation)                           | Easy       | 85 %       |
| 1.3                                                    | Prefix-encode embedding length (`u32`) on the TCP channel before streaming the vector to the evaluator                                             | Easy       | 90 %       |
| 1.4                                                    | Update `evaluator` to read the length, allocate modulus-16 wires, and reuse existing GC matmul gadget                                              | Easy       | 80 %       |
| 1.5                                                    | Integration test: run `plaintext_baseline.py --dump-embeddings` → launch Rust `garbler` + `evaluator`; verify logits match plaintext within ±1 LSB | Easy       | 75 %       |
| **2 Milestone M5 – Baseline Benchmark Harness**        |                                                                                                                                                    |            |            |
| 2.1                                                    | Measure wall, garble, eval, reveal times; read `bytes_sent`/`bytes_recv` from `TcpChannel` counters                                                | Easy       | 90 %       |
| 2.2                                                    | Capture peak memory (Rust via `jemalloc` stats; Python via `psutil`) and append `mem_mb` to CSV                                                    | Easy       | 80 %       |
| 2.3                                                    | Update `scripts/summarise.py` to digest new CSV schema and emit Markdown/LaTeX tables                                                              | Easy       | 85 %       |
| **3 Single GPT-2 Transformer Block (attention + FFN)** |                                                                                                                                                    |            |            |
| 3.1                                                    | Python helper converts quantised weights for Q, K, V, O, FC1, FC2 into `.npy` dumps (one file per matrix)                                          | Easy       | 80 %       |
| 3.2                                                    | Rust loader maps these `.npy` matrices and feeds them into GC linear-layer gadget (Q/K/V/O)                                                        | Easy       | 80 %       |
| 3.3                                                    | Implement GC matmul for attention scores (Q·Kᵀ) under modulus 2¹⁶                                                                                  | Easy       | 75 %       |
| 3.4                                                    | Apply scale = 1/√d during weight quantisation (Python-side) to avoid runtime divisions                                                             | Easy       | 85 %       |
| 3.5                                                    | Choose degree-3 polynomial approximation for softmax; document error bounds                                                                        | Easy       | 70 %       |
| 3.6                                                    | Build polynomial gadget via repeated `mul` + `cmul` gates                                                                                          | Easy       | 75 %       |
| 3.7                                                    | Validate softmax outputs against baseline (≤ 1 % relative error)                                                                                   | Easy       | 70 %       |
| 3.8                                                    | Multiply softmax weights by V → context vectors                                                                                                    | Easy       | 80 %       |
| 3.9                                                    | Pass context through output linear O                                                                                                               | Easy       | 80 %       |
| 3.10                                                   | FFN: FC1 → ReLU (polynomial) → FC2; reuse gadgets                                                                                                  | Easy       | 75 %       |
| 3.11                                                   | End-to-end test: one block logits vs. PyTorch baseline                                                                                             | Easy       | 70 %       |
| **4 Stack All 12 Blocks (GPT-2 Small)**                |                                                                                                                                                    |            |            |
| 4.1                                                    | Loop over block gadget 12× with weight file offsets                                                                                                | Easy       | 80 %       |
| 4.2                                                    | Add positional embeddings in Python dump; Rust simply adds vector element-wise before first block                                                  | Easy       | 85 %       |
| 4.3                                                    | Verify numerical drift every 3 blocks on sample prompts                                                                                            | Easy       | 70 %       |
| **5 Performance & Memory Optimisation**                |                                                                                                                                                    |            |            |
| 5.1                                                    | Profile evaluator memory via `jemalloc`/heaptrack; locate largest buffers                                                                          | Easy       | 75 %       |
| 5.2                                                    | Stream-reveal intermediate activations to free wires early                                                                                         | Easy       | 70 %       |
| 5.3                                                    | Parallelise independent matmuls with rayon thread-pool                                                                                             | Easy       | 80 %       |
| **6 Optional GPU Offload (Evaluator-side)**            |                                                                                                                                                    |            |            |
| 6.1                                                    | Prototype CUDA kernel for mod-2¹⁶ matmul; expose via pybind11/FFI to Rust                                                                          | Easy       | 60 %       |
| 6.2                                                    | Add `--device cuda` flag; fallback to CPU                                                                                                          | Easy       | 70 %       |
| 6.3                                                    | Benchmark GPU vs. CPU on sample prompt                                                                                                             | Easy       | 60 %       |
| **7 Automation & CI**                                  |                                                                                                                                                    |            |            |
| 7.1                                                    | GitHub Actions workflow: cache dependencies, run Python dump, execute Rust smoke test                                                              | Easy       | 85 %       |
| 7.2                                                    | Upload benchmark CSV + markdown as artifacts                                                                                                       | Easy       | 80 %       |
| 7.3                                                    | Introduce central config (`config.py` + `config.rs`) for flags: `--device`, `--num-runs`, `--seed`, `--csv`                                        | Easy       | 85 %       |
| 7.4                                                    | Add `tqdm`/conditional progress bars; works on localhost and CI logs                                                                               | Easy       | 90 %       |
| **8 Documentation & Reporting**                        |                                                                                                                                                    |            |            |
| 8.1                                                    | Update README with build, preprocessing, and run instructions                                                                                      | Easy       | 90 %       |
| 8.2                                                    | Ensure every public Rust function includes GC assumptions (Free XOR, Half Gates) in docstrings                                                     | Easy       | 85 %       |
| 8.3                                                    | Publish benchmarks under `docs/results/` with CSV artefacts                                                                                        | Easy       | 85 %       |

---

## Common Repeated Tasks

| ID     | Description                                                                                                                           | Difficulty | Confidence |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ---------- |
| README | Refresh `README.md`, CHANGELOG, and any affected docs after a feature lands                                                           | Easy       | 95 %       |
| SMOKE  | End-to-end smoke test: `plaintext_baseline.py --dump-embeddings` → Rust `garbler` + `evaluator`; assert logits agree within tolerance | Easy       | 90 %       |
| BENCH  | Run full benchmark harness (`--num-runs`, CSV, markdown summary) on default prompt                                                    | Easy       | 85 %       |
| FMT    | Apply formatters: `cargo fmt`, `rustfmt --check`, `black`, `ruff --fix`                                                               | Easy       | 95 %       |
| LINT   | Static analysis: `cargo clippy --all-targets -- -D warnings`, `ruff`, `mypy`                                                          | Easy       | 90 %       |
| TEST   | Unit tests & quick property tests for numerical gadgets (Rust) and quantisation helpers (Python)                                      | Easy       | 90 %       |
| CLEAN  | Purge build artifacts and generated `.npy` dumps to reclaim disk space                                                                | Easy       | 95 %       |
| CI     | Push or dry-run GitHub Actions workflow to ensure build & smoke tests pass                                                            | Easy       | 85 %       |

---

_Generated automatically; update in **Ask Mode** and re-commit as objectives evolve._
