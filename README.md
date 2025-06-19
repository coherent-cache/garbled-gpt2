# garbled-gpt2

A GPT-2 inference implementation using garbled circuits via the [fancy-garbling](https://galoisinc.github.io/swanky/fancy_garbling/) library. This project demonstrates secure two-party computation for neural network inference with public model weights and private user inputs.

## 📋 Project Status

**Current Milestone: M4** - Dynamic embedding support with real GPT-2 weights

- ✅ **M1**: Garbler/Evaluator socket communication
- ✅ **M2**: Python baseline with quantized weight extraction
- ✅ **M3**: Single linear layer garbled circuit demo
- ✅ **3A.1**: Extended LinearLayer with .npy weight loading
- 🔄 **M4**: In progress - Dynamic embeddings and real model weights

See [PLAN.md](PLAN.md) for detailed milestones and [TASKLIST.md](TASKLIST.md) for specific implementation steps.

## 🚀 Quick Start

### Prerequisites

- **Rust 1.74+** (nightly not required)
- **Python 3.8+** with PyTorch and transformers
- **Git** for cloning dependencies

### Setup

1. **Clone and build**:

   ```bash
   git clone <repository-url>
   cd garbled-gpt2
   cargo build --release
   ```

2. **Install Python dependencies**:

   ```bash
   pip install torch transformers numpy ndarray
   ```

3. **Generate model weights**:

   ```bash
   # Extract GPT-2 transformer block weights
   python plaintext_baseline.py --dump-block-weights --block-id 0 --out-dir weights/

   # Generate embeddings for a test prompt
   python plaintext_baseline.py --dump-embeddings --text "The quick brown fox" --out embedding.npy
   ```

### Running the Demo

**Terminal 1 (Garbler)**:

```bash
cargo run --release --bin garbler -- --listen 0.0.0.0:9000 --embeddings embedding.npy
```

**Terminal 2 (Evaluator)** (can be on different machine):

```bash
cargo run --release --bin evaluator -- --connect 127.0.0.1:9000
```

## 🏗️ Architecture

### Two-Party Computation

- **Garbler**: Holds the input embeddings, creates the garbled circuit
- **Evaluator**: Participates in circuit evaluation, learns the final result
- **Model weights**: Public to both parties (loaded from .npy files)
- **Protocol**: Semi-honest security using fancy-garbling's Half Gates + Free XOR

### Components

```
Python (plaintext_baseline.py)     Rust (garbled circuits)
├── GPT-2 model loading            ├── garbler binary
├── Quantization (Q8.8)            ├── evaluator binary
├── Weight extraction               ├── LinearLayer struct
└── Embedding generation           └── Garbled circuit evaluation
```

## 📁 Project Structure

```
garbled-gpt2/
├── src/
│   ├── lib.rs              # LinearLayer implementation
│   ├── csv_writer.rs       # Benchmark result logging
│   ├── memory.rs           # Memory usage tracking
│   └── bin/
│       ├── garbler.rs      # Garbler binary
│       └── evaluator.rs    # Evaluator binary
├── examples/
│   └── weight_loading_demo.md  # Weight loading usage examples
├── scripts/
│   └── summarise.py        # Benchmark result analysis
├── plaintext_baseline.py   # PyTorch baseline + weight extraction
├── PLAN.md                 # Project roadmap
└── TASKLIST.md            # Detailed task breakdown
```

## 🔧 Usage Examples

### Weight Extraction

```bash
# Extract all transformer block weights (Q, K, V, O, FC1, FC2)
python plaintext_baseline.py --dump-block-weights --block-id 0 --out-dir weights/

# Extract embeddings for any prompt
python plaintext_baseline.py --dump-embeddings --text "Hello world" --out hello.npy

# Run plaintext baseline for comparison
python plaintext_baseline.py --text "The quick brown fox"
```

### LinearLayer API

```rust
use garbled_gpt2::LinearLayer;

// Load transformer weights
let q_layer = LinearLayer::from_npy_weights_only("weights/block_0_q.npy")?;
let fc1_layer = LinearLayer::from_npy_weights_only("weights/block_0_fc1.npy")?;

// Plaintext evaluation
let input = Array1::from(vec![100i16; 768]);
let output = q_layer.eval_plaintext(&input);

// Garbled circuit evaluation
let gc_output = q_layer.eval_garbled(&mut garbler, &input_bundles, modulus)?;
```

### Benchmarking

```bash
# Run multiple benchmarks with CSV output
cargo run --release --bin garbler -- --num-runs 5 --csv benchmark.csv
cargo run --release --bin evaluator -- --num-runs 5 --csv benchmark.csv

# Analyze results
python scripts/summarise.py benchmark_garbler.csv benchmark_evaluator.csv
```

## 📊 Performance

Current single linear layer demo performance (M3 baseline):

- **Input size**: 768 elements (GPT-2 hidden dimension)
- **Garbling time**: ~10-50ms depending on system
- **Communication**: ~1-5MB per evaluation
- **Memory usage**: ~50-200MB peak

## 🔮 Roadmap

**Immediate (M4/M5)**:

- [ ] Load real GPT-2 weights in garbled circuits
- [ ] Complete benchmark harness with memory tracking
- [ ] Dynamic embedding length support

**Short-term (3A-3E)**:

- [ ] Q/K/V attention projections
- [ ] Polynomial softmax approximation
- [ ] Feed-forward network layers
- [ ] Single transformer block integration

**Long-term (4-8)**:

- [ ] Full 12-block GPT-2 stack
- [ ] Performance optimizations
- [ ] Optional GPU acceleration
- [ ] CI/CD automation

## 🔒 Security Properties

- **Semi-honest security**: Protects against honest-but-curious adversaries
- **Input privacy**: User embeddings remain hidden from garbler
- **Output privacy**: Only evaluator learns the final result
- **Public weights**: Model parameters are known to both parties
- **No oblivious transfer**: Required since weights are public

## 🛠️ Development

### Running Tests

```bash
cargo test                    # All tests
cargo test test_npy          # Weight loading tests
cargo test -- --nocapture   # With output
```

### Code Organization

- **Garbling assumptions** documented in all public APIs
- **Error handling** with clear resolution suggestions
- **Backward compatibility** maintained for existing demos
- **Memory tracking** integrated for performance analysis

## 📚 References

- [fancy-garbling documentation](https://galoisinc.github.io/swanky/fancy_garbling/)
- [Swanky cryptographic library](https://github.com/GaloisInc/swanky)
- [GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Half Gates garbling](https://eprint.iacr.org/2014/756.pdf)

---

**Current Tasks**: See [TASKLIST.md](TASKLIST.md) for milestones and [examples/](examples/) for usage examples.
