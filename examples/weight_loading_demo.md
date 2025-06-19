# LinearLayer Weight Loading Demo

This document demonstrates how to use the new `.npy` weight loading functionality in the `LinearLayer` struct.

## Prerequisites

1. Generate transformer block weights using the Python script:

```bash
python plaintext_baseline.py --dump-block-weights --block-id 0 --out-dir weights/
```

## Loading Weight Files

### 1. Bias-free Linear Layer (Q/K/V projections)

```rust
use garbled_gpt2::LinearLayer;

// Load Query projection weights for transformer block 0
let q_layer = LinearLayer::from_npy_weights_only("weights/block_0_q.npy")?;
println!("Q layer: {} x {}", q_layer.input_size(), q_layer.output_size());

// Load Key projection weights
let k_layer = LinearLayer::from_npy_weights_only("weights/block_0_k.npy")?;

// Load Value projection weights
let v_layer = LinearLayer::from_npy_weights_only("weights/block_0_v.npy")?;
```

### 2. Linear Layer with Optional Bias

```rust
// Load weights and bias separately (if bias file exists)
let layer_with_bias = LinearLayer::from_npy_files(
    "weights/block_0_fc1.npy",
    Some("weights/block_0_fc1_bias.npy")  // Optional bias file
)?;

// Load weights only (no bias)
let layer_no_bias = LinearLayer::from_npy_files(
    "weights/block_0_fc1.npy",
    None
)?;
```

## Usage Example

```rust
use ndarray::Array1;

// Load FC1 weights (768 -> 3072 transformation)
let fc1_layer = LinearLayer::from_npy_weights_only("weights/block_0_fc1.npy")?;

// Create test input (768 elements)
let input = Array1::from(vec![100i16; 768]);

// Plaintext evaluation
let output = fc1_layer.eval_plaintext(&input);
println!("Output shape: {}", output.len()); // Should be 3072

// Garbled circuit evaluation (example with dummy garbler)
use fancy_garbling::dummy::Dummy;
let mut dummy = Dummy::new();
let modulus = fancy_garbling::util::modulus_with_width(16);

let encoded_input: Vec<_> = input.iter()
    .map(|&v| dummy.crt_encode(garbled_gpt2::to_mod_q(v as i64, modulus), modulus).unwrap())
    .collect();

let gc_output = fc1_layer.eval_garbled(&mut dummy, &encoded_input, modulus)?;
println!("GC output bundles: {}", gc_output.len());
```

## Garbling Assumptions

The weight loading functionality includes proper documentation about garbling assumptions:

- **Free XOR**: XOR gates are evaluated for free using wire label relationships
- **Half Gates**: Multiplication operations use the Half Gates protocol
- **CRT Representation**: All values use Chinese Remainder Theorem representation
- **Quantization**: Weights are Q8.8 fixed-point (16-bit signed integers)

## Available Weight Files

When you run the Python weight extraction, the following files are created:

- `block_{id}_q.npy` - Query projection (768×768)
- `block_{id}_k.npy` - Key projection (768×768)
- `block_{id}_v.npy` - Value projection (768×768)
- `block_{id}_o.npy` - Output projection (768×768)
- `block_{id}_fc1.npy` - First feed-forward layer (768×3072)
- `block_{id}_fc2.npy` - Second feed-forward layer (3072×768)

All matrices are quantized to 16-bit signed integers using Q8.8 fixed-point format.
