use fancy_garbling::{CrtBundle, CrtGadgets, Fancy, FancyInput, HasModulus};
use ndarray::Array1;

pub mod memory;

/// Linear layer implementation for both plaintext and garbled circuit evaluation
pub struct LinearLayer {
    pub weights: Array1<i16>, // GPT-2 hidden size (768) weights, quantized to 16-bit
    pub bias: i16,            // Bias term, quantized to 16-bit
}

impl LinearLayer {
    /// Create a new linear layer with random weights for testing
    pub fn new_random(input_size: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Initialize with small random weights (simulating quantized GPT-2 weights)
        let weights = Array1::from_iter((0..input_size).map(|_| rng.gen_range(-100..100)));
        let bias = rng.gen_range(-50..50);

        Self { weights, bias }
    }

    /// Create a deterministic linear layer for testing
    pub fn new_test_layer(input_size: usize) -> Self {
        // Create deterministic weights and bias for consistent testing
        let weights = Array1::from_iter((0..input_size).map(|i| ((i % 5) + 1) as i16));
        let bias = 10;
        Self { weights, bias }
    }

    /// Evaluate the linear layer in plaintext
    pub fn eval_plaintext(&self, input: &Array1<i16>) -> i16 {
        assert_eq!(input.len(), self.weights.len());

        let mut result = self.bias as i32;
        for (inp, weight) in input.iter().zip(self.weights.iter()) {
            result += (*inp as i32) * (*weight as i32);
        }

        // Clamp to i16 range to simulate quantization
        result.clamp(i16::MIN as i32, i16::MAX as i32) as i16
    }

    /// Evaluate the linear layer using garbled circuits
    pub fn eval_garbled<W, F>(
        &self,
        fancy: &mut F,
        input: &[CrtBundle<W>],
        modulus: u128,
    ) -> std::result::Result<CrtBundle<W>, <F as Fancy>::Error>
    where
        W: Clone + HasModulus,
        F: Fancy<Item = W> + FancyInput<Item = W> + CrtGadgets,
    {
        assert_eq!(input.len(), self.weights.len());

        // Start with bias
        let mut result = fancy.crt_constant_bundle(to_mod_q(self.bias as i64, modulus), modulus)?;

        // Accumulate weight * input products
        for (inp, &weight) in input.iter().zip(self.weights.iter()) {
            let product = fancy.crt_cmul(inp, to_mod_q(weight as i64, modulus))?;
            result = fancy.crt_add(&result, &product)?;
        }

        Ok(result)
    }
}

/// Convert signed integer to modular representation
pub fn to_mod_q(x: i64, q: u128) -> u128 {
    if x < 0 {
        q - ((-x) as u128 % q)
    } else {
        x as u128 % q
    }
}

/// Convert modular representation back to signed integer
pub fn from_mod_q(x: u128, q: u128) -> i64 {
    let half_q = q / 2;
    if x <= half_q {
        x as i64
    } else {
        -((q - x) as i64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_layer_plaintext() {
        let layer = LinearLayer {
            weights: Array1::from(vec![1, 2, 3]),
            bias: 10,
        };

        let input = Array1::from(vec![4, 5, 6]);
        let result = layer.eval_plaintext(&input);

        // 1*4 + 2*5 + 3*6 + 10 = 4 + 10 + 18 + 10 = 42
        assert_eq!(result, 42);
    }

    #[test]
    fn test_modular_conversion() {
        // Use a square-free 16-bit modulus accepted by fancy-garbling
        let q: u128 = fancy_garbling::util::modulus_with_width(16);

        // Test positive numbers
        assert_eq!(to_mod_q(100, q), 100);
        assert_eq!(from_mod_q(100, q), 100);

        // Test negative numbers
        assert_eq!(to_mod_q(-100, q), q - 100);
        assert_eq!(from_mod_q(q - 100, q), -100);

        // Test round trip
        for x in [-1000, -1, 0, 1, 1000] {
            let modular = to_mod_q(x, q);
            let recovered = from_mod_q(modular, q);
            assert_eq!(x, recovered);
        }
    }

    #[test]
    fn test_linear_layer_garbled_matches_plaintext() {
        use fancy_garbling::dummy::Dummy;
        use fancy_garbling::FancyReveal;

        // Use small layer size for the test
        let input_size = 5;
        let layer = LinearLayer::new_test_layer(input_size);

        // Build a deterministic input vector
        let input_vals: Vec<i16> = (0..input_size as i16).map(|i| i * 3 + 1).collect();
        let input = Array1::from(input_vals.clone());

        // Plaintext reference
        let expected = layer.eval_plaintext(&input);

        // Use a square-free 16-bit modulus accepted by fancy-garbling
        let q: u128 = fancy_garbling::util::modulus_with_width(16);

        // Build Dummy fancy object and encode inputs as CRT bundles
        let mut dummy = Dummy::new();
        let encoded_input: Vec<_> = input_vals
            .iter()
            .map(|&v| dummy.crt_encode(to_mod_q(v as i64, q), q).unwrap())
            .collect();

        // Evaluate in GC and reveal
        let result_bundle = layer
            .eval_garbled(&mut dummy, &encoded_input, q)
            .expect("garbled eval failed");
        let revealed_mod = dummy.crt_reveal(&result_bundle).unwrap();
        let actual = from_mod_q(revealed_mod, q) as i16;

        assert_eq!(expected, actual);
    }
}
