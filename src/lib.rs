use fancy_garbling::{CrtBundle, CrtGadgets, Fancy, FancyInput, HasModulus};
use ndarray::{Array1, Array2};
use std::path::Path;

pub mod csv_writer;
pub mod memory;

/// Linear layer implementation for both plaintext and garbled circuit evaluation
/// Supports both 1D vector weights (legacy) and 2D matrix weights (transformer layers)
#[derive(Debug)]
pub struct LinearLayer {
    pub weights: Array2<i16>, // Weight matrix: [input_size, output_size], quantized to 16-bit
    pub bias: Option<Array1<i16>>, // Optional bias vector, quantized to 16-bit
}

impl LinearLayer {
    /// Create a new linear layer with random weights for testing
    pub fn new_random(input_size: usize, output_size: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Initialize with small random weights (simulating quantized GPT-2 weights)
        let weights =
            Array2::from_shape_fn((input_size, output_size), |_| rng.gen_range(-100..100));
        let bias = Some(Array1::from_shape_fn(output_size, |_| {
            rng.gen_range(-50..50)
        }));

        Self { weights, bias }
    }

    /// Create a deterministic linear layer for testing (backward compatibility)
    pub fn new_test_layer(input_size: usize) -> Self {
        // Create deterministic weights as 1D -> 1D transformation for legacy tests
        let weights = Array2::from_shape_fn((input_size, 1), |(i, _)| ((i % 5) + 1) as i16);
        let bias = Some(Array1::from(vec![10]));
        Self { weights, bias }
    }

    /// Load weights from a .npy file
    ///
    /// # Arguments
    /// * `weight_path` - Path to the .npy file containing the weight matrix
    /// * `bias_path` - Optional path to the .npy file containing bias vector
    ///
    /// # Garbling Assumptions
    /// * Uses Free XOR and Half Gates protocols in fancy-garbling
    /// * Weights are quantized to Q8.8 fixed-point (16-bit signed integers)
    /// * Input modulus must be compatible with fancy-garbling CRT requirements
    ///
    /// # Errors
    /// * Returns error if weight file doesn't exist or can't be read
    /// * Returns error if bias file is specified but doesn't exist
    /// * Returns error if bias dimension doesn't match weight output dimension
    /// * Returns error if weight matrix is not 2D
    pub fn from_npy_files<P: AsRef<Path>>(
        weight_path: P,
        bias_path: Option<P>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let weight_path = weight_path.as_ref();

        // Check if weight file exists
        if !weight_path.exists() {
            return Err(format!(
                "Weight file not found: {}. Please ensure the file exists and run the Python script to generate weights: \
                 python plaintext_baseline.py --dump-block-weights --block-id 0 --out-dir weights/",
                weight_path.display()
            ).into());
        }

        // Load weight matrix
        let weights: Array2<i16> = ndarray_npy::read_npy(weight_path)
            .map_err(|e| format!(
                "Failed to read weight file '{}': {}. The file may be corrupted or in wrong format. \
                 Expected 2D array of int16 values.",
                weight_path.display(), e
            ))?;

        // Validate that it's actually a 2D matrix
        if weights.ndim() != 2 {
            return Err(format!(
                "Weight file '{}' contains {}-dimensional array, expected 2D matrix. \
                 Re-run the Python weight extraction script to fix this.",
                weight_path.display(),
                weights.ndim()
            )
            .into());
        }

        // Load optional bias vector
        let bias = if let Some(bias_path) = bias_path {
            let bias_path = bias_path.as_ref();

            if !bias_path.exists() {
                return Err(format!(
                    "Bias file not found: {}. Either provide a valid bias file or use from_npy_weights_only() instead.",
                    bias_path.display()
                ).into());
            }

            let bias_array: Array1<i16> = ndarray_npy::read_npy(bias_path)
                .map_err(|e| format!(
                    "Failed to read bias file '{}': {}. The file may be corrupted or in wrong format. \
                     Expected 1D array of int16 values.",
                    bias_path.display(), e
                ))?;

            Some(bias_array)
        } else {
            None
        };

        // Validate dimensions
        if let Some(ref b) = bias {
            if b.len() != weights.ncols() {
                return Err(format!(
                    "Bias dimension {} doesn't match weight output dimension {}. \
                     The bias vector must have the same length as the number of output neurons. \
                     Check that you're loading compatible weight and bias files.",
                    b.len(),
                    weights.ncols()
                )
                .into());
            }
        }

        Ok(Self { weights, bias })
    }

    /// Create a bias-free linear layer from weights file (for Q/K/V projections)
    ///
    /// # Garbling Assumptions  
    /// * Designed for transformer attention projections which typically don't use bias
    /// * Compatible with Half Gates garbling in fancy-garbling
    ///
    /// # Errors
    /// * Returns error if weight file doesn't exist or can't be read
    /// * Returns error if file doesn't contain a 2D matrix
    pub fn from_npy_weights_only<P: AsRef<Path>>(
        weight_path: P,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let weight_path = weight_path.as_ref();

        // Check if file exists
        if !weight_path.exists() {
            return Err(format!(
                "Weight file not found: {}. Please ensure the file exists and run the Python script to generate weights: \
                 python plaintext_baseline.py --dump-block-weights --block-id 0 --out-dir weights/",
                weight_path.display()
            ).into());
        }

        let weights: Array2<i16> = ndarray_npy::read_npy(weight_path)
            .map_err(|e| format!(
                "Failed to read weight file '{}': {}. The file may be corrupted or in wrong format. \
                 Expected 2D array of int16 values.",
                weight_path.display(), e
            ))?;

        // Validate that it's actually a 2D matrix
        if weights.ndim() != 2 {
            return Err(format!(
                "Weight file '{}' contains {}-dimensional array, expected 2D matrix. \
                 Re-run the Python weight extraction script to fix this.",
                weight_path.display(),
                weights.ndim()
            )
            .into());
        }

        Ok(Self {
            weights,
            bias: None,
        })
    }

    /// Get input dimension
    pub fn input_size(&self) -> usize {
        self.weights.nrows()
    }

    /// Get output dimension  
    pub fn output_size(&self) -> usize {
        self.weights.ncols()
    }

    /// Evaluate the linear layer in plaintext (supports matrix multiplication)
    pub fn eval_plaintext(&self, input: &Array1<i16>) -> Array1<i16> {
        assert_eq!(input.len(), self.input_size());

        // Matrix-vector multiplication: W^T * x
        let mut result = Array1::zeros(self.output_size());

        for (i, &input_val) in input.iter().enumerate() {
            for j in 0..self.output_size() {
                result[j] += (input_val as i32 * self.weights[[i, j]] as i32) as i16;
            }
        }

        // Add bias if present
        if let Some(ref bias) = self.bias {
            for (out, &bias_val) in result.iter_mut().zip(bias.iter()) {
                *out =
                    (*out as i32 + bias_val as i32).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            }
        }

        result
    }

    /// Legacy method for backward compatibility (1D output)
    pub fn eval_plaintext_scalar(&self, input: &Array1<i16>) -> i16 {
        let result = self.eval_plaintext(input);
        result[0] // Return first (and expected only) element
    }

    /// Evaluate the linear layer using garbled circuits (supports matrix multiplication)
    ///
    /// # Garbling Assumptions
    /// * Uses CRT representation compatible with fancy-garbling
    /// * Leverages Free XOR optimization for XOR gates
    /// * Multiplication uses Half Gates protocol
    pub fn eval_garbled<W, F>(
        &self,
        fancy: &mut F,
        input: &[CrtBundle<W>],
        modulus: u128,
    ) -> std::result::Result<Vec<CrtBundle<W>>, <F as Fancy>::Error>
    where
        W: Clone + HasModulus,
        F: Fancy<Item = W> + FancyInput<Item = W> + CrtGadgets,
    {
        assert_eq!(input.len(), self.input_size());

        let mut outputs = Vec::with_capacity(self.output_size());

        // For each output dimension
        for j in 0..self.output_size() {
            // Start with bias if present, otherwise zero
            let mut result = if let Some(ref bias) = self.bias {
                fancy.crt_constant_bundle(to_mod_q(bias[j] as i64, modulus), modulus)?
            } else {
                fancy.crt_constant_bundle(0, modulus)?
            };

            // Accumulate weight * input products for this output
            for (i, inp) in input.iter().enumerate() {
                let weight_val = self.weights[[i, j]];
                let product = fancy.crt_cmul(inp, to_mod_q(weight_val as i64, modulus))?;
                result = fancy.crt_add(&result, &product)?;
            }

            outputs.push(result);
        }

        Ok(outputs)
    }

    /// Legacy method for backward compatibility (single output)
    pub fn eval_garbled_scalar<W, F>(
        &self,
        fancy: &mut F,
        input: &[CrtBundle<W>],
        modulus: u128,
    ) -> std::result::Result<CrtBundle<W>, <F as Fancy>::Error>
    where
        W: Clone + HasModulus,
        F: Fancy<Item = W> + FancyInput<Item = W> + CrtGadgets,
    {
        let results = self.eval_garbled(fancy, input, modulus)?;
        Ok(results.into_iter().next().unwrap())
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
            weights: Array2::from_shape_fn((3, 1), |(i, _)| (i + 1) as i16),
            bias: Some(Array1::from(vec![10])),
        };

        let input = Array1::from(vec![1, 2, 3]);
        let result = layer.eval_plaintext(&input);

        // 1*1 + 2*2 + 3*3 + 10 = 1 + 4 + 9 + 10 = 24
        assert_eq!(result, Array1::from(vec![24]));
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
        let expected = layer.eval_plaintext_scalar(&input);

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
            .eval_garbled_scalar(&mut dummy, &encoded_input, q)
            .expect("garbled eval failed");
        let revealed_mod = dummy.crt_reveal(&result_bundle).unwrap();
        let actual = from_mod_q(revealed_mod, q) as i16;

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_npy_weight_loading() {
        // This test will only run if the weight files exist
        if std::path::Path::new("weights/block_0_q.npy").exists() {
            println!("Testing .npy weight loading...");

            // Test loading Q weights (should be 768x768)
            let q_layer = LinearLayer::from_npy_weights_only("weights/block_0_q.npy")
                .expect("Failed to load Q weights");
            assert_eq!(q_layer.input_size(), 768);
            assert_eq!(q_layer.output_size(), 768);
            assert!(q_layer.bias.is_none());

            // Test loading FC1 weights (should be 768x3072)
            let fc1_layer = LinearLayer::from_npy_weights_only("weights/block_0_fc1.npy")
                .expect("Failed to load FC1 weights");
            assert_eq!(fc1_layer.input_size(), 768);
            assert_eq!(fc1_layer.output_size(), 3072);
            assert!(fc1_layer.bias.is_none());

            println!("✅ All weight loading tests passed!");
        } else {
            println!("⏩ Skipping .npy weight loading test - weight files not found");
        }
    }

    #[test]
    fn test_npy_error_handling() {
        // Test loading non-existent file
        let result = LinearLayer::from_npy_weights_only("non_existent_file.npy");
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Weight file not found"));
        assert!(error_msg.contains("python plaintext_baseline.py"));

        // Test loading with non-existent bias file
        if std::path::Path::new("weights/block_0_q.npy").exists() {
            let result =
                LinearLayer::from_npy_files("weights/block_0_q.npy", Some("non_existent_bias.npy"));
            assert!(result.is_err());
            let error_msg = result.unwrap_err().to_string();
            assert!(error_msg.contains("Bias file not found"));
        }

        println!("✅ Error handling tests passed!");
    }
}
