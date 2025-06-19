use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::time::Duration;

/// Benchmark results structure for CSV output
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub role: String,                // "garbler" or "evaluator"
    pub wall_time_ms: f64,           // Wall clock time in milliseconds
    pub garble_time_ms: Option<f64>, // Garble time in ms (garbler only)
    pub eval_time_ms: f64,           // Evaluation time in ms
    pub reveal_time_ms: f64,         // Reveal time in ms
    pub bytes_sent: f64,             // Bytes sent to network
    pub bytes_recv: f64,             // Bytes received from network
    pub total_bytes: f64,            // Total communication bytes
    pub initial_mem_mb: f64,         // Initial memory in MB
    pub peak_mem_mb: f64,            // Peak memory in MB
    pub final_mem_mb: f64,           // Final memory in MB
    pub mem_delta_mb: f64,           // Memory delta in MB
    pub input_size: usize,           // Size of input vector
    pub run_id: u32,                 // Run number
    pub seed: u64,                   // Random seed
}

impl BenchmarkResult {
    /// Create a new benchmark result for garbler
    pub fn new_garbler(
        wall_time: Duration,
        garble_time: Duration,
        eval_time: Duration,
        reveal_time: Duration,
        bytes_sent: f64,
        bytes_recv: f64,
        initial_mem_mb: f64,
        peak_mem_mb: f64,
        final_mem_mb: f64,
        input_size: usize,
        run_id: u32,
        seed: u64,
    ) -> Self {
        Self {
            role: "garbler".to_string(),
            wall_time_ms: wall_time.as_secs_f64() * 1000.0,
            garble_time_ms: Some(garble_time.as_secs_f64() * 1000.0),
            eval_time_ms: eval_time.as_secs_f64() * 1000.0,
            reveal_time_ms: reveal_time.as_secs_f64() * 1000.0,
            bytes_sent,
            bytes_recv,
            total_bytes: bytes_sent + bytes_recv,
            initial_mem_mb,
            peak_mem_mb,
            final_mem_mb,
            mem_delta_mb: final_mem_mb - initial_mem_mb,
            input_size,
            run_id,
            seed,
        }
    }

    /// Create a new benchmark result for evaluator
    pub fn new_evaluator(
        wall_time: Duration,
        _eval_time: Duration,
        gc_eval_time: Duration,
        reveal_time: Duration,
        bytes_sent: f64,
        bytes_recv: f64,
        initial_mem_mb: f64,
        peak_mem_mb: f64,
        final_mem_mb: f64,
        input_size: usize,
        run_id: u32,
        seed: u64,
    ) -> Self {
        Self {
            role: "evaluator".to_string(),
            wall_time_ms: wall_time.as_secs_f64() * 1000.0,
            garble_time_ms: None, // Not applicable for evaluator
            eval_time_ms: gc_eval_time.as_secs_f64() * 1000.0,
            reveal_time_ms: reveal_time.as_secs_f64() * 1000.0,
            bytes_sent,
            bytes_recv,
            total_bytes: bytes_sent + bytes_recv,
            initial_mem_mb,
            peak_mem_mb,
            final_mem_mb,
            mem_delta_mb: final_mem_mb - initial_mem_mb,
            input_size,
            run_id,
            seed,
        }
    }
}

/// CSV writer for benchmark results
pub struct CsvWriter {
    path: String,
}

impl CsvWriter {
    /// Create a new CSV writer
    pub fn new(path: String) -> Self {
        Self { path }
    }

    /// Write CSV header if file doesn't exist
    pub fn write_header(&self) -> Result<(), std::io::Error> {
        if !Path::new(&self.path).exists() {
            let mut file = OpenOptions::new()
                .create(true)
                .write(true)
                .open(&self.path)?;

            writeln!(file, "role,wall_time_ms,garble_time_ms,eval_time_ms,reveal_time_ms,bytes_sent,bytes_recv,total_bytes,initial_mem_mb,peak_mem_mb,final_mem_mb,mem_delta_mb,input_size,run_id,seed")?;
        }
        Ok(())
    }

    /// Append a benchmark result to the CSV file
    pub fn write_result(&self, result: &BenchmarkResult) -> Result<(), std::io::Error> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;

        let garble_time_str = match result.garble_time_ms {
            Some(time) => format!("{:.3}", time),
            None => "".to_string(),
        };

        writeln!(
            file,
            "{},{:.3},{},{:.3},{:.3},{:.0},{:.0},{:.0},{:.2},{:.2},{:.2},{:.2},{},{},{}",
            result.role,
            result.wall_time_ms,
            garble_time_str,
            result.eval_time_ms,
            result.reveal_time_ms,
            result.bytes_sent,
            result.bytes_recv,
            result.total_bytes,
            result.initial_mem_mb,
            result.peak_mem_mb,
            result.final_mem_mb,
            result.mem_delta_mb,
            result.input_size,
            result.run_id,
            result.seed
        )?;

        Ok(())
    }
}
