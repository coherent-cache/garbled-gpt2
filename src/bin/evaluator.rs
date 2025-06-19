use anyhow::{Context, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use clap::Parser;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

use fancy_garbling::twopac::semihonest::Evaluator;
use fancy_garbling::util as numbers;
use fancy_garbling::{AllWire, FancyInput, FancyReveal};
use ocelot::ot::AlszReceiver;
use scuttlebutt::{AesRng, Channel, TrackChannel};

use garbled_gpt2::{
    csv_writer::{BenchmarkResult, CsvWriter},
    from_mod_q,
    memory::MemoryTracker,
    LinearLayer,
};

/// Evaluator side – connects to the garbler, waits for PING, replies with PONG.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Address to connect to, e.g. 127.0.0.1:9000
    #[arg(long, default_value = "127.0.0.1:9000")]
    connect: String,

    /// Maximum connection retries
    #[arg(long, default_value_t = 10)]
    retries: u32,

    /// Delay between retries in milliseconds
    #[arg(long, default_value_t = 500)]
    retry_delay_ms: u64,

    /// Path to CSV file for benchmark results
    #[arg(long)]
    csv: Option<String>,

    /// Number of benchmark runs
    #[arg(long, default_value_t = 1)]
    num_runs: u32,

    /// Random seed for reproducible runs
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Set up CSV writer if requested
    let csv_writer = if let Some(csv_path) = &args.csv {
        // Generate unique filename for evaluator to avoid corruption
        let evaluator_csv_path = if csv_path.ends_with(".csv") {
            csv_path.replace(".csv", "_evaluator.csv")
        } else {
            format!("{}_evaluator.csv", csv_path)
        };
        let writer = CsvWriter::new(evaluator_csv_path);
        writer
            .write_header()
            .with_context(|| "Failed to write CSV header")?;
        Some(writer)
    } else {
        None
    };

    // Run for the specified number of runs
    for run_id in 1..=args.num_runs {
        println!("[evaluator] === Run {}/{} ===", run_id, args.num_runs);

        let stream = connect_with_retry(&args.connect, args.retries, args.retry_delay_ms)?;
        println!("[evaluator] connected to {} (run {})", args.connect, run_id);

        // First do the handshake
        handshake_as_evaluator(stream.try_clone()?)?;
        println!("[evaluator] handshake completed (run {})", run_id);

        // Now participate in the linear layer demo
        participate_in_linear_layer_demo(stream, &csv_writer, run_id, args.seed)?;
    }

    println!("[evaluator] ✅ All {} runs completed", args.num_runs);

    Ok(())
}

fn connect_with_retry(addr: &str, retries: u32, delay_ms: u64) -> Result<TcpStream> {
    let mut attempt = 0;
    loop {
        match TcpStream::connect(addr) {
            Ok(stream) => return Ok(stream),
            Err(err) if attempt < retries => {
                println!(
                    "[evaluator] connection attempt {} failed: {}",
                    attempt + 1,
                    err
                );
                std::thread::sleep(Duration::from_millis(delay_ms));
                attempt += 1;
            }
            Err(err) => {
                return Err(err).with_context(|| {
                    format!("failed to connect to {} after {} retries", addr, retries)
                })
            }
        }
    }
}

fn handshake_as_evaluator(mut stream: TcpStream) -> Result<()> {
    println!("[evaluator] waiting for PING handshake from garbler …");

    // Wait for PING
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    if line.trim_end() != "PING" {
        anyhow::bail!("unexpected handshake message: {}", line.trim_end());
    }
    println!("[evaluator] received PING");

    // Send PONG
    stream.write_all(b"PONG\n")?;
    stream.flush()?;
    Ok(())
}

fn participate_in_linear_layer_demo(
    mut stream: TcpStream,
    csv_writer: &Option<CsvWriter>,
    run_id: u32,
    seed: u64,
) -> Result<()> {
    println!("[evaluator] participating in linear layer demo");
    let wall_start = Instant::now();

    // Initialize memory tracking
    let memory_tracker = MemoryTracker::new()
        .map_err(|e| anyhow::anyhow!("Failed to initialize memory tracker: {}", e))?;
    let initial_mem_mb = memory_tracker
        .current_allocated_mb()
        .map_err(|e| anyhow::anyhow!("Failed to read initial memory: {}", e))?;
    println!("[evaluator] initial memory: {:.2} MB", initial_mem_mb);

    // Read the input length (u32 little-endian) sent by the garbler
    let input_len = stream.read_u32::<LittleEndian>()? as usize;
    println!("[evaluator] expecting {} input elements", input_len);

    // Create the *same* deterministic linear layer as the garbler
    let layer = LinearLayer::new_test_layer(input_len);
    let modulus = numbers::modulus_with_width(16); // 16-bit modulus

    println!("[evaluator] using modulus: {}", modulus);

    // Create channel and evaluator
    let channel = Channel::new(stream.try_clone()?, stream);
    let track_channel = TrackChannel::new(channel);
    let rng = AesRng::new();

    let eval_start = Instant::now();
    let mut evaluator =
        Evaluator::<_, AesRng, AlszReceiver, AllWire>::new(track_channel.clone(), rng)?;

    // Receive the encoded inputs from the garbler
    let input_bundles = evaluator.crt_receive_many(input_len, modulus)?;
    println!(
        "[evaluator] received {} encoded input values",
        input_bundles.len()
    );

    // Evaluate the garbled circuit. Both parties must call this.
    let gc_eval_start = Instant::now();
    let gc_result_bundle = layer
        .eval_garbled_scalar(&mut evaluator, &input_bundles, modulus)
        .map_err(|e| anyhow::anyhow!("Garbled circuit evaluation failed: {:?}", e))?;
    let gc_eval_time = gc_eval_start.elapsed();

    // Reveal the result to get the plaintext value
    let reveal_start = Instant::now();
    let modular_result_vec = evaluator.reveal_bundle(&gc_result_bundle)?;
    let primes = numbers::factor(modulus);
    let modular_result = numbers::crt_inv(&modular_result_vec, &primes);
    let result = from_mod_q(modular_result, modulus);
    let reveal_time = reveal_start.elapsed();

    let eval_time = eval_start.elapsed();
    let wall_time = wall_start.elapsed();

    println!("[evaluator] ✅ Milestone 3 validated!");
    println!("[evaluator] Garbled circuit result (revealed): {}", result);
    println!("[evaluator] GC participation took {:?}", gc_eval_time);

    // Emit all timing measurements, byte counters, and memory usage (Steps 2.1 & 2.2)
    let peak_mem_mb = memory_tracker
        .peak_allocated_mb()
        .map_err(|e| anyhow::anyhow!("Failed to read peak memory: {}", e))?;
    let final_mem_mb = memory_tracker
        .current_allocated_mb()
        .map_err(|e| anyhow::anyhow!("Failed to read final memory: {}", e))?;

    println!(
        "[evaluator] wall_time={:?} eval_time={:?} gc_eval_time={:?} reveal_time={:?}",
        wall_time, eval_time, gc_eval_time, reveal_time
    );
    println!(
        "[evaluator] bytes_sent={:.0} bytes_recv={:.0} total_bytes={:.0}",
        track_channel.kilobytes_written() * 1024.0,
        track_channel.kilobytes_read() * 1024.0,
        track_channel.total_kilobytes() * 1024.0
    );
    println!(
        "[evaluator] peak_mem={:.2} MB final_mem={:.2} MB mem_delta={:.2} MB",
        peak_mem_mb,
        final_mem_mb,
        final_mem_mb - initial_mem_mb
    );

    // Write results to CSV if requested
    if let Some(writer) = csv_writer {
        let result = BenchmarkResult::new_evaluator(
            wall_time,
            eval_time,
            gc_eval_time,
            reveal_time,
            track_channel.kilobytes_written() * 1024.0,
            track_channel.kilobytes_read() * 1024.0,
            initial_mem_mb,
            peak_mem_mb,
            final_mem_mb,
            input_len,
            run_id,
            seed,
        );

        writer
            .write_result(&result)
            .with_context(|| "Failed to write benchmark result to CSV")?;
        println!("[evaluator] Benchmark result written to CSV");
    }

    Ok(())
}
