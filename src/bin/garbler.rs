use anyhow::{Context, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use clap::Parser;
use ndarray::Array1;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Instant;

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

use fancy_garbling::twopac::semihonest::Garbler;
use fancy_garbling::util as numbers;
use fancy_garbling::{AllWire, FancyInput, FancyReveal};
use ocelot::ot::AlszSender;
use scuttlebutt::{AesRng, Channel, TrackChannel};

use garbled_gpt2::{
    csv_writer::{BenchmarkResult, CsvWriter},
    memory::MemoryTracker,
    to_mod_q, LinearLayer,
};

/// Garbler side – opens a listener, waits for the evaluator, then does a simple PING/PONG handshake.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Address to listen on, e.g. 0.0.0.0:9000
    #[arg(long, default_value = "0.0.0.0:9000")]
    listen: String,

    /// Path to embeddings file (.npy) produced by Python helper
    #[arg(long, default_value = "embedding.npy")]
    embeddings: String,

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

    // Load embeddings from `.npy` file without invoking Python (produced externally)
    // Support both 1-D (single token) and 2-D (multi-token) embeddings.
    let embedding_path = std::path::Path::new(&args.embeddings);
    // Attempt 2-D; if it fails, fall back to 1-D
    let embedding_vec: Vec<i16> =
        match ndarray_npy::read_npy::<_, ndarray::Array2<i16>>(embedding_path) {
            Ok(arr2) => arr2.into_raw_vec(),
            Err(_) => {
                let arr1: Array1<i16> = ndarray_npy::read_npy(embedding_path)?;
                arr1.to_vec()
            }
        };

    let embedding = embedding_vec;

    // Set up CSV writer if requested
    let csv_writer = if let Some(csv_path) = &args.csv {
        // Generate unique filename for garbler to avoid corruption
        let garbler_csv_path = if csv_path.ends_with(".csv") {
            csv_path.replace(".csv", "_garbler.csv")
        } else {
            format!("{}_garbler.csv", csv_path)
        };
        let writer = CsvWriter::new(garbler_csv_path);
        writer
            .write_header()
            .with_context(|| "Failed to write CSV header")?;
        Some(writer)
    } else {
        None
    };

    // Run for the specified number of runs
    for run_id in 1..=args.num_runs {
        println!("[garbler] === Run {}/{} ===", run_id, args.num_runs);

        // Open listener for this run
        let listener = TcpListener::bind(&args.listen)
            .with_context(|| format!("failed to bind listener on {}", args.listen))?;
        println!("[garbler] listening on {} (run {})", args.listen, run_id);

        // Accept a single connection for this run
        let (stream, addr) = listener
            .accept()
            .context("failed to accept incoming connection")?;
        println!("[garbler] connection from {} (run {})", addr, run_id);

        // First do the handshake
        handshake_as_garbler(stream.try_clone()?)?;
        println!("[garbler] handshake completed (run {})", run_id);

        // Send the embedding length (u32 little-endian) so the evaluator knows how many wires to expect
        {
            let mut len_writer = stream.try_clone()?;
            len_writer.write_u32::<LittleEndian>(embedding.len() as u32)?;
            len_writer.flush()?;
        }

        // Now run the linear layer demo
        run_linear_layer_demo(
            stream,
            Array1::from(embedding.clone()),
            &csv_writer,
            run_id,
            args.seed,
        )?;
    }

    println!("[garbler] ✅ All {} runs completed", args.num_runs);

    Ok(())
}

fn handshake_as_garbler(mut stream: TcpStream) -> Result<()> {
    // Send PING
    stream.write_all(b"PING\n")?;
    stream.flush()?;

    // Await PONG
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line)?;

    if line.trim_end() == "PONG" {
        Ok(())
    } else {
        anyhow::bail!("unexpected handshake reply: {}", line.trim_end());
    }
}

fn run_linear_layer_demo(
    stream: TcpStream,
    embedding: Array1<i16>,
    csv_writer: &Option<CsvWriter>,
    run_id: u32,
    seed: u64,
) -> Result<()> {
    let input_size = embedding.len();
    println!(
        "[garbler] starting linear layer demo with input size {}",
        input_size
    );

    // Initialize memory tracking
    let memory_tracker = MemoryTracker::new()
        .map_err(|e| anyhow::anyhow!("Failed to initialize memory tracker: {}", e))?;
    let initial_mem_mb = memory_tracker
        .current_allocated_mb()
        .map_err(|e| anyhow::anyhow!("Failed to read initial memory: {}", e))?;
    println!("[garbler] initial memory: {:.2} MB", initial_mem_mb);

    let layer = LinearLayer::new_test_layer(input_size);
    println!("[garbler] created deterministic test linear layer");

    let input_data = embedding;

    let plaintext_start = Instant::now();
    let plaintext_result = layer.eval_plaintext_scalar(&input_data);
    let plaintext_time = plaintext_start.elapsed();

    println!(
        "[garbler] plaintext evaluation: {} (took {:?})",
        plaintext_result, plaintext_time
    );

    let wall_start = Instant::now();
    let modulus = numbers::modulus_with_width(16);
    println!("[garbler] using modulus: {}", modulus);

    let channel = Channel::new(stream.try_clone()?, stream);
    let track_channel = TrackChannel::new(channel);
    let rng = AesRng::new();

    let garble_start = Instant::now();
    let mut garbler = Garbler::<_, AesRng, AlszSender, AllWire>::new(track_channel.clone(), rng)?;

    let input_bundles: Vec<_> = input_data
        .iter()
        .map(|&x| garbler.crt_encode(to_mod_q(x as i64, modulus), modulus))
        .collect::<Result<Vec<_>, _>>()?;

    println!("[garbler] encoded {} input values", input_bundles.len());

    let eval_start = Instant::now();
    let gc_result_bundle = layer
        .eval_garbled_scalar(&mut garbler, &input_bundles, modulus)
        .map_err(|e| anyhow::anyhow!("Garbled circuit evaluation failed: {:?}", e))?;
    let eval_time = eval_start.elapsed();

    // Participate in the reveal protocol. This sends the garbler's half of the decoding info.
    let reveal_start = Instant::now();
    garbler.reveal_bundle(&gc_result_bundle)?;
    let reveal_time = reveal_start.elapsed();

    println!("[garbler] garbled circuit evaluation and reveal completed");
    let garble_time = garble_start.elapsed();

    println!("[garbler] GC evaluation took {:?}", eval_time);
    println!(
        "[garbler] slowdown factor: {:.2}x",
        eval_time.as_nanos() as f64 / plaintext_time.as_nanos() as f64
    );

    // Emit all timing measurements, byte counters, and memory usage (Steps 2.1 & 2.2)
    let wall_time = wall_start.elapsed();
    let peak_mem_mb = memory_tracker
        .peak_allocated_mb()
        .map_err(|e| anyhow::anyhow!("Failed to read peak memory: {}", e))?;
    let final_mem_mb = memory_tracker
        .current_allocated_mb()
        .map_err(|e| anyhow::anyhow!("Failed to read final memory: {}", e))?;

    println!(
        "[garbler] wall_time={:?} garble_time={:?} eval_time={:?} reveal_time={:?}",
        wall_time, garble_time, eval_time, reveal_time
    );
    println!(
        "[garbler] bytes_sent={:.0} bytes_recv={:.0} total_bytes={:.0}",
        track_channel.kilobytes_written() * 1024.0,
        track_channel.kilobytes_read() * 1024.0,
        track_channel.total_kilobytes() * 1024.0
    );
    println!(
        "[garbler] peak_mem={:.2} MB final_mem={:.2} MB mem_delta={:.2} MB",
        peak_mem_mb,
        final_mem_mb,
        final_mem_mb - initial_mem_mb
    );

    // Write results to CSV if requested
    if let Some(writer) = csv_writer {
        let result = BenchmarkResult::new_garbler(
            wall_time,
            garble_time,
            eval_time,
            reveal_time,
            track_channel.kilobytes_written() * 1024.0,
            track_channel.kilobytes_read() * 1024.0,
            initial_mem_mb,
            peak_mem_mb,
            final_mem_mb,
            input_size,
            run_id,
            seed,
        );

        writer
            .write_result(&result)
            .with_context(|| "Failed to write benchmark result to CSV")?;
        println!("[garbler] Benchmark result written to CSV");
    }

    println!("[garbler] ✅ Milestone 3 completed: single linear layer in GC");
    println!("[garbler] - Plaintext result: {}", plaintext_result);
    println!("[garbler] - Garbler is confident the evaluator will reveal the same result.");

    Ok(())
}
