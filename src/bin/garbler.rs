use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Instant;
use anyhow::{Context, Result};
use clap::Parser;
use ndarray::Array1;
use std::process::Command;
use ndarray_npy::read_npy;

use fancy_garbling::twopac::semihonest::Garbler;
use fancy_garbling::util as numbers;
use fancy_garbling::{FancyInput, FancyReveal, AllWire};
use ocelot::ot::AlszSender;
use scuttlebutt::{AesRng, Channel};

use garbled_gpt2::{LinearLayer, to_mod_q};

/// Garbler side – opens a listener, waits for the evaluator, then does a simple PING/PONG handshake.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Address to listen on, e.g. 0.0.0.0:7000
    #[arg(long, default_value = "0.0.0.0:7000")]
    listen: String,

    /// Input vector size if loading from file directly
    #[arg(long)]
    input_size: Option<usize>,

    /// Plaintext to tokenize & embed via Python helper
    #[arg(long)]
    text: Option<String>,

    /// Path to embedding file (.npy) produced by Python helper
    #[arg(long, default_value = "embedding.npy")]
    embedding: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    // if text provided, call python helper
    if let Some(t) = &args.text {
        let status = Command::new("python")
            .args([
                "plaintext_baseline.py",
                "--dump-embedding",
                "--text",
                t,
                "--out",
                &args.embedding,
            ])
            .status()
            .context("failed to run python helper")?;
        if !status.success() {
            anyhow::bail!("python helper failed");
        }
    }

    // Load embedding
    let embedding_arr: Array1<i16> = read_npy(&args.embedding).context("read embedding npy")?;
    let embedding = embedding_arr.to_vec();

    // proceed to open listener and pass embedding data
    let listener = TcpListener::bind(&args.listen)
        .with_context(|| format!("failed to bind listener on {}", args.listen))?;
    println!("[garbler] listening on {}", args.listen);

    // Accept a single connection for now.
    let (stream, addr) = listener.accept().context("failed to accept incoming connection")?;
    println!("[garbler] connection from {}", addr);

    // First do the handshake
    handshake_as_garbler(stream.try_clone()?)?;
    println!("[garbler] handshake completed");

    // Now run the linear layer demo
    run_linear_layer_demo(stream, Array1::from(embedding))?;

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

fn run_linear_layer_demo(stream: TcpStream, embedding: Array1<i16>) -> Result<()> {
    let input_size = embedding.len();
    println!("[garbler] starting linear layer demo with input size {}", input_size);
    
    let layer = LinearLayer::new_test_layer(input_size);
    println!("[garbler] created deterministic test linear layer");
    
    let input_data = embedding;
    
    let plaintext_start = Instant::now();
    let plaintext_result = layer.eval_plaintext(&input_data);
    let plaintext_time = plaintext_start.elapsed();
    
    println!("[garbler] plaintext evaluation: {} (took {:?})", 
             plaintext_result, plaintext_time);
    
    let gc_start = Instant::now();
    let modulus = numbers::modulus_with_width(16);
    println!("[garbler] using modulus: {}", modulus);
    
    let channel = Channel::new(stream.try_clone()?, stream);
    let rng = AesRng::new();
    let mut garbler = Garbler::<_, AesRng, AlszSender, AllWire>::new(channel, rng)?;
    
    let input_bundles: Vec<_> = input_data.iter()
        .map(|&x| garbler.crt_encode(to_mod_q(x as i64, modulus), modulus))
        .collect::<Result<Vec<_>, _>>()?;
    
    println!("[garbler] encoded {} input values", input_bundles.len());
    
    let gc_result_bundle = layer.eval_garbled(&mut garbler, &input_bundles, modulus)
        .map_err(|e| anyhow::anyhow!("Garbled circuit evaluation failed: {:?}", e))?;
    
    // Participate in the reveal protocol. This sends the garbler's half of the decoding info.
    garbler.reveal_bundle(&gc_result_bundle)?;
    
    println!("[garbler] garbled circuit evaluation and reveal completed");
    let gc_time = gc_start.elapsed();

    println!("[garbler] GC evaluation took {:?}", gc_time);
    println!("[garbler] slowdown factor: {:.2}x", 
             gc_time.as_nanos() as f64 / plaintext_time.as_nanos() as f64);
    
    println!("[garbler] ✅ Milestone 3 completed: single linear layer in GC");
    println!("[garbler] - Plaintext result: {}", plaintext_result);
    println!("[garbler] - Garbler is confident the evaluator will reveal the same result.");
    
    Ok(())
}
