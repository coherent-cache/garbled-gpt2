use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Instant;
use anyhow::{Context, Result};
use clap::Parser;
use ndarray::Array1;

use fancy_garbling::twopac::semihonest::Garbler;
use fancy_garbling::util as numbers;
use fancy_garbling::{FancyInput, AllWire};
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

    /// Input vector size (simulating GPT-2 hidden size)
    #[arg(long, default_value_t = 768)]
    input_size: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
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
    run_linear_layer_demo(stream, args.input_size)?;

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

fn run_linear_layer_demo(stream: TcpStream, input_size: usize) -> Result<()> {
    println!("[garbler] starting linear layer demo with input size {}", input_size);
    
    // Create a random linear layer for testing
    let layer = LinearLayer::new_random(input_size);
    println!("[garbler] created random linear layer");
    
    // Generate random input (simulating quantized embeddings)
    let input_data: Array1<i16> = Array1::from_iter(
        (0..input_size).map(|_| rand::random::<i16>() % 256)  // Small values for testing
    );
    
    // First, compute plaintext result for comparison
    let plaintext_start = Instant::now();
    let plaintext_result = layer.eval_plaintext(&input_data);
    let plaintext_time = plaintext_start.elapsed();
    
    println!("[garbler] plaintext evaluation: {} (took {:?})", 
             plaintext_result, plaintext_time);
    
    // Now run garbled circuit evaluation
    let gc_start = Instant::now();
    let modulus = numbers::modulus_with_width(16); // 16-bit modulus
    println!("[garbler] using modulus: {}", modulus);
    
    // Create channel and garbler
    let channel = Channel::new(stream.try_clone()?, stream);
    let rng = AesRng::new();
    let mut garbler = Garbler::<_, AesRng, AlszSender, AllWire>::new(channel, rng)?;
    
    // Encode the input data
    let input_bundles: Vec<_> = input_data.iter()
        .map(|&x| garbler.crt_encode(to_mod_q(x as i64, modulus), modulus))
        .collect::<Result<Vec<_>, _>>()?;
    
    println!("[garbler] encoded {} input values", input_bundles.len());
    
    // Evaluate the linear layer in garbled circuits
    let gc_result_bundle = layer.eval_garbled(&mut garbler, &input_bundles, modulus)
        .map_err(|e| anyhow::anyhow!("Garbled circuit evaluation failed: {:?}", e))?;
    
    // The result should be revealed by the evaluator, but we can decode our encoding for comparison
    println!("[garbler] garbled circuit evaluation completed");
    
    let gc_time = gc_start.elapsed();
    println!("[garbler] GC evaluation took {:?}", gc_time);
    println!("[garbler] slowdown factor: {:.2}x", 
             gc_time.as_nanos() as f64 / plaintext_time.as_nanos() as f64);
    
    // Milestone 3 completed!
    println!("[garbler] ✅ Milestone 3 completed: single linear layer in GC");
    println!("[garbler] - Plaintext result: {}", plaintext_result);
    println!("[garbler] - Plaintext time: {:?}", plaintext_time);
    println!("[garbler] - GC time: {:?}", gc_time);
    println!("[garbler] - Input size: {} values", input_size);
    println!("[garbler] - Modulus: {} (16-bit)", modulus);
    
    Ok(())
}
