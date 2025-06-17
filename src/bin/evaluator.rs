use anyhow::{Context, Result};
use clap::Parser;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

use fancy_garbling::twopac::semihonest::Evaluator;
use fancy_garbling::util as numbers;
use fancy_garbling::{FancyInput, FancyReveal, AllWire};
use ocelot::ot::AlszReceiver;
use scuttlebutt::{AesRng, Channel};

use garbled_gpt2::{from_mod_q, LinearLayer};

/// Evaluator side – connects to the garbler, waits for PING, replies with PONG.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Address to connect to, e.g. 127.0.0.1:7000
    #[arg(long, default_value = "127.0.0.1:7000")]
    connect: String,

    /// Maximum connection retries
    #[arg(long, default_value_t = 10)]
    retries: u32,

    /// Delay between retries in milliseconds
    #[arg(long, default_value_t = 500)]
    retry_delay_ms: u64,

    /// Input vector size (simulating GPT-2 hidden size)
    #[arg(long, default_value_t = 768)]
    input_size: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let stream = connect_with_retry(&args.connect, args.retries, args.retry_delay_ms)?;
    println!("[evaluator] connected to {}", args.connect);

    // First do the handshake
    handshake_as_evaluator(stream.try_clone()?)?;
    println!("[evaluator] handshake completed");

    // Now participate in the linear layer demo
    participate_in_linear_layer_demo(stream, args.input_size)?;

    Ok(())
}

fn connect_with_retry(addr: &str, retries: u32, delay_ms: u64) -> Result<TcpStream> {
    let mut attempt = 0;
    loop {
        match TcpStream::connect(addr) {
            Ok(stream) => return Ok(stream),
            Err(err) if attempt < retries => {
                println!("[evaluator] connection attempt {} failed: {}", attempt + 1, err);
                std::thread::sleep(Duration::from_millis(delay_ms));
                attempt += 1;
            }
            Err(err) => return Err(err).with_context(|| format!("failed to connect to {} after {} retries", addr, retries)),
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

fn participate_in_linear_layer_demo(stream: TcpStream, input_size: usize) -> Result<()> {
    println!("[evaluator] participating in linear layer demo");
    let gc_start = Instant::now();
    
    // Create the *same* deterministic linear layer as the garbler
    let layer = LinearLayer::new_test_layer(input_size);
    let modulus = numbers::modulus_with_width(16); // 16-bit modulus
    
    println!("[evaluator] using modulus: {}", modulus);
    
    // Create channel and evaluator
    let channel = Channel::new(stream.try_clone()?, stream);
    let rng = AesRng::new();
    let mut evaluator = Evaluator::<_, AesRng, AlszReceiver, AllWire>::new(channel, rng)?;
    
    // Receive the encoded inputs from the garbler
    let input_bundles = evaluator.crt_receive_many(input_size, modulus)?;
    println!("[evaluator] received {} encoded input values", input_bundles.len());
    
    // Evaluate the garbled circuit. Both parties must call this.
    let gc_result_bundle = layer.eval_garbled(&mut evaluator, &input_bundles, modulus)
        .map_err(|e| anyhow::anyhow!("Garbled circuit evaluation failed: {:?}", e))?;
        
    // Reveal the result to get the plaintext value
    let modular_result_vec = evaluator.reveal_bundle(&gc_result_bundle)?;
    let primes = numbers::factor(modulus);
    let modular_result = numbers::crt_inv(&modular_result_vec, &primes);
    let result = from_mod_q(modular_result, modulus);

    let gc_time = gc_start.elapsed();

    println!("[evaluator] ✅ Milestone 3 validated!");
    println!("[evaluator] Garbled circuit result (revealed): {}", result);
    println!("[evaluator] GC participation took {:?}", gc_time);
    
    Ok(())
}
