use anyhow::{Context, Result};
use clap::Parser;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::time::Duration;

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
}

fn main() -> Result<()> {
    let args = Args::parse();
    let stream = connect_with_retry(&args.connect, args.retries, args.retry_delay_ms)?;
    println!("[evaluator] connected to {}", args.connect);
    println!("[evaluator] waiting for PING handshake from garbler …");

    handshake_as_evaluator(stream)?;
    println!("[evaluator] handshake completed – milestone 1 success 🎉");

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
