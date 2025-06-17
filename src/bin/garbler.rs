use anyhow::{Context, Result};
use clap::Parser;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};

/// Garbler side – opens a listener, waits for the evaluator, then does a simple PING/PONG handshake.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Address to listen on, e.g. 0.0.0.0:7000
    #[arg(long, default_value = "0.0.0.0:7000")]
    listen: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let listener = TcpListener::bind(&args.listen)
        .with_context(|| format!("failed to bind listener on {}", args.listen))?;
    println!("[garbler] listening on {}", args.listen);

    // Accept a single connection for now.
    let (stream, addr) = listener
        .accept()
        .context("failed to accept incoming connection")?;
    println!("[garbler] connection from {}", addr);

    handshake_as_garbler(stream)?;
    println!("[garbler] handshake completed – milestone 1 success 🎉");

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
