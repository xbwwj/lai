use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command as ProcessCommand, ExitCode, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Parser, Subcommand};
use console::style;
use pdf_extract::extract_text_from_mem;
use reqwest::blocking::Client;
use serde_json::{Value, json};
use tempfile::TempDir;

const DEFAULT_MODEL: &str = "tencent/Hy-MT2-7B-GGUF:Q4_K_M";

const HEALTH_URL: &str = "http://localhost/health";
const CHAT_COMPLETIONS_URL: &str = "http://localhost/v1/chat/completions";

const DEFAULT_CHUNK_CHARS: usize = 4000;
const MAX_SERVER_LOG_BYTES: usize = 1024 * 1024;

fn main() -> Result<ExitCode> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Translate(translate) => translate.run(),
    }
}

#[derive(Debug, Parser)]
#[command(name = "lai", version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Translate(Translate),
}

/// Translate one or more text files or PDF files into Simplified Chinese.
#[derive(Debug, Args)]
struct Translate {
    /// Input files.
    #[arg(value_name = "FILE", required = true, num_args = 1..)]
    files: Vec<PathBuf>,

    /// Write translated text to stdout instead of <filename>.zh.md.
    /// By default, each translation is written beside its input file.
    #[arg(long)]
    stdout: bool,

    /// Maximum Unicode characters per translation request.
    #[arg(long, default_value_t = DEFAULT_CHUNK_CHARS, value_name = "CHARS")]
    chunk_chars: usize,

    /// Hugging Face model repository and quantization.
    #[arg(long, default_value = DEFAULT_MODEL)]
    model: String,

    /// Path to llama-server executable.
    #[arg(long, default_value = "llama-server", value_name = "PATH")]
    llama_server: PathBuf,

    /// Maximum seconds to wait for model download and startup.
    /// Set to 0 to disable the timeout.
    #[arg(long, default_value_t = 1800, value_name = "SECONDS")]
    startup_timeout_secs: u64,
}

impl Translate {
    fn run(self) -> Result<ExitCode> {
        let started_at = Instant::now();

        if self.chunk_chars == 0 {
            bail!("--chunk-chars must be greater than zero");
        }

        let server = LlamaServer::spawn(&self)?;

        let interrupted = Arc::new(AtomicBool::new(false));
        server.install_signal_handler(Arc::clone(&interrupted))?;

        // Health client with a short timeout.
        let health_client = Client::builder()
            .unix_socket(server.socket_path.clone())
            .timeout(Duration::from_secs(2))
            .build()
            .context("failed to build Unix-socket health client")?;

        // Wait for model download and server initialization.
        let startup_timeout = Duration::from_secs(self.startup_timeout_secs);
        let startup_result = server.wait_until_ready(&health_client, startup_timeout, &interrupted);

        if interrupted.load(Ordering::SeqCst) {
            status_error("Interrupted", "while waiting for llama-server");
            return Ok(ExitCode::from(130));
        }

        startup_result?;

        // API client. No fixed timeout for long translation requests.
        let api_client = Client::builder()
            .unix_socket(server.socket_path.clone())
            .timeout(None)
            .build()
            .context("failed to build Unix-socket API client")?;

        // Translate files sequentially.
        let mut failed = false;
        let mut completed = 0usize;

        for file in &self.files {
            if interrupted.load(Ordering::SeqCst) {
                break;
            }

            match translate_file(&api_client, file, self.chunk_chars, self.stdout) {
                Ok(()) => completed += 1,
                Err(error) => {
                    if interrupted.load(Ordering::SeqCst) {
                        break;
                    }

                    status_error("Error", format!("{}: {error:#}", file.display()));

                    failed = true;

                    // If the server crashed, print its log and stop.
                    if let Some(status) = server.try_wait()? {
                        status_error("Error", format!("llama-server exited ({status})"));
                        eprintln!("{}", server.read_log());
                        break;
                    }
                }
            }
        }

        if interrupted.load(Ordering::SeqCst) {
            status_error("Interrupted", "translation cancelled");
            Ok(ExitCode::from(130))
        } else if failed {
            status_error(
                "Failed",
                format!("{completed}/{} files translated", self.files.len()),
            );
            Ok(ExitCode::FAILURE)
        } else {
            status(
                "Finished",
                format!(
                    "{} file(s) translated in {:.2}s",
                    completed,
                    started_at.elapsed().as_secs_f64()
                ),
            );
            Ok(ExitCode::SUCCESS)
        }
    }
}

// -----------------------------------------------------------------------------
// Cargo-style terminal status output
// -----------------------------------------------------------------------------

/// Print a Cargo-style status line to stderr.
///
/// Status labels are right-aligned so a sequence of stages forms a compact,
/// easy-to-scan log. `console` automatically avoids ANSI styling when colors
/// are disabled or stderr is not a terminal.
fn status(stage: &str, message: impl std::fmt::Display) {
    eprintln!(
        "{}  {message}",
        style(format!("{stage:>12}")).green().bold()
    );
}

fn status_error(stage: &str, message: impl std::fmt::Display) {
    eprintln!("{}  {message}", style(format!("{stage:>12}")).red().bold());
}

// -----------------------------------------------------------------------------
// Input processing
// -----------------------------------------------------------------------------

fn is_pdf_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("pdf"))
}

/// Read a text file or extract text from a PDF entirely in memory.
fn read_input(file: &Path) -> Result<String> {
    let content = if is_pdf_file(file) {
        status("Extracting", format!("PDF text from {}", file.display()));

        let bytes =
            fs::read(file).with_context(|| format!("failed to read PDF: {}", file.display()))?;

        extract_text_from_mem(&bytes)
            .with_context(|| format!("failed to extract text from PDF: {}", file.display()))?
    } else {
        status("Reading", file.display());

        fs::read_to_string(file)
            .with_context(|| format!("failed to read text file: {}", file.display()))?
    };

    if content.trim().is_empty() {
        if is_pdf_file(file) {
            bail!(
                "no text extracted from PDF; \
                 the document may be scanned or image-only (OCR is not supported)"
            );
        }

        bail!("input file contains no text");
    }

    Ok(content)
}

/// Convert an input path into its translated output path.
///
/// article.md  -> article.zh.md
/// article.pdf -> article.zh.md
/// article.txt -> article.zh.md
fn translated_output_path(input: &Path) -> PathBuf {
    let stem = input
        .file_stem()
        .or_else(|| input.file_name())
        .unwrap_or_default();

    let mut filename = stem.to_os_string();
    filename.push(".zh.md");

    input.with_file_name(filename)
}

// -----------------------------------------------------------------------------
// In-memory text splitting
// -----------------------------------------------------------------------------

/// Take at most `max_chars` Unicode characters from the beginning of `text`.
///
/// Prefer splitting immediately after the last whitespace character within
/// the limit. All original whitespace is preserved.
fn take_prefix(text: &str, max_chars: usize) -> (&str, &str) {
    let mut count = 0;
    let mut end = 0;
    let mut whitespace_end = None;
    let mut has_more = false;

    for (index, ch) in text.char_indices() {
        if count == max_chars {
            has_more = true;
            break;
        }

        count += 1;
        end = index + ch.len_utf8();

        if ch.is_whitespace() {
            whitespace_end = Some(end);
        }
    }

    if !has_more {
        return (text, "");
    }

    let split = whitespace_end.unwrap_or(end);

    (&text[..split], &text[split..])
}

/// Split text into chunks without creating intermediate files.
///
/// - Prefer line boundaries.
/// - Keep each chunk within `max_chars`.
/// - Split oversized lines at whitespace where possible.
/// - Preserve original whitespace and line breaks.
fn split_text(text: &str, max_chars: usize) -> Vec<String> {
    let mut chunks = Vec::new();

    let mut current = String::new();
    let mut current_chars = 0;

    for line in text.split_inclusive('\n') {
        let mut remaining = line;

        while !remaining.is_empty() {
            let (part, rest) = take_prefix(remaining, max_chars);

            let part_chars = part.chars().count();

            if current_chars > 0 && current_chars + part_chars > max_chars {
                chunks.push(std::mem::take(&mut current));
                current_chars = 0;
            }

            current.push_str(part);
            current_chars += part_chars;

            remaining = rest;

            // An oversized line was split.
            // Keep each piece in a separate chunk.
            if !remaining.is_empty() {
                chunks.push(std::mem::take(&mut current));
                current_chars = 0;
            }
        }
    }

    if !current.is_empty() {
        chunks.push(current);
    }

    chunks
}

// -----------------------------------------------------------------------------
// Translation
// -----------------------------------------------------------------------------

fn translate_file(
    client: &Client,
    file: &Path,
    chunk_chars: usize,
    write_to_stdout: bool,
) -> Result<()> {
    if !file.is_file() {
        bail!("file not found");
    }

    // Read the entire input into memory.
    // PDF extraction also happens entirely in memory.
    let content = read_input(file)?;

    // Split the extracted text in memory.
    status(
        "Chunking",
        format!("{} (max {chunk_chars} chars/chunk)", file.display()),
    );

    let chunks = split_text(&content, chunk_chars);
    let chunk_count = chunks.len();

    if chunk_count == 0 {
        bail!("no text chunks generated");
    }

    status(
        "Chunked",
        format!("{} into {chunk_count} chunks", file.display()),
    );

    // Accumulate the entire translation in memory.
    // Publish the complete result only after all requests succeed.
    let mut translation = String::new();

    // Use stable Cargo-style status lines instead of a dynamic progress bar.
    for (index, chunk) in chunks.iter().enumerate() {
        let chunk_index = index + 1;

        status(
            "Translating",
            format!("{} ({chunk_index}/{chunk_count})", file.display()),
        );

        let translated_chunk = request_translation(client, chunk).with_context(|| {
            format!(
                "translation failed at chunk {}/{}",
                chunk_index, chunk_count
            )
        })?;

        // Separate adjacent translated chunks.
        if index > 0 {
            translation.push_str("\n\n");
        }

        translation.push_str(&translated_chunk);
    }

    status(
        "Translated",
        format!("{} ({chunk_count} chunks)", file.display()),
    );

    // Write the result only after all requests succeed.
    if write_to_stdout {
        status(
            "Writing",
            format!("translation to stdout ({})", file.display()),
        );

        let stdout = std::io::stdout();
        let mut stdout = stdout.lock();

        stdout
            .write_all(translation.as_bytes())
            .context("failed to write translation to stdout")?;

        if !translation.ends_with('\n') {
            stdout.write_all(b"\n")?;
        }

        stdout.write_all(b"\n")?;
        stdout.flush()?;

        status(
            "Wrote",
            format!("translation to stdout ({})", file.display()),
        );
    } else {
        let output = translated_output_path(file);

        status("Writing", output.display());

        fs::write(&output, translation.as_bytes())
            .with_context(|| format!("failed to write output file: {}", output.display()))?;

        status("Wrote", output.display());
    }

    Ok(())
}

fn request_translation(client: &Client, content: &str) -> Result<String> {
    let prompt = format!(
        "Translate the following text into Simplified Chinese. \
         Preserve the original structure, headings, lists, citations, \
         formulas, code blocks, and paragraph breaks as much as possible. \
         Do not summarize, omit content, or add explanations. \
         Output only the translation:\n\n{content}"
    );

    let payload = json!({
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "stream": false
    });

    let response = client
        .post(CHAT_COMPLETIONS_URL)
        .json(&payload)
        .send()
        .context("failed to send chat completion request")?;

    let status = response.status();

    let body = response
        .text()
        .context("failed to read chat completion response")?;

    let value: Value = serde_json::from_str(&body)
        .with_context(|| format!("server returned invalid JSON (HTTP {status}): {body}"))?;

    if !status.is_success() {
        let message = value
            .pointer("/error/message")
            .and_then(Value::as_str)
            .unwrap_or(body.as_str());

        bail!("HTTP {status}: {message}");
    }

    if let Some(translation) = value
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
    {
        return Ok(translation.to_owned());
    }

    if let Some(message) = value.pointer("/error/message").and_then(Value::as_str) {
        bail!("{message}");
    }

    bail!("unexpected chat completion response: {body}");
}

struct LlamaServer {
    child: Arc<Mutex<Child>>,
    _temp_dir: TempDir,
    socket_path: PathBuf,
    log: Arc<Mutex<Vec<u8>>>,
    log_reader: Option<thread::JoinHandle<()>>,
}

impl LlamaServer {
    fn spawn(args: &Translate) -> Result<Self> {
        let temp_dir = tempfile::Builder::new()
            .prefix("local-translator-")
            .tempdir()
            .context("failed to create temporary socket directory")?;

        // llama.cpp detects AF_UNIX from the ".sock" suffix.
        let socket_path = temp_dir.path().join("llama.sock");

        status("Starting", format!("llama-server -hf {}", args.model));

        let mut child = match ProcessCommand::new(&args.llama_server)
            .arg("-hf")
            .arg(&args.model)
            .arg("--host")
            .arg(&socket_path)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(error) => {
                status_error("Error", "failed to start llama-server");

                return Err(error).with_context(|| {
                    format!(
                        "failed to start {} -hf {}",
                        args.llama_server.display(),
                        args.model
                    )
                });
            }
        };

        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| anyhow!("failed to capture llama-server stderr"))?;

        // Keep the last 1 MiB of server logs in memory.
        let log = Arc::new(Mutex::new(Vec::new()));

        let reader_log = Arc::clone(&log);

        let log_reader = thread::spawn(move || {
            capture_server_log(stderr, reader_log);
        });

        status("Started", "llama-server process");

        Ok(Self {
            child: Arc::new(Mutex::new(child)),
            _temp_dir: temp_dir,
            socket_path,
            log,
            log_reader: Some(log_reader),
        })
    }

    /// Install Ctrl-C handling.
    ///
    /// Kill llama-server immediately so an in-flight request over the
    /// Unix socket is interrupted as well.
    fn install_signal_handler(&self, interrupted: Arc<AtomicBool>) -> Result<()> {
        let child = Arc::clone(&self.child);

        ctrlc::set_handler(move || {
            interrupted.store(true, Ordering::SeqCst);

            if let Ok(mut child) = child.lock() {
                let _ = child.kill();
            }
        })
        .context("failed to install Ctrl-C handler")
    }

    fn try_wait(&self) -> Result<Option<ExitStatus>> {
        let mut child = self
            .child
            .lock()
            .map_err(|_| anyhow!("llama-server child lock was poisoned"))?;

        child
            .try_wait()
            .context("failed to query llama-server status")
    }

    fn wait_until_ready(
        &self,
        client: &Client,
        timeout: Duration,
        interrupted: &AtomicBool,
    ) -> Result<()> {
        status(
            "Waiting",
            "for model download and llama-server initialization",
        );

        let deadline = if timeout.is_zero() {
            None
        } else {
            Some(Instant::now() + timeout)
        };

        loop {
            if interrupted.load(Ordering::SeqCst) {
                status_error("Interrupted", "while waiting for llama-server");
                bail!("interrupted while waiting for llama-server");
            }

            // Check the health endpoint through the Unix socket.
            if let Ok(response) = client.get(HEALTH_URL).send() {
                if response.status().is_success() {
                    status("Ready", "llama-server");
                    return Ok(());
                }
            }

            // Check whether the server process has exited.
            if let Some(status) = self.try_wait()? {
                status_error(
                    "Error",
                    format!("llama-server exited during startup ({status})"),
                );

                bail!(
                    "llama-server exited during startup ({status})\n\
                     Server log:\n{}",
                    self.read_log()
                );
            }

            // Check startup deadline.
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                status_error(
                    "Timeout",
                    format!(
                        "llama-server did not become ready within {} seconds",
                        timeout.as_secs()
                    ),
                );

                bail!(
                    "llama-server startup timed out after {} seconds\n\
                     Server log:\n{}",
                    timeout.as_secs(),
                    self.read_log()
                );
            }

            thread::sleep(Duration::from_secs(1));
        }
    }

    fn read_log(&self) -> String {
        match self.log.lock() {
            Ok(log) if !log.is_empty() => String::from_utf8_lossy(&log).into_owned(),
            Ok(_) => "<server log is empty>".to_owned(),
            Err(_) => "<server log lock was poisoned>".to_owned(),
        }
    }
}

/// Drain server stderr continuously to prevent pipe blockage.
///
/// Only the last MAX_SERVER_LOG_BYTES bytes are retained.
fn capture_server_log<R: Read>(mut reader: R, log: Arc<Mutex<Vec<u8>>>) {
    let mut buffer = [0u8; 8192];

    loop {
        let count = match reader.read(&mut buffer) {
            Ok(0) | Err(_) => break,
            Ok(count) => count,
        };

        let Ok(mut log) = log.lock() else {
            break;
        };

        if count >= MAX_SERVER_LOG_BYTES {
            log.clear();

            log.extend_from_slice(&buffer[count - MAX_SERVER_LOG_BYTES..count]);
        } else {
            let overflow = log
                .len()
                .saturating_add(count)
                .saturating_sub(MAX_SERVER_LOG_BYTES);

            if overflow > 0 {
                log.drain(..overflow);
            }

            log.extend_from_slice(&buffer[..count]);
        }
    }
}

impl Drop for LlamaServer {
    fn drop(&mut self) {
        if let Ok(mut child) = self.child.lock() {
            match child.try_wait() {
                // Already exited and reaped.
                Ok(Some(_)) => {}

                // Still running, or status couldn't be queried.
                _ => {
                    let _ = child.kill();
                    let _ = child.wait();
                }
            }
        }

        // The child has stopped, so stderr should reach EOF.
        if let Some(reader) = self.log_reader.take() {
            let _ = reader.join();
        }
    }
}
