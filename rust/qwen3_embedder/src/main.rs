use std::io::{self, BufRead, Write};

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use fastembed::Qwen3TextEmbedding;
use serde::{Deserialize, Serialize};

const DEFAULT_MODEL: &str = "Qwen/Qwen3-Embedding-8B";
const DEFAULT_TASK: &str =
    "Given a developer query, retrieve relevant code snippets and technical documentation that answer the query";

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Request {
    Health,
    EmbedDocuments { texts: Vec<String> },
    EmbedQuery { text: String, task: Option<String> },
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Response {
    Ready {
        model: String,
        vector_size: usize,
        backend: String,
        dtype: String,
        max_length: usize,
    },
    Embeddings { embeddings: Vec<Vec<f32>> },
    Error { message: String },
}

struct Embedder {
    model_name: String,
    backend: String,
    dtype: DType,
    max_length: usize,
    model: Qwen3TextEmbedding,
}

impl Embedder {
    fn load() -> Result<Self> {
        let model_name = std::env::var("QWEN3_EMBEDDING_MODEL")
            .unwrap_or_else(|_| DEFAULT_MODEL.to_string());
        let max_length = std::env::var("QWEN3_MAX_LENGTH")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1024);

        let preferred_device = select_device();
        let preferred_dtype = dtype_for_device(&preferred_device);

        match load_on(&model_name, &preferred_device, preferred_dtype, max_length) {
            Ok(model) => Ok(Self {
                model_name,
                backend: device_name(&preferred_device),
                dtype: preferred_dtype,
                max_length,
                model,
            }),
            Err(first_error) => {
                let cpu = Device::Cpu;
                let cpu_dtype = DType::F32;
                let model = load_on(&model_name, &cpu, cpu_dtype, max_length)
                    .with_context(|| format!("failed after preferred backend error: {first_error:?}"))?;
                Ok(Self {
                    model_name,
                    backend: "cpu".to_string(),
                    dtype: cpu_dtype,
                    max_length,
                    model,
                })
            }
        }
    }

    fn health(&self) -> Response {
        Response::Ready {
            model: self.model_name.clone(),
            vector_size: self.model.config().hidden_size,
            backend: self.backend.clone(),
            dtype: format!("{:?}", self.dtype).to_lowercase(),
            max_length: self.max_length,
        }
    }

    fn embed_documents(&self, texts: Vec<String>) -> Result<Response> {
        let refs = texts.iter().map(String::as_str).collect::<Vec<_>>();
        let embeddings = self.model.embed(&refs)?;
        Ok(Response::Embeddings { embeddings })
    }

    fn embed_query(&self, text: String, task: Option<String>) -> Result<Response> {
        let task = task.unwrap_or_else(|| DEFAULT_TASK.to_string());
        let query = format!("Instruct: {task}\nQuery: {text}");
        let refs = [query.as_str()];
        let embeddings = self.model.embed(&refs)?;
        Ok(Response::Embeddings { embeddings })
    }
}

fn load_on(
    model_name: &str,
    device: &Device,
    dtype: DType,
    max_length: usize,
) -> Result<Qwen3TextEmbedding> {
    Qwen3TextEmbedding::from_hf(model_name, device, dtype, max_length)
        .with_context(|| format!("failed to load {model_name} on {}", device_name(device)))
}

fn select_device() -> Device {
    let requested = std::env::var("QWEN3_DEVICE")
        .unwrap_or_else(|_| "auto".to_string())
        .to_lowercase();
    if requested == "cpu" {
        return Device::Cpu;
    }

    #[cfg(target_os = "macos")]
    {
        if requested == "auto" || requested == "metal" {
            if let Ok(device) = Device::new_metal(0) {
                return device;
            }
        }
    }

    Device::Cpu
}

fn dtype_for_device(device: &Device) -> DType {
    let requested = std::env::var("QWEN3_DTYPE").unwrap_or_default().to_lowercase();
    match requested.as_str() {
        "f32" => return DType::F32,
        "f16" => return DType::F16,
        "bf16" => return DType::BF16,
        _ => {}
    }

    if matches!(device, Device::Cpu) {
        DType::F32
    } else {
        DType::F16
    }
}

fn device_name(device: &Device) -> String {
    if matches!(device, Device::Cpu) {
        "cpu".to_string()
    } else {
        "metal".to_string()
    }
}

fn write_response(response: &Response) -> Result<()> {
    let mut stdout = io::stdout().lock();
    serde_json::to_writer(&mut stdout, response)?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let embedder = Embedder::load()?;
    write_response(&embedder.health())?;

    for line in io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let response = match serde_json::from_str::<Request>(&line) {
            Ok(Request::Health) => embedder.health(),
            Ok(Request::EmbedDocuments { texts }) => match embedder.embed_documents(texts) {
                Ok(response) => response,
                Err(error) => Response::Error {
                    message: error.to_string(),
                },
            },
            Ok(Request::EmbedQuery { text, task }) => match embedder.embed_query(text, task) {
                Ok(response) => response,
                Err(error) => Response::Error {
                    message: error.to_string(),
                },
            },
            Err(error) => Response::Error {
                message: error.to_string(),
            },
        };

        write_response(&response)?;
    }

    Ok(())
}
