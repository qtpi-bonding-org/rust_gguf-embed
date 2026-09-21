//! Given a path to a GGUF embedding model, produce fixed-dimension,
//! L2-normalized embedding vectors via llama.cpp.
//!
//! Scope is deliberately narrow: the llama.cpp backend singleton, model +
//! context lifetime, tokenize/batch/pool, MRL truncate+normalize, and the
//! `unsafe impl Send` this requires (see [`embedder`] module docs). Model
//! *acquisition* (which URL, which model, download/caching policy) is the
//! caller's problem — this crate only ever takes a [`std::path::PathBuf`]
//! to an already-present file.
//!
//! Extracted from episutra-frb (2026-09-21) so concordium/catuskoti could
//! share the same llama.cpp embedder implementation — including its
//! hand-justified `unsafe impl Send for SendLlamaContext` — rather than
//! duplicating it. See that impl's doc comment before touching it.

mod backend;
mod embedder;
mod vector;

#[cfg(feature = "catuskoti")]
mod catuskoti_impl;

pub use backend::{backend, gpu_layers_override};
pub use embedder::{GgufEmbedder, GgufEmbedderConfig};
pub use vector::{truncate_and_normalize, validate_raw_vector, EmbeddingError};

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum GgufEmbedError {
    #[error("llama.cpp backend failed to initialize")]
    BackendInit,
    #[error("failed to load GGUF model at {path:?}: {reason}")]
    ModelLoad { path: PathBuf, reason: String },
    #[error("failed to create llama.cpp context: {0}")]
    ContextCreation(String),
    #[error("embedder load task panicked: {0}")]
    LoadTaskPanicked(String),
    #[error("embedder lock poisoned")]
    LockPoisoned,
    #[error("tokenize failed: {0}")]
    Tokenize(String),
    #[error("batch construction failed: {0}")]
    BatchConstruction(String),
    #[error("encode failed: {0}")]
    Encode(String),
    #[error("failed to read pooled embedding: {0}")]
    ReadEmbedding(String),
    #[error(transparent)]
    Validation(#[from] EmbeddingError),
    #[error("embed() returned no output for non-empty input")]
    EmptyResult,
}
