//! Process-global llama.cpp backend. `llama_backend_init` may only run once
//! per process — every `GgufEmbedder` (and any co-located llama.cpp-based
//! generator a caller builds alongside it) must share this one instance.

use std::sync::OnceLock;

use llama_cpp_2::llama_backend::LlamaBackend;

static LLAMA_BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

pub fn backend() -> Result<&'static LlamaBackend, crate::GgufEmbedError> {
    if LLAMA_BACKEND.get().is_none() {
        if let Ok(b) = LlamaBackend::init() {
            // Route llama.cpp / ggml C-layer logs into tracing so the
            // caller's subscriber filter controls what reaches stderr. Must
            // run once, after init(). Caller is responsible for having a
            // tracing subscriber registered by the time embeds happen.
            llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default());
            let _ = LLAMA_BACKEND.set(b);
        }
    }
    LLAMA_BACKEND.get().ok_or(crate::GgufEmbedError::BackendInit)
}

/// Parses a GPU layer count from an environment variable — a pure helper,
/// not called internally by [`crate::GgufEmbedder`] (GPU layers are a
/// [`crate::GgufEmbedderConfig`] field, not something this crate reads from
/// the environment itself). Kept here so callers that do want env-var
/// control (episutra) don't duplicate the parsing/arch-default logic; a
/// caller with its own config source (e.g. concordium's compose config)
/// can ignore this entirely.
///
/// `None` ⇒ leave `LlamaModelParams` at its library default (Metal offload on
/// macOS). `Some(n)` ⇒ explicit layer count (`0` forces CPU-only).
///
/// Absent an override: Apple Silicon has unified memory and full Metal tensor
/// support, so offload is left at the library default. Intel Macs commonly
/// pair this backend with a discrete GPU where Metal support is partial —
/// confirmed numerically broken on an AMD Radeon Pro 5300M — so `x86_64`
/// defaults to CPU-only.
pub fn gpu_layers_override(env_var: &str) -> Option<u32> {
    if let Ok(raw) = std::env::var(env_var) {
        match raw.parse::<u32>() {
            Ok(n) => return Some(n),
            Err(_) => tracing::warn!(
                env_var,
                value = %raw,
                "not a valid u32 — ignoring, using the architecture default"
            ),
        }
    }
    if cfg!(target_arch = "aarch64") { None } else { Some(0) }
}

#[cfg(test)]
mod tests {
    use super::gpu_layers_override;

    #[test]
    fn env_override_wins_per_variable() {
        let saved_embed = std::env::var("GGUF_EMBED_TEST_GPU_LAYERS_A").ok();
        let saved_llm = std::env::var("GGUF_EMBED_TEST_GPU_LAYERS_B").ok();

        unsafe { std::env::set_var("GGUF_EMBED_TEST_GPU_LAYERS_A", "0") };
        unsafe { std::env::set_var("GGUF_EMBED_TEST_GPU_LAYERS_B", "7") };
        assert_eq!(gpu_layers_override("GGUF_EMBED_TEST_GPU_LAYERS_A"), Some(0));
        assert_eq!(
            gpu_layers_override("GGUF_EMBED_TEST_GPU_LAYERS_B"),
            Some(7),
            "the two knobs are read independently"
        );

        unsafe { std::env::remove_var("GGUF_EMBED_TEST_GPU_LAYERS_B") };
        assert_eq!(
            gpu_layers_override("GGUF_EMBED_TEST_GPU_LAYERS_B"),
            if cfg!(target_arch = "aarch64") { None } else { Some(0) },
            "unset falls back to the architecture default",
        );

        unsafe { std::env::set_var("GGUF_EMBED_TEST_GPU_LAYERS_B", "not-a-number") };
        assert_eq!(
            gpu_layers_override("GGUF_EMBED_TEST_GPU_LAYERS_B"),
            if cfg!(target_arch = "aarch64") { None } else { Some(0) },
            "unparseable falls back to the architecture default",
        );

        match saved_embed {
            Some(v) => unsafe { std::env::set_var("GGUF_EMBED_TEST_GPU_LAYERS_A", v) },
            None => unsafe { std::env::remove_var("GGUF_EMBED_TEST_GPU_LAYERS_A") },
        }
        match saved_llm {
            Some(v) => unsafe { std::env::set_var("GGUF_EMBED_TEST_GPU_LAYERS_B", v) },
            None => unsafe { std::env::remove_var("GGUF_EMBED_TEST_GPU_LAYERS_B") },
        }
    }
}
