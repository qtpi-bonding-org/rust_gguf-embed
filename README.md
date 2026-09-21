# gguf-embed

Given a path to a GGUF embedding model, produce fixed-dimension, L2-normalized
embedding vectors via [llama.cpp](https://github.com/ggml-org/llama.cpp)
(through `llama-cpp-2`).

## Scope

In: the llama.cpp backend singleton (`backend()`, process-global — may only
init once per process), model + context lifetime, tokenize/batch/pool, MRL
truncate+normalize, and the `unsafe impl Send for SendLlamaContext` this
requires (see its doc comment in `src/embedder.rs` before touching it — the
soundness argument is scoped to the exact set of `LlamaContext` methods
called there).

Out: model *acquisition* — which URL, which model, download/caching policy.
Callers pass a `PathBuf` to an already-present file.

```rust
let config = GgufEmbedderConfig::new(768); // MRL truncation dimension
let embedder = GgufEmbedder::new(gguf_path, config).await?;
let vectors = embedder.embed_batch(&texts).await?;
```

## `catuskoti` feature

Off by default. Turns on `impl catuskoti::traits::Embedder for GgufEmbedder`,
so catuskoti-based consumers (episutra, concordium) don't need their own
adapter shim. Pulls in `catuskoti` as a path dependency — assumes catuskoti
is checked out as a sibling under `systematic-action/concordium`, same
layout episutra and the rest of `my-libs` already assume.

## History

Extracted from `episutra-frb` (2026-09-21) so concordium/catuskoti could
reuse the same implementation — including the hand-justified `unsafe impl`
above — rather than duplicating it. Four things were fixed in the move that
were correct inside episutra but stop being correct once shared: `eprintln!`
heartbeat logging → `tracing::warn!`; a hardcoded env-var GPU-layers read →
a `GgufEmbedderConfig` field (callers that want env-var control compute it
themselves via `gpu_layers_override` and pass the result in); an unconditional
`Box::leak` per construction → a path-keyed process-wide model cache (two
constructions for the same GGUF path now share one leaked model instead of
silently leaking a second); and `Box<dyn Error + Send + Sync>` + `format!`
strings → a `thiserror` enum, so a caller can distinguish "model missing" from
a transient embed failure.

Deliberately unchanged: the serialized single-context design (one `Mutex`
around the one `LlamaContext`) — embeds are assumed to run off any hot path
(batch ingest, not interactive request/response). Parallelizing this would
break the `unsafe impl Send` justification above; don't, without re-justifying
it first.
