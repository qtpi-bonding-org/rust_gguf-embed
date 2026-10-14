# Multimodal embeddings (design, not implemented)

Status: design only, written 2026-10-14. Nothing here can be run until a
`llama-cpp-sys-2` release vendors llama.cpp with the `gemma-embedding2`
architecture (llama.cpp PR #30054, merged 2026-10-06). See episutra's
`docs/known-issues.md`, "EmbeddingGemma 2 swap".

## What the model needs

EmbeddingGemma 2 embeds text, images, video and audio into one 768-dim space.
Text goes through the main GGUF exactly as today. Images and audio go through
libmtmd: a second file, `mmproj-embeddinggemma-2-{Q8_0,BF16}.gguf`, holds the
vision and audio encoders. Same repo as the model: `ggml-org/embeddinggemma-2-GGUF`.

## What `llama-cpp-2` already gives us (behind its `mtmd` feature)

- `MtmdContext::init_from_file(mmproj_path, &LlamaModel, &MtmdContextParams)`
  and `support_vision()` / `support_audio()` / `get_audio_sample_rate()`.
- `MtmdBitmap::from_image_data(nx, ny, rgb)`, `from_file`, `from_buffer`,
  `from_audio_data(&[f32])` (PCM f32 at the sample rate above).
- `MtmdContext::tokenize(MtmdInputText, &[&MtmdBitmap]) -> MtmdInputChunks`.
  The text must contain one media marker (`mtmd_default_marker()`, currently
  `<__media__>`) per bitmap.
- `MtmdInputChunks::eval_chunks(mtmd_ctx, llama_ctx, n_past, seq_id, n_batch,
  logits_last)`: decodes text chunks and encodes + decodes media chunks.
- `MtmdContextParams { use_gpu, n_threads, image_min_tokens, image_max_tokens, .. }`.

`MtmdContext` is `Send + Sync` for tokenize; encode/eval take `&mut`.

## Proposed API

Additive; the text API does not change. Cargo feature `mtmd` (off by default,
so consumers that only embed text do not compile libmtmd):

```rust
// config
GgufEmbedderConfig::with_mmproj(PathBuf)      // optional; None = text only
GgufEmbedderConfig::with_media_tokens(min, max) // maps to image_min/max_tokens

// embedder (feature "mtmd")
GgufEmbedder::supports_images(&self) -> bool
GgufEmbedder::supports_audio(&self) -> bool
GgufEmbedder::embed_image_blocking(&self, rgb: &[u8], width: u32, height: u32) -> Result<Vec<f32>, _>
GgufEmbedder::embed_audio_blocking(&self, pcm_f32: &[f32]) -> Result<Vec<f32>, _>
```

Same contract as text: pooled, truncated to `dimension`, L2-normalized, via the
existing `truncate_and_normalize` / `validate_raw_vector`. New error variants:
`MmprojLoad`, `MediaUnsupported`, `MediaTokenize`, `MediaEval`. A missing
mmproj with a media call returns `MediaUnsupported`, never a panic.

Decoding files (JPEG, PNG, WAV, MP3) is the caller's job; this crate takes raw
pixels and PCM, in line with "model acquisition and policy belong to the
caller". `MtmdBitmap::from_file` is not exposed (it pulls in libmtmd's own
decoders and a path policy).

## Design constraints and open questions (all need the real model)

1. **Pooling across a media item.** The text path uses one `encode()` per
   bucket with mean pooling and `n_batch = n_ubatch = n_ctx`, so a sequence
   pools in a single ubatch. `eval_chunks` splits work by `n_batch`. A media
   item whose token count exceeds one ubatch could pool wrongly. Cap the media
   token budget (`image_max_tokens`) so one item always fits `n_ctx`, and
   verify by comparing against llama-server's `/v1/embeddings` output.
2. **Reading the result.** Expected: after `eval_chunks(.., seq_id = 0)`, read
   `ctx.embeddings_seq_ith(0)` as the text path does. Unverified.
3. **Token count.** llama.cpp #30195 (open): vision may produce more tokens than
   configured. Check `chunks.total_tokens()` against `n_ctx` before eval and
   return an error instead of truncating.
4. **Shared context.** One persistent `LlamaContext` behind a `Mutex` today.
   Media calls must use the same lock and `clear_kv_cache()`, and the
   `SendLlamaContext` justification must be re-checked for `eval_chunks`.
   `MtmdContext` needs its own `&mut` access: wrap it in the same mutex.
5. **GPU policy.** `gpu_layers == Some(0)` registers no devices for the text
   model; set `MtmdContextParams.use_gpu = false` in the same case.
6. **Prefixes.** Text queries/documents use task prefixes
   (`task: search result | query: ...`, `title: ... | text: ...`). Whether the
   media marker needs a prefix is unknown; check the model card and the
   llama.cpp server behavior.
7. **Memory.** The mmproj adds weights (the vision tower is ~170M and the audio
   tower ~300M parameters). Measure resident size before enabling by default.

## Test plan once the model loads

- Unit (no model): marker counting and argument validation helpers.
- Ignored real-model tests, mirroring the existing ones: an image of a known
  object embeds closer to its caption than to an unrelated caption; repeated
  calls are deterministic (cosine > 0.9999); text still matches the text-only
  path after media calls (no KV-cache bleed).
