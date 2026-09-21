use std::collections::HashMap;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::mpsc::RecvTimeoutError;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;

use crate::backend::backend;
use crate::vector::{truncate_and_normalize, validate_raw_vector};
use crate::GgufEmbedError;

/// Heartbeat cadence for a stalled/slow `embed_batch` call. Pick a value
/// that matches whatever caller-side stall-detection you correlate against
/// (episutra matches its outbox-worker heartbeat at 15s).
const DEFAULT_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(15);

/// Configuration for a [`GgufEmbedder`]. `dimension` is the only field
/// without a sensible library default — it's an MRL-truncation target the
/// caller decides (tied to whatever vector index the embeddings feed).
#[derive(Debug, Clone)]
pub struct GgufEmbedderConfig {
    /// Truncate+normalize pooled embeddings to this many dimensions.
    pub dimension: usize,
    /// `None` leaves GPU offload at the llama.cpp library default. Callers
    /// that want an env-var-driven override can compute this via
    /// [`crate::gpu_layers_override`] and pass the result in — this crate
    /// never reads the environment itself.
    pub gpu_layers: Option<u32>,
    /// Context window. Chunks longer than this are truncated. Also bounds
    /// n_batch/n_ubatch (both pinned to this value so a whole sequence
    /// pools in a single ubatch, required for pooled embeddings).
    pub n_ctx: u32,
    /// Max chunks packed into one multi-sequence `encode()` call.
    pub max_seqs: i32,
    /// Heartbeat log cadence for a long-running `embed_batch` call.
    pub heartbeat_interval: Duration,
}

impl GgufEmbedderConfig {
    /// EmbeddingGemma is trained at a 2048-token context; 8 max sequences
    /// per batch is a reasonable default bound on per-batch KV-cache
    /// bookkeeping (see [`bucket_by_token_budget`]).
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            gpu_layers: None,
            n_ctx: 2048,
            max_seqs: 8,
            heartbeat_interval: DEFAULT_HEARTBEAT_INTERVAL,
        }
    }

    pub fn with_gpu_layers(mut self, gpu_layers: Option<u32>) -> Self {
        self.gpu_layers = gpu_layers;
        self
    }

    pub fn with_n_ctx(mut self, n_ctx: u32) -> Self {
        self.n_ctx = n_ctx;
        self
    }

    pub fn with_max_seqs(mut self, max_seqs: i32) -> Self {
        self.max_seqs = max_seqs;
        self
    }
}

/// Greedily packs chunk token-counts into buckets bounded by `max_tokens`
/// total and `max_seqs` items, preserving input order. A single chunk
/// whose own count exceeds `max_tokens` still gets a one-item bucket —
/// it never blocks packing of the chunks around it.
fn bucket_by_token_budget(token_counts: &[usize], max_tokens: usize, max_seqs: usize) -> Vec<Vec<usize>> {
    let mut buckets: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut current_tokens: usize = 0;
    for (idx, &count) in token_counts.iter().enumerate() {
        let would_exceed_tokens = !current.is_empty() && current_tokens + count > max_tokens;
        let would_exceed_seqs = current.len() >= max_seqs;
        if would_exceed_tokens || would_exceed_seqs {
            buckets.push(std::mem::take(&mut current));
            current_tokens = 0;
        }
        current.push(idx);
        current_tokens += count;
    }
    if !current.is_empty() {
        buckets.push(current);
    }
    buckets
}

/// Wraps `LlamaContext` so it can live behind a `Mutex` inside `GgufEmbedder`.
/// `LlamaContext` has no `Send`/`Sync` impl anywhere in `llama-cpp-2`
/// (it holds a raw `NonNull<llama_context>`) — safe here because `LlamaModel`
/// itself already asserts `Send + Sync`, and every `LlamaContext` method used
/// in this file (`encode`, `clear_kv_cache`, `embeddings_seq_ith`) only
/// requires non-concurrent access, which the enclosing `Mutex` already
/// guarantees — never that the context stays pinned to the thread that
/// created it. If you add a call to a different `LlamaContext` method here,
/// re-check that it holds too before relying on this impl.
struct SendLlamaContext(LlamaContext<'static>);
unsafe impl Send for SendLlamaContext {}

impl std::ops::Deref for SendLlamaContext {
    type Target = LlamaContext<'static>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for SendLlamaContext {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Process-wide cache of loaded models, keyed by GGUF path. `LlamaModel` is
/// never freed once loaded (a `GgufEmbedder`'s context borrows it for
/// `'static`), so this makes "load each distinct GGUF path at most once" a
/// guarantee this crate provides, not an obligation callers have to
/// remember — two `GgufEmbedder::new` calls for the same path share one
/// leaked model instead of silently leaking a second one.
static MODEL_CACHE: OnceLock<Mutex<HashMap<PathBuf, &'static LlamaModel>>> = OnceLock::new();

fn get_or_load_model(gguf_path: &Path, gpu_layers: Option<u32>) -> Result<&'static LlamaModel, GgufEmbedError> {
    let cache = MODEL_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().map_err(|_| GgufEmbedError::LockPoisoned)?;
    if let Some(model) = guard.get(gguf_path) {
        return Ok(model);
    }
    let backend = backend()?;
    let mut params = LlamaModelParams::default();
    if let Some(n) = gpu_layers {
        params = params.with_n_gpu_layers(n);
    }
    let model = LlamaModel::load_from_file(backend, gguf_path, &params)
        .map_err(|e| GgufEmbedError::ModelLoad { path: gguf_path.to_path_buf(), reason: e.to_string() })?;
    let model: &'static LlamaModel = Box::leak(Box::new(model));
    guard.insert(gguf_path.to_path_buf(), model);
    Ok(model)
}

/// Loads a GGUF embedding model via llama.cpp and produces fixed-dimension,
/// L2-normalized embedding vectors. See the crate docs for scope.
pub struct GgufEmbedder {
    model: &'static LlamaModel,
    // Holds the one persistent llama.cpp context (one model, one Metal
    // queue, one KV-cache buffer) AND serializes access to it. This design
    // assumes embeds run off any hot path (batch ingest, not interactive
    // request/response) — serializing is the safe, simple choice under that
    // assumption. Built once at load time instead of per `embed_blocking`
    // call so the n_ctx-token KV-cache buffer isn't repeatedly
    // allocated/freed. Do not parallelize this without re-justifying
    // `SendLlamaContext`'s unsafe impl above.
    ctx: Mutex<SendLlamaContext>,
    dimension: usize,
    n_ctx: usize,
    max_seqs: usize,
    heartbeat_interval: Duration,
}

impl GgufEmbedder {
    /// Load the GGUF embedding model from `gguf_path`. Blocking model load
    /// runs on a blocking thread so this doesn't stall the async executor.
    pub async fn new(gguf_path: PathBuf, config: GgufEmbedderConfig) -> Result<Self, GgufEmbedError> {
        tokio::task::spawn_blocking(move || Self::load_blocking(gguf_path, config))
            .await
            .map_err(|e| GgufEmbedError::LoadTaskPanicked(e.to_string()))?
    }

    fn load_blocking(gguf_path: PathBuf, config: GgufEmbedderConfig) -> Result<Self, GgufEmbedError> {
        let model = get_or_load_model(&gguf_path, config.gpu_layers)?;
        let backend = backend()?;
        let ctx = model
            .new_context(backend, Self::ctx_params(config.n_ctx, config.max_seqs))
            .map_err(|e| GgufEmbedError::ContextCreation(e.to_string()))?;
        Ok(Self {
            model,
            ctx: Mutex::new(SendLlamaContext(ctx)),
            dimension: config.dimension,
            n_ctx: config.n_ctx as usize,
            max_seqs: config.max_seqs as usize,
            heartbeat_interval: config.heartbeat_interval,
        })
    }

    fn ctx_params(n_ctx: u32, max_seqs: i32) -> LlamaContextParams {
        let threads = std::thread::available_parallelism().map(|n| n.get() as i32).unwrap_or(4);
        LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_batch(n_ctx)
            .with_n_ubatch(n_ctx)
            .with_n_threads(threads)
            .with_n_threads_batch(threads)
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::Mean)
            .with_n_seq_max(max_seqs as u32)
            // Without this, llama.cpp splits n_ctx evenly across n_seq_max
            // sequences instead of sharing the full context window — see
            // llama-context.cpp:215-228.
            .with_kv_unified(true)
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Synchronous batch embed. One context for the whole batch; each text is
    /// decoded as a single mean-pooled sequence. Returns vectors
    /// truncated+normalized to this embedder's configured dimension.
    pub fn embed_blocking(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, GgufEmbedError> {
        // Heartbeat: this call is either waiting on `self.ctx`'s lock (held by
        // another embed_blocking call, since embeds are serialized) or inside
        // llama.cpp's encode() — a genuinely stuck call and a genuinely slow
        // one look identical from outside without this. `recv_timeout` (not a
        // plain `sleep` loop) so completion wakes this thread immediately
        // instead of it always waiting out a full tick before joining.
        let n_texts = texts.len();
        let interval = self.heartbeat_interval;
        let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
        let heartbeat = std::thread::spawn(move || {
            let mut elapsed = Duration::ZERO;
            loop {
                match done_rx.recv_timeout(interval) {
                    Ok(()) | Err(RecvTimeoutError::Disconnected) => break,
                    Err(RecvTimeoutError::Timeout) => {
                        elapsed += interval;
                        tracing::warn!(
                            n_texts,
                            elapsed_ms = elapsed.as_millis() as u64,
                            "gguf-embed: still embedding"
                        );
                    }
                }
            }
        });

        let result = self.embed_blocking_inner(texts);

        let _ = done_tx.send(());
        let _ = heartbeat.join();
        result
    }

    fn embed_blocking_inner(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, GgufEmbedError> {
        let dim = self.dimension;
        let mut ctx = self.ctx.lock().map_err(|_| GgufEmbedError::LockPoisoned)?;

        // Tokenize + truncate every text up front. Empty texts short-circuit
        // to a zero vector and never enter a batch; everything else is
        // queued for bucketing so bucket decisions use real (post-
        // truncation) token counts.
        let n_ctx = self.n_ctx;
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); texts.len()];
        let mut pending_idx: Vec<usize> = Vec::new();
        let mut pending_tokens: Vec<Vec<LlamaToken>> = Vec::new();
        for (i, text) in texts.iter().enumerate() {
            let mut tokens = self
                .model
                .str_to_token(text, AddBos::Always)
                .map_err(|e| GgufEmbedError::Tokenize(e.to_string()))?;
            tokens.truncate(n_ctx);
            if tokens.is_empty() {
                out[i] = vec![0.0; dim];
                continue;
            }
            pending_idx.push(i);
            pending_tokens.push(tokens);
        }

        let token_counts: Vec<usize> = pending_tokens.iter().map(Vec::len).collect();
        let buckets = bucket_by_token_budget(&token_counts, n_ctx, self.max_seqs);

        for bucket in buckets {
            let total_tokens: usize = bucket.iter().map(|&b| token_counts[b]).sum();
            let mut batch = LlamaBatch::new(total_tokens, bucket.len() as i32);
            for (seq_id, &b) in bucket.iter().enumerate() {
                // logits_all = true marks every token as an output, which mean-pooling
                // requires (otherwise llama.cpp logs an override warning per call).
                batch
                    .add_sequence(&pending_tokens[b], seq_id as i32, true)
                    .map_err(|e| GgufEmbedError::BatchConstruction(e.to_string()))?;
            }
            ctx.clear_kv_cache();
            // EmbeddingGemma is a bidirectional encoder → use encode(), not decode().
            // (decode() auto-redirects to encode() but warns on every call.)
            ctx.encode(&mut batch).map_err(|e| GgufEmbedError::Encode(e.to_string()))?;
            for (seq_id, &b) in bucket.iter().enumerate() {
                let raw = ctx
                    .embeddings_seq_ith(seq_id as i32)
                    .map_err(|e| GgufEmbedError::ReadEmbedding(e.to_string()))?;
                validate_raw_vector(raw, dim)?;
                out[pending_idx[b]] = truncate_and_normalize(raw.to_vec(), dim);
            }
        }

        Ok(out)
    }

    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, GgufEmbedError> {
        let mut v = self.embed_blocking(std::slice::from_ref(&text.to_string()))?;
        v.pop().ok_or(GgufEmbedError::EmptyResult)
    }

    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, GgufEmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.embed_blocking(texts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_by_token_budget_empty_input() {
        assert_eq!(bucket_by_token_budget(&[], 2048, 8), Vec::<Vec<usize>>::new());
    }

    #[test]
    fn bucket_by_token_budget_single_small_chunk() {
        assert_eq!(bucket_by_token_budget(&[100], 2048, 8), vec![vec![0]]);
    }

    #[test]
    fn bucket_by_token_budget_packs_multiple_small_chunks_together() {
        let counts = vec![100, 200, 150];
        assert_eq!(bucket_by_token_budget(&counts, 2048, 8), vec![vec![0, 1, 2]]);
    }

    #[test]
    fn bucket_by_token_budget_splits_on_token_budget() {
        let counts = vec![1200, 1200, 100];
        assert_eq!(bucket_by_token_budget(&counts, 2048, 8), vec![vec![0], vec![1, 2]]);
    }

    #[test]
    fn bucket_by_token_budget_splits_on_max_seqs() {
        let counts = vec![10; 10];
        let buckets = bucket_by_token_budget(&counts, 2048, 8);
        assert_eq!(buckets, vec![vec![0, 1, 2, 3, 4, 5, 6, 7], vec![8, 9]]);
    }

    #[test]
    fn bucket_by_token_budget_oversized_chunk_gets_solo_bucket() {
        let counts = vec![3000, 100];
        assert_eq!(bucket_by_token_budget(&counts, 2048, 8), vec![vec![0], vec![1]]);
    }

    #[test]
    fn bucket_by_token_budget_preserves_order() {
        let counts = vec![500, 500, 500, 500, 500];
        let buckets = bucket_by_token_budget(&counts, 2048, 8);
        let flattened: Vec<usize> = buckets.into_iter().flatten().collect();
        assert_eq!(flattened, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn ctx_params_sets_n_seq_max_for_batched_embedding() {
        assert_eq!(GgufEmbedder::ctx_params(2048, 8).n_seq_max(), 8);
    }

    #[test]
    fn ctx_params_sets_kv_unified_so_sequences_share_the_full_context() {
        // Without this, llama.cpp divides n_ctx by n_seq_max per-sequence,
        // silently corrupting any chunk over that per-seq ceiling. Verified
        // against llama-context.cpp:215-228 in the vendored llama.cpp source.
        assert!(GgufEmbedder::ctx_params(2048, 8).kv_unified());
    }

    // ── Real-GGUF regression tests ───────────────────────────────────────
    //
    // `#[ignore]`d — these load an actual GGUF file. Set GGUF=/path/to/model.gguf
    // or drop one at ./.gguf-embed-cache/embeddinggemma-300m-qat-Q8_0.gguf
    // (episutra-frb/examples/llama_embed_ram.rs has the download step).
    // Moved verbatim from episutra-frb's embedder.rs (2026-09-21 extraction)
    // — these test this crate's own behavior, not anything episutra-specific.

    fn test_gguf_path() -> PathBuf {
        std::env::var("GGUF").map(PathBuf::from).unwrap_or_else(|_| {
            PathBuf::from("./.gguf-embed-cache/embeddinggemma-300m-qat-Q8_0.gguf")
        })
    }

    #[tokio::test]
    #[ignore = "loads the real GGUF model"]
    async fn embed_batch_matches_individual_embeds() {
        let embedder = GgufEmbedder::new(test_gguf_path(), GgufEmbedderConfig::new(128))
            .await
            .expect("load embedder");

        // Two LARGE chunks (~1300+ tokens each via repetition — comfortably
        // over the 256-token/seq ceiling that kv_unified=false would silently
        // impose, and together over n_ctx=2048, so bucket_by_token_budget is
        // forced to split them across at least two buckets) plus several
        // small chunks that pack alongside them. This is deliberately NOT a
        // set of tiny strings — a batch of tiny texts would fit in one bucket
        // under 256 tokens total and would never have exercised either the
        // kv_unified fix or the multi-bucket path.
        let large = "The quick brown fox jumps over the lazy dog. ".repeat(120);
        let texts: Vec<String> = vec![
            large.clone(),
            large,
            "Rust is a systems programming language.".to_string(),
            "SurrealDB is an embedded graph database.".to_string(),
            "EmbeddingGemma produces 768-dim vectors.".to_string(),
            "catuskoti batches chunks per note.".to_string(),
        ];

        let batched = embedder.embed_batch(&texts).await.expect("batch embed failed");

        let mut individual = Vec::with_capacity(texts.len());
        for t in &texts {
            individual.push(embedder.embed(t).await.expect("single embed failed"));
        }

        assert_eq!(batched.len(), individual.len());
        for (i, (b, s)) in batched.iter().zip(individual.iter()).enumerate() {
            assert_eq!(b.len(), s.len(), "dim mismatch at index {i}");
            let dot: f32 = b.iter().zip(s.iter()).map(|(x, y)| x * y).sum();
            assert!(
                dot > 0.999,
                "batched vs individual embedding diverged at index {i}: cosine sim {dot}"
            );
        }
    }

    #[tokio::test]
    #[ignore = "loads the real GGUF model"]
    async fn embed_repeated_calls_are_consistent() {
        let embedder = GgufEmbedder::new(test_gguf_path(), GgufEmbedderConfig::new(128))
            .await
            .expect("load embedder");

        let text = "Persistent context reuse must not leak state between calls.".to_string();

        let first = embedder.embed(&text).await.expect("first embed failed");
        let second = embedder.embed(&text).await.expect("second embed failed");

        assert_eq!(first.len(), second.len());
        let dot: f32 = first.iter().zip(second.iter()).map(|(x, y)| x * y).sum();
        assert!(
            dot > 0.9999,
            "same text embedded twice via the persistent context diverged: cosine sim {dot}"
        );

        // Re-run a multi-bucket batch (large chunk + small ones) twice in a
        // row against the now-persistent context, to catch any interaction
        // between multi-sequence batching and cross-call context reuse —
        // e.g. stale KV-cache entries from a PREVIOUS CALL's last bucket
        // bleeding into the next call's first bucket, which a single-small
        // -text batch would never surface.
        let large = "The quick brown fox jumps over the lazy dog. ".repeat(120);
        let texts: Vec<String> = vec![
            large,
            "First note in a simulated multi-note ingest.".to_string(),
            "Second note, different content entirely.".to_string(),
        ];
        let batch_one = embedder.embed_batch(&texts).await.expect("batch one failed");
        let batch_two = embedder.embed_batch(&texts).await.expect("batch two failed");
        for (i, (a, b)) in batch_one.iter().zip(batch_two.iter()).enumerate() {
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            assert!(dot > 0.9999, "batch {i} diverged across repeated calls: cosine sim {dot}");
        }
    }

    #[tokio::test]
    #[ignore = "loads the real GGUF model, runs ~600 embed_batch calls — minutes, not seconds"]
    async fn embed_survives_many_repeated_batch_calls() {
        // Reproduces a real ingest run's shape as closely as possible against
        // one persistent context: hundreds of consecutive `embed_batch` calls,
        // each with two short chunk texts — regression test for a hang
        // observed in production after ~500 prior notes had gone through the
        // same context (episutra's tree-of-life reindex).
        let embedder = GgufEmbedder::new(test_gguf_path(), GgufEmbedderConfig::new(128))
            .await
            .expect("load embedder");

        const CALLS: usize = 600;
        const PER_CALL_TIMEOUT: Duration = Duration::from_secs(30);

        for i in 0..CALLS {
            let texts = vec![
                format!("The order Testorder{i} is a taxonomic group with 3 described descendants in the GBIF backbone."),
                format!("Section body {i}: a short unrelated sentence about a made-up organism."),
            ];
            let result = tokio::time::timeout(PER_CALL_TIMEOUT, embedder.embed_batch(&texts)).await;
            match result {
                Err(_) => panic!(
                    "embed_batch call #{i} of {CALLS} did not complete within {PER_CALL_TIMEOUT:?} \
                     — reproduces the indefinite hang seen in production after ~500 prior calls"
                ),
                Ok(Err(e)) => panic!("embed_batch call #{i} of {CALLS} returned an error: {e}"),
                Ok(Ok(vectors)) => assert_eq!(vectors.len(), texts.len(), "call #{i}: wrong vector count"),
            }
        }
    }
}
