//! `impl catuskoti::traits::Embedder for GgufEmbedder`, feature-gated so
//! non-catuskoti consumers of this crate don't pull catuskoti in.

use std::error::Error;

use catuskoti::traits::Embedder;

use crate::GgufEmbedder;

#[async_trait::async_trait]
impl Embedder for GgufEmbedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, Box<dyn Error>> {
        GgufEmbedder::embed(self, text).await.map_err(|e| -> Box<dyn Error> { e.to_string().into() })
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
        GgufEmbedder::embed_batch(self, texts)
            .await
            .map_err(|e| -> Box<dyn Error> { e.to_string().into() })
    }

    fn dimension(&self) -> usize {
        GgufEmbedder::dimension(self)
    }
}
