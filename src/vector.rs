/// A raw embedding vector failed validation before normalization.
/// Normalization would otherwise mask each of these as a clean unit vector,
/// silently poisoning a downstream vector index with a corrupted embedding.
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    /// The model returned fewer elements than the truncation target `dim` —
    /// the actual silent-corruption path: `truncate_and_normalize` would
    /// otherwise take fewer than `dim` elements and normalize a short vector
    /// cleanly, poisoning the index with a wrong-length embedding.
    #[error("embedding dimension mismatch: got {got} elements, need at least {expected}")]
    DimensionMismatch { got: usize, expected: usize },
    /// The raw vector contains NaN or +/-Inf.
    #[error("embedding contains NaN or infinite values")]
    NonFiniteVector,
    /// The raw vector's norm is below epsilon — indistinguishable from noise
    /// once normalized. Subsumes the exact-zero case.
    #[error("embedding norm is below epsilon (degenerate/near-zero vector)")]
    DegenerateVector,
}

/// Guard the RAW vector before normalization — normalization masks all of
/// these as a unit vector. `dim` is the truncation target `truncate_and_normalize`
/// will be called with; `raw` must carry at least that many elements.
pub fn validate_raw_vector(raw: &[f32], dim: usize) -> Result<(), EmbeddingError> {
    if raw.len() < dim {
        return Err(EmbeddingError::DimensionMismatch { got: raw.len(), expected: dim });
    }
    if raw.iter().any(|x| !x.is_finite()) {
        return Err(EmbeddingError::NonFiniteVector);
    }
    let norm: f32 = raw.iter().take(dim).map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-6 {
        // subsumes the exact-zero check and catches denormal noise
        return Err(EmbeddingError::DegenerateVector);
    }
    Ok(())
}

/// Take the first `dim` elements of `v` and L2-normalise. Returns the zero vector if norm < ε.
/// Kept as a backstop even where callers already validate via [validate_raw_vector].
pub fn truncate_and_normalize(v: Vec<f32>, dim: usize) -> Vec<f32> {
    let truncated: Vec<f32> = v.into_iter().take(dim).collect();
    let norm: f32 = truncated.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-9 {
        return truncated;
    }
    truncated.into_iter().map(|x| x / norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_produces_correct_dimension() {
        let v: Vec<f32> = (0..768).map(|i| i as f32).collect();
        let result = truncate_and_normalize(v, 128);
        assert_eq!(result.len(), 128);
    }

    #[test]
    fn normalize_produces_unit_vector() {
        let v = vec![3.0f32, 4.0, 0.0, 0.0];
        let result = truncate_and_normalize(v, 2);
        let mag: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((mag - 1.0).abs() < 1e-6, "magnitude was {mag}");
    }

    #[test]
    fn normalize_handles_zero_vector() {
        let v = vec![0.0f32; 768];
        let result = truncate_and_normalize(v, 128);
        assert_eq!(result.len(), 128);
        assert!(result.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn validate_raw_vector_rejects_wrong_length() {
        let v = vec![1.0f32; 64];
        let result = validate_raw_vector(&v, 128);
        assert!(matches!(
            result,
            Err(EmbeddingError::DimensionMismatch { got: 64, expected: 128 })
        ));
    }

    #[test]
    fn validate_raw_vector_rejects_non_finite() {
        let mut v = vec![1.0f32; 128];
        v[10] = f32::NAN;
        assert!(matches!(validate_raw_vector(&v, 128), Err(EmbeddingError::NonFiniteVector)));

        let mut v = vec![1.0f32; 128];
        v[20] = f32::INFINITY;
        assert!(matches!(validate_raw_vector(&v, 128), Err(EmbeddingError::NonFiniteVector)));
    }

    #[test]
    fn validate_raw_vector_rejects_near_zero_norm() {
        let v = vec![0.0f32; 128];
        assert!(matches!(validate_raw_vector(&v, 128), Err(EmbeddingError::DegenerateVector)));
    }

    #[test]
    fn validate_raw_vector_accepts_healthy_vector() {
        let v: Vec<f32> = (0..768).map(|i| i as f32 + 1.0).collect();
        assert!(validate_raw_vector(&v, 128).is_ok());
    }
}
