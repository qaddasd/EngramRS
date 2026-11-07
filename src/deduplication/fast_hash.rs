use ahash::AHasher;
use faer::Mat;
use std::hash::Hasher;

/// Deterministically generate 64 random hyperplanes for SimHash LSH in dimension `d`.
///
/// Uses a 64-bit linear congruential generator (Knuth / MMIX parameters) with a fixed seed
/// to guarantee bitwise reproducibility across runs and platforms without external dependencies.
pub fn generate_simhash_hyperplanes(d: usize) -> Vec<f32> {
    let total_elements = 64 * d;
    let mut planes = Vec::with_capacity(total_elements);
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    for _ in 0..total_elements {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = (state >> 40) as f32 / ((1_u64 << 24) - 1) as f32;
        planes.push(2.0 * unit - 1.0);
    }
    planes
}

/// Compute 64-bit Locality-Sensitive Hashing (SimHash) signature for an f32 slice.
///
/// Bit `k` is 1 if the dot product of `slice` with hyperplane `k` is >= 0, else 0.
#[inline(always)]
pub fn compute_simhash_slice(slice: &[f32], hyperplanes: &[f32]) -> u64 {
    let d = slice.len();
    debug_assert!(hyperplanes.len() >= 64 * d);
    let mut hash: u64 = 0;
    for bit in 0..64 {
        let plane_offset = bit * d;
        let plane = &hyperplanes[plane_offset..plane_offset + d];
        let mut dot = 0.0f32;
        for i in 0..d {
            dot += slice[i] * plane[i];
        }
        if dot >= 0.0 {
            hash |= 1u64 << bit;
        }
    }
    hash
}

/// Compute 64-bit SimHash signature directly from a matrix row without memory allocation.
#[inline(always)]
pub fn compute_simhash_matrix_row(mat: &Mat<f32>, row: usize, hyperplanes: &[f32]) -> u64 {
    let d = mat.ncols();
    debug_assert!(hyperplanes.len() >= 64 * d);
    let mut hash: u64 = 0;
    for bit in 0..64 {
        let plane_offset = bit * d;
        let mut dot = 0.0f32;
        for i in 0..d {
            dot += mat.read(row, i) * hyperplanes[plane_offset + i];
        }
        if dot >= 0.0 {
            hash |= 1u64 << bit;
        }
    }
    hash
}

/// Fast cosine similarity between two slices.
#[inline(always)]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a_sq = 0.0f32;
    let mut norm_b_sq = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a_sq += a[i] * a[i];
        norm_b_sq += b[i] * b[i];
    }
    let denom = (norm_a_sq * norm_b_sq).sqrt();
    if denom <= 1e-12 {
        0.0
    } else {
        dot / denom
    }
}

/// Fast cosine similarity between a matrix row and an existing slice without allocation.
#[inline(always)]
pub fn matrix_row_cosine_similarity(mat: &Mat<f32>, row: usize, slice: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a_sq = 0.0f32;
    let mut norm_b_sq = 0.0f32;
    for (i, &b) in slice.iter().enumerate() {
        let a = mat.read(row, i);
        dot += a * b;
        norm_a_sq += a * a;
        norm_b_sq += b * b;
    }
    let denom = (norm_a_sq * norm_b_sq).sqrt();
    if denom <= 1e-12 {
        0.0
    } else {
        dot / denom
    }
}

/// Zero-allocation 64-bit hashing for a slice of f32 values.
///
/// Normalizes -0.0 to +0.0 so negative zero does not create a hash divergence.
#[inline(always)]
pub fn hash_f32_slice(slice: &[f32]) -> u64 {
    let mut hasher = AHasher::default();
    for &val in slice {
        let bits = if val == 0.0 { 0u32 } else { val.to_bits() };
        hasher.write_u32(bits);
    }
    hasher.finish()
}

/// Zero-allocation 64-bit hashing for a single matrix row without allocating a vector.
#[inline(always)]
pub fn hash_matrix_row(mat: &Mat<f32>, row: usize, cols: usize) -> u64 {
    let mut hasher = AHasher::default();
    for col in 0..cols {
        let val = mat.read(row, col);
        let bits = if val == 0.0 { 0u32 } else { val.to_bits() };
        hasher.write_u32(bits);
    }
    hasher.finish()
}

/// Check if a matrix row is all zeros without any memory allocation.
#[inline(always)]
pub fn is_matrix_row_zero(mat: &Mat<f32>, row: usize, cols: usize) -> bool {
    for col in 0..cols {
        if mat.read(row, col) != 0.0 {
            return false;
        }
    }
    true
}

/// Compare a matrix row against an existing slice for bitwise equality (normalizing -0.0 to +0.0).
#[inline(always)]
pub fn matrix_row_equals_slice(mat: &Mat<f32>, row: usize, slice: &[f32]) -> bool {
    if mat.ncols() != slice.len() {
        return false;
    }
    for (col, &b) in slice.iter().enumerate() {
        let a = mat.read(row, col);
        let a_bits = if a == 0.0 { 0u32 } else { a.to_bits() };
        let b_bits = if b == 0.0 { 0u32 } else { b.to_bits() };
        if a_bits != b_bits {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_f32_slice_normalizes_zero() {
        let positive_zero = [0.0f32, 1.0, 2.0];
        let negative_zero = [-0.0f32, 1.0, 2.0];
        assert_eq!(
            hash_f32_slice(&positive_zero),
            hash_f32_slice(&negative_zero)
        );
    }

    #[test]
    fn test_is_matrix_row_zero() {
        let m = Mat::from_fn(
            2,
            3,
            |row, col| {
                if row == 0 {
                    0.0
                } else {
                    (col + 1) as f32
                }
            },
        );
        assert!(is_matrix_row_zero(&m, 0, 3));
        assert!(!is_matrix_row_zero(&m, 1, 3));
    }

    #[test]
    fn test_simhash_and_cosine_similarity() {
        let d = 16;
        let planes = generate_simhash_hyperplanes(d);
        assert_eq!(planes.len(), 64 * d);

        let v1 = vec![1.0f32; d];
        let mut v2 = vec![1.0f32; d];
        v2[0] += 0.001; // tiny perturbation like RoPE / position

        let cos_sim = cosine_similarity(&v1, &v2);
        assert!(cos_sim > 0.999);

        let h1 = compute_simhash_slice(&v1, &planes);
        let h2 = compute_simhash_slice(&v2, &planes);
        let hamming_dist = (h1 ^ h2).count_ones();
        assert!(hamming_dist <= 2);
    }
}
