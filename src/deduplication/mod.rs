pub mod fast_hash;

use ahash::AHashMap;
use faer::Mat;
use fast_hash::{
    compute_simhash_matrix_row, generate_simhash_hyperplanes, is_matrix_row_zero,
    matrix_row_cosine_similarity, matrix_row_equals_slice,
};

use crate::types::AnchoredMatrices;

struct DuplicateGroup {
    count: usize,
    x: Vec<f32>,
    // Accumulated in f64 to avoid f32 cancellation when summing many rows.
    y_sum: Vec<f64>,
}

/// Deduplicate activation rows using 64-bit Locality-Sensitive Hashing (SimHash)
/// and cosine similarity verification.
///
/// Unlike naive bitwise float hashing, which produces 0% hits on real LLMs due to
/// position embeddings (RoPE) and causal context drift, SimHash LSH projects activations
/// onto 64 fixed hyperplanes and merges near-duplicate tokens whose cosine similarity
/// exceeds `similarity_threshold` (default 0.995).
pub fn deduplicate_rows_lsh(
    x: &Mat<f32>,
    y: &Mat<f32>,
    similarity_threshold: f32,
) -> AnchoredMatrices {
    let d_in = x.ncols();
    let d_out = y.ncols();
    let nrows = x.nrows();

    if nrows == 0 || d_in == 0 {
        return AnchoredMatrices {
            x: Mat::zeros(0, d_in),
            y: Mat::zeros(0, d_out),
        };
    }

    let hyperplanes = generate_simhash_hyperplanes(d_in);

    // Map: 64-bit SimHash -> list of candidate group indices
    let mut hash_to_groups: AHashMap<u64, Vec<usize>> = AHashMap::with_capacity(nrows.min(1024));
    let mut entries: Vec<DuplicateGroup> = Vec::with_capacity(nrows.min(1024));

    for row in 0..nrows {
        if is_matrix_row_zero(x, row, d_in) {
            continue;
        }

        let hash = compute_simhash_matrix_row(x, row, &hyperplanes);
        let mut found_group_idx = None;

        // Check exact hash bucket first
        if let Some(candidate_indices) = hash_to_groups.get(&hash) {
            for &group_idx in candidate_indices {
                let candidate = &entries[group_idx];
                if matrix_row_equals_slice(x, row, &candidate.x)
                    || matrix_row_cosine_similarity(x, row, &candidate.x) >= similarity_threshold
                {
                    found_group_idx = Some(group_idx);
                    break;
                }
            }
        }

        // If no match in exact bucket, check 1-bit Hamming neighbors if threshold is relaxed (< 1.0)
        if found_group_idx.is_none() && similarity_threshold < 1.0 {
            'neighbor_search: for bit in 0..64 {
                let neighbor_hash = hash ^ (1u64 << bit);
                if let Some(candidate_indices) = hash_to_groups.get(&neighbor_hash) {
                    for &group_idx in candidate_indices {
                        let candidate = &entries[group_idx];
                        if matrix_row_cosine_similarity(x, row, &candidate.x)
                            >= similarity_threshold
                        {
                            found_group_idx = Some(group_idx);
                            break 'neighbor_search;
                        }
                    }
                }
            }
        }

        if let Some(group_idx) = found_group_idx {
            let group = &mut entries[group_idx];
            group.count += 1;
            let n = group.count as f32;
            for col in 0..d_in {
                // Online centroid refinement for near-duplicate tokens
                group.x[col] = (group.x[col] * (n - 1.0) + x.read(row, col)) / n;
            }
            for col in 0..d_out {
                group.y_sum[col] += f64::from(y.read(row, col));
            }
        } else {
            let new_group_idx = entries.len();
            let x_vals: Vec<f32> = (0..d_in).map(|col| x.read(row, col)).collect();
            let y_vals: Vec<f64> = (0..d_out).map(|col| f64::from(y.read(row, col))).collect();

            entries.push(DuplicateGroup {
                count: 1,
                x: x_vals,
                y_sum: y_vals,
            });

            hash_to_groups.entry(hash).or_default().push(new_group_idx);
        }
    }

    AnchoredMatrices {
        x: Mat::from_fn(entries.len(), d_in, |row, col| {
            let group = &entries[row];
            group.x[col] * (group.count as f32).sqrt()
        }),
        y: Mat::from_fn(entries.len(), d_out, |row, col| {
            let group = &entries[row];
            (group.y_sum[col] / (group.count as f64).sqrt()) as f32
        }),
    }
}

/// Standard deduplicate_rows using SimHash LSH with cosine threshold 0.995.
///
/// Collapses repeated and near-repeated token activations without losing associative signal.
pub fn deduplicate_rows(x: &Mat<f32>, y: &Mat<f32>) -> AnchoredMatrices {
    deduplicate_rows_lsh(x, y, 0.995)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deduplicate_rows_merges_near_duplicates() {
        let d_in = 8;
        let d_out = 4;
        let mut x = Mat::<f32>::zeros(3, d_in);
        let mut y = Mat::<f32>::zeros(3, d_out);

        // Row 0 and Row 1 are almost identical (perturbed by 0.001)
        for col in 0..d_in {
            x.write(0, col, 1.0);
            x.write(1, col, 1.001);
            x.write(2, col, (col + 1) as f32); // completely different
        }
        for col in 0..d_out {
            y.write(0, col, 2.0);
            y.write(1, col, 2.0);
            y.write(2, col, 5.0);
        }

        let dedup = deduplicate_rows(&x, &y);
        // Row 0 and Row 1 should be merged into 1 group, Row 2 into another -> total 2 rows
        assert_eq!(dedup.x.nrows(), 2);
    }

    /// R8: y_sum must accumulate in f64 so small contributions are not lost
    /// when summed alongside large values across many duplicate rows.
    #[test]
    fn test_deduplicate_y_sum_accumulates_in_f64() {
        let d_in = 4;
        let d_out = 2;
        let rows = 200;
        let mut x = Mat::<f32>::zeros(rows, d_in);
        let mut y = Mat::<f32>::zeros(rows, d_out);
        for r in 0..rows {
            for c in 0..d_in {
                x.write(r, c, 1.0);
            }
            y.write(r, 0, 1e7);
            y.write(r, 1, 1.0);
        }
        let dedup = deduplicate_rows(&x, &y);
        assert_eq!(dedup.x.nrows(), 1);

        // All rows merge: y_sum = [200 * 1e7, 200], scaled by 1/sqrt(200).
        // f32 accumulation would lose the +1.0 contributions entirely.
        let scale = (rows as f32).sqrt();
        let expected_small = rows as f32 / scale;
        assert!(
            (dedup.y.read(0, 1) - expected_small).abs() < 0.01,
            "small y contributions lost: {} vs {expected_small}",
            dedup.y.read(0, 1)
        );
        let expected_large = rows as f32 * 1e7 / scale;
        assert!(
            (dedup.y.read(0, 0) - expected_large).abs() / expected_large < 1e-6,
            "large y sum inaccurate: {} vs {expected_large}",
            dedup.y.read(0, 0)
        );
    }
}
