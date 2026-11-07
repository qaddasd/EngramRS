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
    y_sum: Vec<f32>,
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
                group.y_sum[col] += y.read(row, col);
            }
        } else {
            let new_group_idx = entries.len();
            let x_vals: Vec<f32> = (0..d_in).map(|col| x.read(row, col)).collect();
            let y_vals: Vec<f32> = (0..d_out).map(|col| y.read(row, col)).collect();

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
            group.y_sum[col] / (group.count as f32).sqrt()
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
}
