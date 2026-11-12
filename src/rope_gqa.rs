use std::borrow::Cow;

/// Return the number of query heads associated with each key/value head.
///
/// Query heads are assigned to KV heads in contiguous groups. For example,
/// with eight query heads and two KV heads, query heads 0..4 use KV head 0
/// and query heads 4..8 use KV head 1.
pub fn gqa_group_size(num_query_heads: usize, num_kv_heads: usize) -> Result<usize, String> {
    if num_query_heads == 0 {
        return Err("num_query_heads must be greater than zero".to_string());
    }
    if num_kv_heads == 0 {
        return Err("num_kv_heads must be greater than zero".to_string());
    }
    if num_query_heads < num_kv_heads {
        return Err(format!(
            "num_query_heads ({num_query_heads}) must be at least num_kv_heads ({num_kv_heads})"
        ));
    }
    if num_query_heads % num_kv_heads != 0 {
        return Err(format!(
            "num_query_heads ({num_query_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        ));
    }
    Ok(num_query_heads / num_kv_heads)
}

/// Map a query-head index to its corresponding key/value-head index.
pub fn kv_head_for_query(
    query_head: usize,
    num_query_heads: usize,
    num_kv_heads: usize,
) -> Result<usize, String> {
    let group_size = gqa_group_size(num_query_heads, num_kv_heads)?;
    if query_head >= num_query_heads {
        return Err(format!(
            "query_head index {query_head} is out of range for {num_query_heads} query heads"
        ));
    }
    Ok(query_head / group_size)
}

/// Align KV heads with query heads for grouped-query attention.
///
/// `kv` is row-major with shape `[rows, num_kv_heads, head_dim]`. The result
/// has shape `[rows, num_query_heads, head_dim]`. No allocation or copying is
/// performed when the query and KV head counts are already equal; in that case
/// the returned `Cow` borrows `kv`.
pub fn repeat_kv_heads<'a>(
    kv: &'a [f32],
    rows: usize,
    num_query_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Cow<'a, [f32]>, String> {
    let group_size = gqa_group_size(num_query_heads, num_kv_heads)?;
    if head_dim == 0 {
        return Err("head_dim must be greater than zero".to_string());
    }

    let expected_len = rows
        .checked_mul(num_kv_heads)
        .and_then(|n| n.checked_mul(head_dim))
        .ok_or_else(|| "KV tensor shape overflows usize".to_string())?;
    if kv.len() != expected_len {
        return Err(format!(
            "KV tensor has {} elements, expected {expected_len} for shape [{rows}, {num_kv_heads}, {head_dim}]",
            kv.len()
        ));
    }

    if num_query_heads == num_kv_heads {
        return Ok(Cow::Borrowed(kv));
    }

    let output_len = rows
        .checked_mul(num_query_heads)
        .and_then(|n| n.checked_mul(head_dim))
        .ok_or_else(|| "expanded KV tensor shape overflows usize".to_string())?;
    let mut expanded = vec![0.0; output_len];

    let kv_row_len = num_kv_heads * head_dim;
    let query_row_len = num_query_heads * head_dim;
    for row in 0..rows {
        let source_row = &kv[row * kv_row_len..(row + 1) * kv_row_len];
        let destination_row = &mut expanded[row * query_row_len..(row + 1) * query_row_len];
        for kv_head in 0..num_kv_heads {
            let head = &source_row[kv_head * head_dim..(kv_head + 1) * head_dim];
            let first_query_head = kv_head * group_size;
            for query_head in first_query_head..first_query_head + group_size {
                destination_row[query_head * head_dim..(query_head + 1) * head_dim]
                    .copy_from_slice(head);
            }
        }
    }

    Ok(Cow::Owned(expanded))
}

fn validate_rope_lengths(
    tensor_len: usize,
    cos_len: usize,
    sin_len: usize,
    rows: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<(), String> {
    if num_heads == 0 {
        return Err("num_heads must be greater than zero".to_string());
    }
    if head_dim == 0 || head_dim % 2 != 0 {
        return Err(format!(
            "head_dim must be a positive even number, got {head_dim}"
        ));
    }

    let expected_tensor_len = rows
        .checked_mul(num_heads)
        .and_then(|n| n.checked_mul(head_dim))
        .ok_or_else(|| "rotary tensor shape overflows usize".to_string())?;
    if tensor_len != expected_tensor_len {
        return Err(format!(
            "rotary tensor has {tensor_len} elements, expected {expected_tensor_len} for shape [{rows}, {num_heads}, {head_dim}]"
        ));
    }

    let half_dim = head_dim / 2;
    let expected_table_len = rows
        .checked_mul(half_dim)
        .ok_or_else(|| "rotary table shape overflows usize".to_string())?;
    if cos_len != expected_table_len || sin_len != expected_table_len {
        return Err(format!(
            "cos and sin must each contain {expected_table_len} elements for shape [{rows}, {half_dim}], got {cos_len} and {sin_len}"
        ));
    }
    Ok(())
}

fn rotate_in_place(
    tensor: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    rows: usize,
    num_heads: usize,
    head_dim: usize,
    inverse: bool,
) -> Result<(), String> {
    validate_rope_lengths(
        tensor.len(),
        cos.len(),
        sin.len(),
        rows,
        num_heads,
        head_dim,
    )?;

    let half_dim = head_dim / 2;
    let row_tensor_len = num_heads * head_dim;
    for row in 0..rows {
        let row_cos = &cos[row * half_dim..(row + 1) * half_dim];
        let row_sin = &sin[row * half_dim..(row + 1) * half_dim];
        let row_tensor = &mut tensor[row * row_tensor_len..(row + 1) * row_tensor_len];
        for head in row_tensor.chunks_exact_mut(head_dim) {
            for dimension in 0..half_dim {
                let first = head[dimension];
                let second = head[dimension + half_dim];
                let cosine = row_cos[dimension];
                let sine = row_sin[dimension];
                if inverse {
                    head[dimension] = first * cosine + second * sine;
                    head[dimension + half_dim] = second * cosine - first * sine;
                } else {
                    head[dimension] = first * cosine - second * sine;
                    head[dimension + half_dim] = second * cosine + first * sine;
                }
            }
        }
    }
    Ok(())
}

/// Apply standard half-dimension rotary position embeddings.
///
/// `tensor` is row-major with shape `[rows, num_heads, head_dim]`; `cos` and
/// `sin` have shape `[rows, head_dim / 2]`. The same rotary table is applied to
/// every head in a row.
pub fn apply_rotary_embeddings(
    tensor: &[f32],
    cos: &[f32],
    sin: &[f32],
    rows: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>, String> {
    let mut rotated = tensor.to_vec();
    rotate_in_place(&mut rotated, cos, sin, rows, num_heads, head_dim, false)?;
    Ok(rotated)
}

/// Undo rotary position embeddings applied by [`apply_rotary_embeddings`].
pub fn invert_rotary_embeddings(
    tensor: &[f32],
    cos: &[f32],
    sin: &[f32],
    rows: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>, String> {
    let mut restored = tensor.to_vec();
    rotate_in_place(&mut restored, cos, sin, rows, num_heads, head_dim, true)?;
    Ok(restored)
}

/// Build per-token cos/sin tables for NeoX half-split RoPE.
///
/// `positions` holds one i64 position per token; `base` is the theta base
/// (e.g. 10000.0); `rot_dim` is the (even) number of rotated dimensions per
/// head. Tables have shape `[positions.len(), rot_dim / 2]` with
/// `angle(t, i) = position[t] * base^(-2i / rot_dim)`.
pub fn rope_cos_sin_tables(
    positions: &[i64],
    base: f64,
    rot_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    if rot_dim == 0 || rot_dim % 2 != 0 {
        return Err(format!(
            "rot_dim must be a positive even number, got {rot_dim}"
        ));
    }
    if !base.is_finite() || base <= 0.0 {
        return Err(format!(
            "rope theta base must be finite and positive, got {base}"
        ));
    }
    let half_dim = rot_dim / 2;
    let log_base = base.ln();
    let table_len = positions
        .len()
        .checked_mul(half_dim)
        .ok_or_else(|| "rotary table shape overflows usize".to_string())?;
    let mut cos = Vec::with_capacity(table_len);
    let mut sin = Vec::with_capacity(table_len);
    for &position in positions {
        for dimension in 0..half_dim {
            let inv_freq = (-(2.0 * dimension as f64 / rot_dim as f64) * log_base).exp();
            let angle = position as f64 * inv_freq;
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    Ok((cos, sin))
}

#[allow(clippy::too_many_arguments)]
fn rope_with_positions(
    tensor: &[f32],
    positions: &[i64],
    base: f64,
    partial_rotary_factor: f64,
    rows: usize,
    num_heads: usize,
    head_dim: usize,
    inverse: bool,
) -> Result<Vec<f32>, String> {
    if num_heads == 0 {
        return Err("num_heads must be greater than zero".to_string());
    }
    if head_dim == 0 {
        return Err("head_dim must be greater than zero".to_string());
    }
    if !partial_rotary_factor.is_finite()
        || partial_rotary_factor <= 0.0
        || partial_rotary_factor > 1.0
    {
        return Err(format!(
            "partial_rotary_factor must be in (0, 1], got {partial_rotary_factor}"
        ));
    }
    if positions.len() != rows {
        return Err(format!(
            "positions has {} entries, expected one per token ({rows})",
            positions.len()
        ));
    }

    // NeoX half-split: rotate the first rot_dim dimensions of each head,
    // rounded down to an even count; the remaining dimensions pass through.
    let mut rot_dim = (head_dim as f64 * partial_rotary_factor) as usize;
    rot_dim -= rot_dim % 2;
    if rot_dim < 2 {
        return Err(format!(
            "partial_rotary_factor {partial_rotary_factor} leaves rot_dim={rot_dim}; need at least 2 rotated dimensions"
        ));
    }

    let expected_len = rows
        .checked_mul(num_heads)
        .and_then(|n| n.checked_mul(head_dim))
        .ok_or_else(|| "rotary tensor shape overflows usize".to_string())?;
    if tensor.len() != expected_len {
        return Err(format!(
            "rotary tensor has {} elements, expected {expected_len} for shape [{rows}, {num_heads}, {head_dim}]",
            tensor.len()
        ));
    }

    let (cos, sin) = rope_cos_sin_tables(positions, base, rot_dim)?;
    let half_dim = rot_dim / 2;
    let row_tensor_len = num_heads * head_dim;
    let mut rotated = tensor.to_vec();
    for row in 0..rows {
        let row_cos = &cos[row * half_dim..(row + 1) * half_dim];
        let row_sin = &sin[row * half_dim..(row + 1) * half_dim];
        let row_tensor = &mut rotated[row * row_tensor_len..(row + 1) * row_tensor_len];
        for head in row_tensor.chunks_exact_mut(head_dim) {
            for dimension in 0..half_dim {
                let first = head[dimension];
                let second = head[dimension + half_dim];
                let cosine = row_cos[dimension];
                let sine = row_sin[dimension];
                if inverse {
                    head[dimension] = first * cosine + second * sine;
                    head[dimension + half_dim] = second * cosine - first * sine;
                } else {
                    head[dimension] = first * cosine - second * sine;
                    head[dimension + half_dim] = second * cosine + first * sine;
                }
            }
        }
    }
    Ok(rotated)
}

/// Apply NeoX half-split RoPE with per-token positions and partial rotation.
///
/// `tensor` is row-major `[rows, num_heads, head_dim]`; `positions` holds one
/// i64 position per token; `base` is the theta base; `partial_rotary_factor`
/// in (0, 1] selects the fraction of head dimensions to rotate (rounded down
/// to an even count). Unrotated trailing dimensions are copied unchanged.
#[allow(clippy::too_many_arguments)]
pub fn rope_apply(
    tensor: &[f32],
    positions: &[i64],
    base: f64,
    partial_rotary_factor: f64,
    rows: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>, String> {
    rope_with_positions(
        tensor,
        positions,
        base,
        partial_rotary_factor,
        rows,
        num_heads,
        head_dim,
        false,
    )
}

/// Undo RoPE applied by [`rope_apply`] with the same positions/base/factor.
#[allow(clippy::too_many_arguments)]
pub fn rope_inverse(
    tensor: &[f32],
    positions: &[i64],
    base: f64,
    partial_rotary_factor: f64,
    rows: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>, String> {
    rope_with_positions(
        tensor,
        positions,
        base,
        partial_rotary_factor,
        rows,
        num_heads,
        head_dim,
        true,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "element {index}: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn gqa_head_mapping_uses_contiguous_groups() {
        assert_eq!(gqa_group_size(8, 2).unwrap(), 4);
        let mapping: Vec<_> = (0..8)
            .map(|head| kv_head_for_query(head, 8, 2).unwrap())
            .collect();
        assert_eq!(mapping, [0, 0, 0, 0, 1, 1, 1, 1]);
        assert!(gqa_group_size(7, 2).is_err());
        assert!(gqa_group_size(2, 4).is_err());
        assert!(kv_head_for_query(8, 8, 2).is_err());
    }

    #[test]
    fn repeat_kv_heads_repeats_each_head_within_its_row() {
        let kv = [
            1.0, 2.0, 3.0, 4.0, // row 0: two heads
            5.0, 6.0, 7.0, 8.0, // row 1: two heads
        ];
        let expanded = repeat_kv_heads(&kv, 2, 4, 2, 2).unwrap();
        assert_close(
            &expanded,
            &[
                1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0, 5.0, 6.0, 7.0, 8.0, 7.0, 8.0,
            ],
        );
    }

    #[test]
    fn repeat_kv_heads_borrows_when_already_aligned() {
        let values = [1.0, 2.0, 3.0, 4.0];
        let aligned = repeat_kv_heads(&values, 1, 2, 2, 2).unwrap();
        assert!(matches!(aligned, Cow::Borrowed(_)));
        assert_eq!(aligned.as_ptr(), values.as_ptr());
        assert!(repeat_kv_heads(&values, 1, 4, 2, 3).is_err());
        assert!(repeat_kv_heads(&values[..3], 1, 2, 2, 2).is_err());
    }

    #[test]
    fn rotary_embedding_matches_quarter_turn_and_round_trips() {
        let tensor = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let cos = [0.0, 1.0];
        let sin = [1.0, 0.0];
        let rotated = apply_rotary_embeddings(&tensor, &cos, &sin, 1, 2, 4).unwrap();
        assert_close(&rotated, &[-3.0, 2.0, 1.0, 4.0, -7.0, 6.0, 5.0, 8.0]);
        let restored = invert_rotary_embeddings(&rotated, &cos, &sin, 1, 2, 4).unwrap();
        assert_close(&restored, &tensor);
    }

    #[test]
    fn rotary_embedding_uses_a_table_entry_per_row() {
        let tensor = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let cos = [1.0, 0.0];
        let sin = [0.0, 1.0];
        let rotated = apply_rotary_embeddings(&tensor, &cos, &sin, 2, 2, 2).unwrap();
        assert_close(&rotated, &[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, -1.0, 0.0]);
        let restored = invert_rotary_embeddings(&rotated, &cos, &sin, 2, 2, 2).unwrap();
        assert_close(&restored, &tensor);
    }

    #[test]
    fn rotary_embedding_validates_dimensions_and_lengths() {
        let tensor = [0.0; 8];
        let table = [0.0; 2];
        assert!(apply_rotary_embeddings(&tensor, &table, &table, 1, 1, 3).is_err());
        assert!(apply_rotary_embeddings(&tensor, &table, &table, 2, 1, 4).is_err());
        assert!(apply_rotary_embeddings(&tensor, &table[..1], &table, 1, 1, 4).is_err());
        assert!(apply_rotary_embeddings(&tensor, &table, &table, 1, 0, 4).is_err());
    }

    /// R7: apply followed by inverse must be the identity (1e-5), and partial
    /// rotation must leave trailing head dimensions untouched.
    #[test]
    fn rope_apply_inverse_round_trip_is_identity() {
        let (rows, heads, head_dim) = (5, 3, 8);
        let tensor: Vec<f32> = (0..rows * heads * head_dim)
            .map(|i| ((i * 13 + 7) as f32).sin())
            .collect();
        let positions = [0i64, 1, 7, 100, 12345];
        let applied = rope_apply(&tensor, &positions, 10000.0, 0.5, rows, heads, head_dim).unwrap();
        let restored =
            rope_inverse(&applied, &positions, 10000.0, 0.5, rows, heads, head_dim).unwrap();
        for (index, (&actual, &expected)) in restored.iter().zip(&tensor).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "element {index}: {actual} vs {expected}"
            );
        }
        // rot_dim = 8 * 0.5 = 4: dimensions 4..8 pass through unchanged.
        for row in 0..rows {
            for head in 0..heads {
                for dim in 4..head_dim {
                    let index = (row * heads + head) * head_dim + dim;
                    assert_eq!(applied[index], tensor[index]);
                }
            }
        }
    }

    /// R7: hand-computed 2-token, head_dim=4 NeoX example.
    /// Token 0 (position 0) is unchanged; token 1 (position 1) rotates with
    /// inv_freq = [10000^0, 10000^(-2/4)] = [1, 0.01].
    #[test]
    fn rope_apply_matches_hand_computed_two_token_example() {
        let tensor = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let positions = [0i64, 1];
        let out = rope_apply(&tensor, &positions, 10000.0, 1.0, 2, 1, 4).unwrap();
        assert_eq!(&out[..4], &tensor[..4]);

        let c1 = 1.0f64.cos();
        let s1 = 1.0f64.sin();
        let c2 = 0.01f64.cos();
        let s2 = 0.01f64.sin();
        // Token 1 holds [5, 6, 7, 8]; NeoX pairs (5,7) and (6,8).
        let expected = [
            (5.0 * c1 - 7.0 * s1) as f32,
            (6.0 * c2 - 8.0 * s2) as f32,
            (7.0 * c1 + 5.0 * s1) as f32,
            (8.0 * c2 + 6.0 * s2) as f32,
        ];
        for (index, (&actual, &expected)) in out[4..].iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "element {index}: {actual} vs {expected}"
            );
        }
    }

    /// R7: torch-style GQA expansion — query head h maps to KV head h // n_rep.
    #[test]
    fn repeat_kv_heads_maps_query_head_to_kv_head_div_n_rep() {
        let n_rep = 3;
        let n_kv = 2;
        let head_dim = 2;
        let rows = 2;
        let kv: Vec<f32> = (0..rows * n_kv * head_dim).map(|i| i as f32).collect();
        let expanded = repeat_kv_heads(&kv, rows, n_kv * n_rep, n_kv, head_dim).unwrap();
        for row in 0..rows {
            for head in 0..n_kv * n_rep {
                let kv_head = head / n_rep;
                for dim in 0..head_dim {
                    assert_eq!(
                        expanded[(row * n_kv * n_rep + head) * head_dim + dim],
                        kv[(row * n_kv + kv_head) * head_dim + dim]
                    );
                }
            }
        }
    }

    #[test]
    fn rope_with_positions_validates_inputs() {
        let tensor = [0.0f32; 8];
        let positions = [0i64, 1];
        assert!(rope_apply(&tensor, &positions, 10000.0, 0.0, 2, 1, 4).is_err());
        assert!(rope_apply(&tensor, &positions, 10000.0, 1.5, 2, 1, 4).is_err());
        assert!(rope_apply(&tensor, &positions, 0.0, 1.0, 2, 1, 4).is_err());
        assert!(rope_apply(&tensor, &[0], 10000.0, 1.0, 2, 1, 4).is_err());
        // head_dim 4 * 0.2 = 0.8 -> rot_dim 0: rejected.
        assert!(rope_apply(&tensor, &positions, 10000.0, 0.2, 2, 1, 4).is_err());
        assert!(rope_apply(&tensor[..7], &positions, 10000.0, 1.0, 2, 1, 4).is_err());
    }
}
