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
}
