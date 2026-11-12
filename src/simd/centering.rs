//! Sink-token centering kernels.
//!
//! These are plain scalar loops intended to auto-vectorize (AVX2 / AVX-512 /
//! NEON) under the compiler; no explicit SIMD intrinsics are used.

use half::bf16;

/// Subtract sink-token (first token / row 0) and unpack BF16 to F32 on-the-fly.
///
/// Scalar loops intended to auto-vectorize; no explicit SIMD intrinsics.
pub fn center_and_unpack_bf16(raw_bf16: &[u16], nrows: usize, d_model: usize, out_f32: &mut [f32]) {
    assert!(raw_bf16.len() >= nrows * d_model);
    assert!(out_f32.len() >= nrows * d_model);
    if nrows == 0 || d_model == 0 {
        return;
    }

    // Hoist the sink-row BF16 -> F32 conversion out of the per-row loop:
    // the sink is identical for every row, so convert it exactly once.
    let sink_f32: Vec<f32> = raw_bf16[0..d_model]
        .iter()
        .map(|&bits| bf16::from_bits(bits).to_f32())
        .collect();
    for row in 0..nrows {
        let in_offset = row * d_model;
        let out_offset = row * d_model;
        for col in 0..d_model {
            let val = bf16::from_bits(raw_bf16[in_offset + col]).to_f32();
            out_f32[out_offset + col] = val - sink_f32[col];
        }
    }
}

/// Batched sink-centering and unpacking for 3D inputs [batch, sequence, d_model].
///
/// For each batch item, token 0 of that sequence is subtracted from all tokens in the sequence.
pub fn center_and_unpack_bf16_batched(
    raw_bf16: &[u16],
    batch_size: usize,
    seq_len: usize,
    d_model: usize,
    out_f32: &mut [f32],
) {
    let total_elements = batch_size * seq_len * d_model;
    assert!(raw_bf16.len() >= total_elements);
    assert!(out_f32.len() >= total_elements);
    if batch_size == 0 || seq_len == 0 || d_model == 0 {
        return;
    }

    for b in 0..batch_size {
        let batch_offset = b * seq_len * d_model;
        let sink_offset = batch_offset;
        // Convert this batch's sink row once, not once per sequence row.
        let sink_f32: Vec<f32> = raw_bf16[sink_offset..sink_offset + d_model]
            .iter()
            .map(|&bits| bf16::from_bits(bits).to_f32())
            .collect();

        for s in 0..seq_len {
            let in_pos = batch_offset + s * d_model;
            let out_pos = in_pos;
            for col in 0..d_model {
                let val = bf16::from_bits(raw_bf16[in_pos + col]).to_f32();
                out_f32[out_pos + col] = val - sink_f32[col];
            }
        }
    }
}

/// Subtract sink-token (first row) from all rows in an F32 matrix.
pub fn center_f32(raw_f32: &[f32], nrows: usize, d_model: usize, out_f32: &mut [f32]) {
    assert!(raw_f32.len() >= nrows * d_model);
    assert!(out_f32.len() >= nrows * d_model);
    if nrows == 0 || d_model == 0 {
        return;
    }

    let sink_row = &raw_f32[0..d_model];
    for row in 0..nrows {
        let in_offset = row * d_model;
        let out_offset = row * d_model;
        for col in 0..d_model {
            out_f32[out_offset + col] = raw_f32[in_offset + col] - sink_row[col];
        }
    }
}

/// Batched sink-centering for 3D F32 inputs [batch, sequence, d_model].
pub fn center_f32_batched(
    raw_f32: &[f32],
    batch_size: usize,
    seq_len: usize,
    d_model: usize,
    out_f32: &mut [f32],
) {
    let total_elements = batch_size * seq_len * d_model;
    assert!(raw_f32.len() >= total_elements);
    assert!(out_f32.len() >= total_elements);
    if batch_size == 0 || seq_len == 0 || d_model == 0 {
        return;
    }

    for b in 0..batch_size {
        let batch_offset = b * seq_len * d_model;
        let sink_offset = batch_offset;
        let sink_row = &raw_f32[sink_offset..sink_offset + d_model];

        for s in 0..seq_len {
            let in_pos = batch_offset + s * d_model;
            let out_pos = in_pos;
            for col in 0..d_model {
                out_f32[out_pos + col] = raw_f32[in_pos + col] - sink_row[col];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_center_and_unpack_bf16_subtracts_first_row() {
        // First row: [1.0, 2.0] in bf16
        // Second row: [3.0, 5.0] in bf16
        let r0c0 = bf16::from_f32(1.0).to_bits();
        let r0c1 = bf16::from_f32(2.0).to_bits();
        let r1c0 = bf16::from_f32(3.0).to_bits();
        let r1c1 = bf16::from_f32(5.0).to_bits();

        let raw = vec![r0c0, r0c1, r1c0, r1c1];
        let mut out = vec![0.0f32; 4];
        center_and_unpack_bf16(&raw, 2, 2, &mut out);

        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], 2.0); // 3.0 - 1.0
        assert_eq!(out[3], 3.0); // 5.0 - 2.0
    }

    #[test]
    fn test_center_f32_subtracts_sink() {
        let raw = vec![10.0, 20.0, 15.0, 25.0];
        let mut out = vec![0.0f32; 4];
        center_f32(&raw, 2, 2, &mut out);

        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], 5.0);
        assert_eq!(out[3], 5.0);
    }

    /// R9: hoisting the per-batch sink conversion must not change behavior.
    #[test]
    fn test_center_and_unpack_bf16_batched_subtracts_per_batch_sink() {
        let raw: Vec<u16> = vec![
            // batch 0: sink [1, 2], row [3, 5]
            bf16::from_f32(1.0).to_bits(),
            bf16::from_f32(2.0).to_bits(),
            bf16::from_f32(3.0).to_bits(),
            bf16::from_f32(5.0).to_bits(),
            // batch 1: sink [10, 20], row [30, 50]
            bf16::from_f32(10.0).to_bits(),
            bf16::from_f32(20.0).to_bits(),
            bf16::from_f32(30.0).to_bits(),
            bf16::from_f32(50.0).to_bits(),
        ];
        let mut out = vec![0.0f32; 8];
        center_and_unpack_bf16_batched(&raw, 2, 2, 2, &mut out);
        assert_eq!(out, vec![0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 20.0, 30.0]);
    }
}
