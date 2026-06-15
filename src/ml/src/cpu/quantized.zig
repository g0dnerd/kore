const std = @import("std");

/// Quantize f32 values to integer type T with the given scale factor.
/// out[i] = clamp(round(input[i] * scale), min_T, max_T)
pub fn quantize(comptime T: type, comptime n: usize, input: *const [n]f32, scale: f32) [n]T {
    const min_val: f32 = @floatFromInt(std.math.minInt(T));
    const max_val: f32 = @floatFromInt(std.math.maxInt(T));
    var out: [n]T = undefined;
    for (0..n) |i| {
        const scaled = input.*[i] * scale;
        const clamped = @min(@max(scaled, min_val), max_val);
        out[i] = @intFromFloat(@round(clamped));
    }
    return out;
}

/// Dequantize integer values back to f32.
/// out[i] = input[i] / scale
pub fn dequantize(comptime T: type, comptime n: usize, input: *const [n]T, scale: f32) [n]f32 {
    var out: [n]f32 = undefined;
    for (0..n) |i| {
        const val: f32 = @floatFromInt(input.*[i]);
        out[i] = val / scale;
    }
    return out;
}

/// Quantized matrix multiply with i32 accumulation.
/// A[M,K] @ B[K,N] -> C[M,N]
/// Loop order (i,k,j): the inner j-loop is SIMD-vectorized when N is a
/// multiple of the platform i32 vector width, which it is for the typical
/// NNUE shapes (N=32). Falls back to scalar otherwise.
pub fn matmul(
    comptime T: type,
    comptime M: usize,
    comptime K: usize,
    comptime N: usize,
    a: *const [M * K]T,
    b: *const [K * N]T,
) [M * N]i32 {
    var out: [M * N]i32 = [_]i32{0} ** (M * N);

    const lanes = comptime std.simd.suggestVectorLength(i32) orelse 1;
    if (comptime lanes > 1 and N % lanes == 0) {
        const VecAcc = @Vector(lanes, i32);
        const VecT = @Vector(lanes, T);
        const groups = N / lanes;

        for (0..M) |i| {
            var acc: [groups]VecAcc = undefined;
            inline for (0..groups) |g| acc[g] = @splat(0);

            for (0..K) |k| {
                const a_val: i32 = @intCast(a.*[i * K + k]);
                const av: VecAcc = @splat(a_val);
                const b_row_off = k * N;
                inline for (0..groups) |g| {
                    const b_chunk: VecT = b.*[b_row_off + g * lanes ..][0..lanes].*;
                    const b_wide: VecAcc = b_chunk;
                    acc[g] += av * b_wide;
                }
            }

            inline for (0..groups) |g| {
                out[i * N + g * lanes ..][0..lanes].* = acc[g];
            }
        }
        return out;
    }

    for (0..M) |i| {
        for (0..K) |k| {
            const a_val: i32 = @intCast(a.*[i * K + k]);
            for (0..N) |j| {
                const b_val: i32 = @intCast(b.*[k * N + j]);
                out[i * N + j] += a_val * b_val;
            }
        }
    }
    return out;
}

const vec_len_i8 = std.simd.suggestVectorLength(i8) orelse 16;
const VecI8 = @Vector(vec_len_i8, i8);
const VecI16w = @Vector(vec_len_i8, i16);
const VecI32w = @Vector(vec_len_i8, i32);

/// SIMD i8 dot product with i32 accumulation.
/// Sign-extends i8 -> i16 -> i32 lane-wise; LLVM lowers the paired
/// (sext i16) * (sext i16) -> i32 pattern to vpmaddwd on AVX2 and smlal on NEON.
pub fn dotProduct_i8(comptime n: usize, a: *const [n]i8, b: *const [n]i8) i32 {
    const a_s: []const i8 = a;
    const b_s: []const i8 = b;

    var acc: VecI32w = @splat(0);
    const full = (n / vec_len_i8) * vec_len_i8;

    var i: usize = 0;
    while (i < full) : (i += vec_len_i8) {
        const va_i8: VecI8 = a_s[i..][0..vec_len_i8].*;
        const vb_i8: VecI8 = b_s[i..][0..vec_len_i8].*;
        const va_i16: VecI16w = @intCast(va_i8);
        const vb_i16: VecI16w = @intCast(vb_i8);
        const va_i32: VecI32w = @intCast(va_i16);
        const vb_i32: VecI32w = @intCast(vb_i16);
        acc += va_i32 * vb_i32;
    }

    var sum: i32 = @reduce(.Add, acc);
    while (i < n) : (i += 1) {
        const av: i32 = @intCast(a_s[i]);
        const bv: i32 = @intCast(b_s[i]);
        sum += av * bv;
    }
    return sum;
}

/// i8 linear forward with i32 accumulation — the NNUE FC1/FC2 kernel.
/// out[j] = bias[j] + sum_k(input[k] * weight[j * in_features + k])
/// Weight layout: row-major [out_features, in_features] so each row is contiguous
/// along the reduction dimension.
pub fn linearForward_i8(
    comptime in_features: usize,
    comptime out_features: usize,
    input: *const [in_features]i8,
    weight: *const [out_features * in_features]i8,
    bias: *const [out_features]i32,
) [out_features]i32 {
    var output: [out_features]i32 = bias.*;
    const w: []const i8 = weight;
    for (0..out_features) |j| {
        output[j] += dotProduct_i8(in_features, input, w[j * in_features ..][0..in_features]);
    }
    return output;
}

const vec_len_i32 = std.simd.suggestVectorLength(i32) orelse 4;
const VecI32 = @Vector(vec_len_i32, i32);
const VecI32_i8 = @Vector(vec_len_i32, i8);
const VecShift = @Vector(vec_len_i32, u5);

/// Fused post-FC activation: out[i] = clamp(input[i] >> shift, 0, 127) narrowed to i8.
/// Replaces the two-step (scale to i16, then clippedRelu to i8) used after each
/// quantized FC layer in NNUE inference. Equivalent to clamp(@divTrunc(x, 2^shift), 0, 127)
/// as i8 for this clamp range: negative values always clamp to 0 regardless of whether
/// `>>` or `@divTrunc` is used, so `>>` is safe and avoids the compensated-divide sequence.
pub fn shiftClippedRelu_i8(
    comptime n: usize,
    comptime shift: u5,
    input: *const [n]i32,
) [n]i8 {
    var out: [n]i8 = undefined;
    const in_s: []const i32 = input;
    var out_s: []i8 = &out;

    const zero: VecI32 = @splat(0);
    const max_v: VecI32 = @splat(127);
    const shift_v: VecShift = @splat(shift);

    const full = (n / vec_len_i32) * vec_len_i32;
    var i: usize = 0;
    while (i < full) : (i += vec_len_i32) {
        const v: VecI32 = in_s[i..][0..vec_len_i32].*;
        const shifted: VecI32 = v >> shift_v;
        const clamped: VecI32 = @min(@max(shifted, zero), max_v);
        const narrow: VecI32_i8 = @intCast(clamped);
        out_s[i..][0..vec_len_i32].* = narrow;
    }
    while (i < n) : (i += 1) {
        const shifted = in_s[i] >> shift;
        const clamped = @min(@max(shifted, @as(i32, 0)), @as(i32, 127));
        out_s[i] = @intCast(clamped);
    }
    return out;
}

test "quantize and dequantize i8 round-trip" {
    const input = [_]f32{ 1.0, -0.5, 0.25, 0.0 };
    const scale: f32 = 127.0;
    const q = quantize(i8, 4, &input, scale);
    const dq = dequantize(i8, 4, &q, scale);

    // round(1.0 * 127) = 127 -> 127/127 = 1.0
    try std.testing.expectEqual(@as(i8, 127), q[0]);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), dq[0], 0.01);

    // round(-0.5 * 127) = -64 -> -64/127 ≈ -0.504
    try std.testing.expectEqual(@as(i8, -64), q[1]);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), dq[1], 0.01);

    // round(0.25 * 127) = 32 -> 32/127 ≈ 0.252
    try std.testing.expectEqual(@as(i8, 32), q[2]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), dq[2], 0.01);

    // 0.0 -> 0
    try std.testing.expectEqual(@as(i8, 0), q[3]);
    try std.testing.expectEqual(@as(f32, 0), dq[3]);
}

test "quantize i16 round-trip" {
    const input = [_]f32{ 1.0, -1.0, 0.5 };
    const scale: f32 = 32767.0;
    const q = quantize(i16, 3, &input, scale);
    const dq = dequantize(i16, 3, &q, scale);

    try std.testing.expectEqual(@as(i16, 32767), q[0]);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), dq[0], 0.001);
    try std.testing.expectEqual(@as(i16, -32767), q[1]);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), dq[1], 0.001);
}

test "quantize saturation" {
    // Value exceeds i8 range: 2.0 * 127 = 254, clamped to 127
    const input = [_]f32{2.0};
    const q = quantize(i8, 1, &input, 127.0);
    try std.testing.expectEqual(@as(i8, 127), q[0]);

    // Negative saturation: -2.0 * 127 = -254, clamped to -128
    const input2 = [_]f32{-2.0};
    const q2 = quantize(i8, 1, &input2, 127.0);
    try std.testing.expectEqual(@as(i8, -128), q2[0]);
}

test "matmul i8" {
    // A [2x3] = {{1, 2, 3}, {4, 5, 6}}
    // B [3x2] = {{7, 8}, {9, 10}, {11, 12}}
    // C [2x2]:
    //   C[0][0] = 1*7 + 2*9 + 3*11 = 58
    //   C[0][1] = 1*8 + 2*10 + 3*12 = 64
    //   C[1][0] = 4*7 + 5*9 + 6*11 = 139
    //   C[1][1] = 4*8 + 5*10 + 6*12 = 154
    const a = [_]i8{ 1, 2, 3, 4, 5, 6 };
    const b = [_]i8{ 7, 8, 9, 10, 11, 12 };
    const c = matmul(i8, 2, 3, 2, &a, &b);

    try std.testing.expectEqual(@as(i32, 58), c[0]);
    try std.testing.expectEqual(@as(i32, 64), c[1]);
    try std.testing.expectEqual(@as(i32, 139), c[2]);
    try std.testing.expectEqual(@as(i32, 154), c[3]);
}

test "dotProduct_i8 small" {
    const a = [_]i8{ 1, 2, 3 };
    const b = [_]i8{ 4, 5, 6 };
    try std.testing.expectEqual(@as(i32, 32), dotProduct_i8(3, &a, &b));
}

test "dotProduct_i8 SIMD-aligned" {
    var a: [64]i8 = undefined;
    var b: [64]i8 = undefined;
    for (0..64) |i| {
        a[i] = @intCast(@mod(i, 16));
        b[i] = 1;
    }
    // (0+1+...+15) repeated 4 times = 120 * 4 = 480
    try std.testing.expectEqual(@as(i32, 480), dotProduct_i8(64, &a, &b));
}

test "dotProduct_i8 saturated lanes" {
    // Exercises the full i8 range; verifies i16 intermediate doesn't overflow.
    const a = [_]i8{ 127, -128, 127, -128 };
    const b = [_]i8{ 127, -128, -128, 127 };
    // 127*127 + 128*128 - 127*128 - 128*127 = 16129 + 16384 - 16256 - 16256 = 1
    try std.testing.expectEqual(@as(i32, 1), dotProduct_i8(4, &a, &b));
}

test "dotProduct_i8 matches scalar matmul" {
    const a = [_]i8{ 3, -7, 11, -2, 5, 120, -64, 8, 1, -1, 42, 0, -9, 17, 33, -50 };
    const b = [_]i8{ -4, 9, 2, 8, -11, 3, 13, -6, 22, -17, 50, -25, 0, 1, -1, 127 };
    // Compare against the existing generic matmul with M=N=1, K=16.
    const ref = matmul(i8, 1, 16, 1, &a, &b);
    try std.testing.expectEqual(ref[0], dotProduct_i8(16, &a, &b));
}

test "linearForward_i8 hand-computed" {
    const input = [_]i8{ 1, 2, 3 };
    // weight row 0 = [1,2,3], row 1 = [4,5,6]
    const weight = [_]i8{ 1, 2, 3, 4, 5, 6 };
    const bias = [_]i32{ 100, -50 };
    // out[0] = 100 + 1 + 4 + 9 = 114
    // out[1] = -50 + 4 + 10 + 18 = -18
    const out = linearForward_i8(3, 2, &input, &weight, &bias);
    try std.testing.expectEqual(@as(i32, 114), out[0]);
    try std.testing.expectEqual(@as(i32, -18), out[1]);
}

test "linearForward_i8 SIMD-aligned shape" {
    // Shape representative of NNUE FC1: 32 -> 16
    var input: [32]i8 = undefined;
    var weight: [16 * 32]i8 = undefined;
    var bias: [16]i32 = undefined;
    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rng = prng.random();
    for (0..32) |i| input[i] = rng.intRangeAtMost(i8, -128, 127);
    for (0..16 * 32) |i| weight[i] = rng.intRangeAtMost(i8, -128, 127);
    for (0..16) |i| bias[i] = rng.intRangeAtMost(i32, -1000, 1000);

    const got = linearForward_i8(32, 16, &input, &weight, &bias);

    // Reference via scalar matmul: (1x32) @ (32x16) needs weight transposed.
    var weight_t: [32 * 16]i8 = undefined;
    for (0..16) |j| for (0..32) |k| {
        weight_t[k * 16 + j] = weight[j * 32 + k];
    };
    const ref = matmul(i8, 1, 32, 16, &input, &weight_t);
    for (0..16) |j| {
        try std.testing.expectEqual(bias[j] + ref[j], got[j]);
    }
}

test "shiftClippedRelu_i8 matches reference clamp" {
    // Cover: negative values, values <127 after shift, values >127 after shift,
    // and an odd tail-element count (33 not a multiple of typical i32 vec widths).
    const n = 33;
    var input: [n]i32 = undefined;
    var prng = std.Random.DefaultPrng.init(0xDEADBEEF);
    const rng = prng.random();
    for (0..n) |i| input[i] = rng.intRangeAtMost(i32, -100_000, 100_000);

    const got = shiftClippedRelu_i8(n, 6, &input);

    for (0..n) |i| {
        const shifted = input[i] >> 6;
        const expected_i32 = @min(@max(shifted, @as(i32, 0)), @as(i32, 127));
        const expected: i8 = @intCast(expected_i32);
        try std.testing.expectEqual(expected, got[i]);
    }
}

test "shiftClippedRelu_i8 equivalent to divTrunc+clamp for clamp range" {
    // Proves the shift-vs-divTrunc substitution is safe for the [0,127] clamp:
    // negatives produce 0 in both paths; positives are identical.
    const n = 32;
    var input: [n]i32 = undefined;
    var prng = std.Random.DefaultPrng.init(0xFEEDFACE);
    const rng = prng.random();
    for (0..n) |i| input[i] = rng.intRangeAtMost(i32, -1_000_000, 1_000_000);

    const got = shiftClippedRelu_i8(n, 6, &input);

    for (0..n) |i| {
        const via_div = @divTrunc(input[i], 64);
        const expected_i32 = @min(@max(via_div, @as(i32, 0)), @as(i32, 127));
        const expected: i8 = @intCast(expected_i32);
        try std.testing.expectEqual(expected, got[i]);
    }
}

test "matmul i16" {
    // Same as above but i16
    const a = [_]i16{ 1, 2, 3, 4, 5, 6 };
    const b = [_]i16{ 7, 8, 9, 10, 11, 12 };
    const c = matmul(i16, 2, 3, 2, &a, &b);

    try std.testing.expectEqual(@as(i32, 58), c[0]);
    try std.testing.expectEqual(@as(i32, 64), c[1]);
    try std.testing.expectEqual(@as(i32, 139), c[2]);
    try std.testing.expectEqual(@as(i32, 154), c[3]);
}
