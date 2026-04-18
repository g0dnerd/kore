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
