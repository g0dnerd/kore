const std = @import("std");

const vec_len = std.simd.suggestVectorLength(f32) orelse 4;
const Vec = @Vector(vec_len, f32);

/// SIMD dot product of two f32 arrays.
pub fn dotProduct(comptime n: usize, a: *const [n]f32, b: *const [n]f32) f32 {
    const a_s: []const f32 = a;
    const b_s: []const f32 = b;

    var acc: Vec = @splat(0);
    const full = (n / vec_len) * vec_len;

    var i: usize = 0;
    while (i < full) : (i += vec_len) {
        const va: Vec = a_s[i..][0..vec_len].*;
        const vb: Vec = b_s[i..][0..vec_len].*;
        acc += va * vb;
    }

    var sum = @reduce(.Add, acc);
    while (i < n) : (i += 1) {
        sum += a_s[i] * b_s[i];
    }
    return sum;
}

/// SIMD linear forward: out[j] = bias[j] + sum_k(input[k] * weight[j * in + k])
/// Weight is row-major: row j contains the weights for output neuron j.
pub fn linearForward(
    comptime in_features: usize,
    comptime out_features: usize,
    input: *const [in_features]f32,
    weight: *const [out_features * in_features]f32,
    bias: *const [out_features]f32,
) [out_features]f32 {
    var output: [out_features]f32 = bias.*;
    const w: []const f32 = weight;
    for (0..out_features) |j| {
        output[j] += dotProduct(in_features, input, w[j * in_features ..][0..in_features]);
    }
    return output;
}

/// SIMD clipped ReLU: clamp(x, 0, max_val)
pub fn clippedRelu(comptime n: usize, input: *const [n]f32, comptime max_val: f32) [n]f32 {
    const zero: Vec = @splat(0);
    const max_v: Vec = @splat(max_val);

    var out: [n]f32 = undefined;
    const in_s: []const f32 = input;
    var out_s: []f32 = &out;

    const full = (n / vec_len) * vec_len;
    var i: usize = 0;
    while (i < full) : (i += vec_len) {
        const v: Vec = in_s[i..][0..vec_len].*;
        out_s[i..][0..vec_len].* = @min(@max(v, zero), max_v);
    }
    while (i < n) : (i += 1) {
        out[i] = @min(@max(in_s[i], @as(f32, 0)), max_val);
    }
    return out;
}

/// Sigmoid: 1 / (1 + exp(-x))
pub fn sigmoid(comptime n: usize, input: *const [n]f32) [n]f32 {
    var out: [n]f32 = undefined;
    for (0..n) |i| {
        out[i] = 1.0 / (1.0 + @exp(-input.*[i]));
    }
    return out;
}

test "dotProduct" {
    const a = [_]f32{ 1, 2, 3 };
    const b = [_]f32{ 4, 5, 6 };
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    try std.testing.expectApproxEqAbs(@as(f32, 32), dotProduct(3, &a, &b), 1e-6);
}

test "dotProduct SIMD-aligned" {
    // Test with a size that exercises full SIMD vectors
    var a: [16]f32 = undefined;
    var b: [16]f32 = undefined;
    for (0..16) |i| {
        a[i] = @floatFromInt(i + 1);
        b[i] = 1;
    }
    // sum of 1..16 = 136
    try std.testing.expectApproxEqAbs(@as(f32, 136), dotProduct(16, &a, &b), 1e-4);
}

test "linearForward hand-computed" {
    // Input: [3] = {1, 2, 3}
    // Weight: [2x3] = {1, 0, -1,  0, 1, 0} (row-major)
    // Bias: [2] = {0.5, -0.5}
    //
    // out[0] = 0.5 + (1*1 + 2*0 + 3*(-1)) = 0.5 + (1 - 3) = -1.5
    // out[1] = -0.5 + (1*0 + 2*1 + 3*0) = -0.5 + 2 = 1.5
    const input = [_]f32{ 1, 2, 3 };
    const weight = [_]f32{ 1, 0, -1, 0, 1, 0 };
    const bias = [_]f32{ 0.5, -0.5 };

    const out = linearForward(3, 2, &input, &weight, &bias);

    try std.testing.expectApproxEqAbs(@as(f32, -1.5), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), out[1], 1e-6);
}

test "linearForward identity weight" {
    // Identity-like weight selects specific input features
    const input = [_]f32{ 1, 2, 3, 4 };
    const weight = [_]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
    };
    const bias = [_]f32{ 0, 0, 0 };

    const out = linearForward(4, 3, &input, &weight, &bias);

    try std.testing.expectApproxEqAbs(@as(f32, 1), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2), out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3), out[2], 1e-6);
}

test "clippedRelu" {
    const input = [_]f32{ -2, -0.5, 0, 0.5, 1, 3, 6, 10 };
    const out = clippedRelu(8, &input, 6.0);

    try std.testing.expectApproxEqAbs(@as(f32, 0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), out[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), out[4], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3), out[5], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6), out[6], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6), out[7], 1e-6);
}

test "sigmoid" {
    const input = [_]f32{ 0, 1, -1, 10, -10 };
    const out = sigmoid(5, &input);

    // sigmoid(0) = 0.5
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out[0], 1e-6);
    // sigmoid(1) ≈ 0.7310586
    try std.testing.expectApproxEqAbs(@as(f32, 0.7310586), out[1], 1e-5);
    // sigmoid(-1) ≈ 0.2689414
    try std.testing.expectApproxEqAbs(@as(f32, 0.2689414), out[2], 1e-5);
    // sigmoid(10) ≈ 1.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[3], 1e-4);
    // sigmoid(-10) ≈ 0.0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[4], 1e-4);
}
