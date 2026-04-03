const std = @import("std");
const Context = @import("Context.zig");
const Buffer = @import("Buffer.zig");
const Program = @import("Program.zig");

const Ops = @This();

// Forward ops
add_bias_prog: Program,
clipped_relu_prog: Program,
sigmoid_prog: Program,
sparse_accumulate_prog: Program,
concat_prog: Program,
reduce_sum_prog: Program,
squared_sum_prog: Program,
fill_prog: Program,
matmul_prog: Program,

// Backward ops
concat_backward_prog: Program,
matmul_backward_input_prog: Program,
matmul_backward_weight_prog: Program,
add_bias_backward_prog: Program,
clipped_relu_backward_prog: Program,
sigmoid_backward_prog: Program,
sparse_accumulate_backward_prog: Program,
weighted_add_prog: Program,

// Loss & optimizer
mse_loss_prog: Program,
adam_update_prog: Program,

// Quantization
quantize_i8_prog: Program,

pub fn init(ctx: *const Context) Context.Error!Ops {
    // Forward
    var p_add_bias = try Program.create(ctx, @embedFile("kernels/add_bias.cl"), "add_bias");
    errdefer p_add_bias.release();
    var p_clipped_relu = try Program.create(ctx, @embedFile("kernels/clipped_relu.cl"), "clipped_relu");
    errdefer p_clipped_relu.release();
    var p_sigmoid = try Program.create(ctx, @embedFile("kernels/sigmoid.cl"), "sigmoid");
    errdefer p_sigmoid.release();
    var p_sparse = try Program.create(ctx, @embedFile("kernels/sparse_accumulate.cl"), "sparse_accumulate");
    errdefer p_sparse.release();
    var p_concat = try Program.create(ctx, @embedFile("kernels/concat.cl"), "concat");
    errdefer p_concat.release();
    var p_reduce = try Program.create(ctx, @embedFile("kernels/reduce_sum.cl"), "reduce_sum");
    errdefer p_reduce.release();
    var p_sq_sum = try Program.create(ctx, @embedFile("kernels/squared_sum.cl"), "squared_sum");
    errdefer p_sq_sum.release();
    var p_fill = try Program.create(ctx, @embedFile("kernels/fill.cl"), "fill");
    errdefer p_fill.release();
    var p_matmul = try Program.create(ctx, @embedFile("kernels/matmul.cl"), "matmul");
    errdefer p_matmul.release();

    // Backward
    var p_concat_b = try Program.create(ctx, @embedFile("kernels/concat_backward.cl"), "concat_backward");
    errdefer p_concat_b.release();
    var p_mm_bi = try Program.create(ctx, @embedFile("kernels/matmul_backward_input.cl"), "matmul_backward_input");
    errdefer p_mm_bi.release();
    var p_mm_bw = try Program.create(ctx, @embedFile("kernels/matmul_backward_weight.cl"), "matmul_backward_weight");
    errdefer p_mm_bw.release();
    var p_ab_b = try Program.create(ctx, @embedFile("kernels/add_bias_backward.cl"), "add_bias_backward");
    errdefer p_ab_b.release();
    var p_cr_b = try Program.create(ctx, @embedFile("kernels/clipped_relu_backward.cl"), "clipped_relu_backward");
    errdefer p_cr_b.release();
    var p_sig_b = try Program.create(ctx, @embedFile("kernels/sigmoid_backward.cl"), "sigmoid_backward");
    errdefer p_sig_b.release();
    var p_sp_b = try Program.create(ctx, @embedFile("kernels/sparse_accumulate_backward.cl"), "sparse_accumulate_backward");
    errdefer p_sp_b.release();
    var p_wa = try Program.create(ctx, @embedFile("kernels/weighted_add.cl"), "weighted_add");
    errdefer p_wa.release();
    var p_mse = try Program.create(ctx, @embedFile("kernels/mse_loss.cl"), "mse_loss");
    errdefer p_mse.release();
    var p_adam = try Program.create(ctx, @embedFile("kernels/adam_update.cl"), "adam_update");
    errdefer p_adam.release();
    var p_quant_i8 = try Program.create(ctx, @embedFile("kernels/quantize.cl"), "quantize_i8");
    errdefer p_quant_i8.release();

    return .{
        .add_bias_prog = p_add_bias,
        .clipped_relu_prog = p_clipped_relu,
        .sigmoid_prog = p_sigmoid,
        .sparse_accumulate_prog = p_sparse,
        .concat_prog = p_concat,
        .reduce_sum_prog = p_reduce,
        .squared_sum_prog = p_sq_sum,
        .fill_prog = p_fill,
        .matmul_prog = p_matmul,
        .concat_backward_prog = p_concat_b,
        .matmul_backward_input_prog = p_mm_bi,
        .matmul_backward_weight_prog = p_mm_bw,
        .add_bias_backward_prog = p_ab_b,
        .clipped_relu_backward_prog = p_cr_b,
        .sigmoid_backward_prog = p_sig_b,
        .sparse_accumulate_backward_prog = p_sp_b,
        .weighted_add_prog = p_wa,
        .mse_loss_prog = p_mse,
        .adam_update_prog = p_adam,
        .quantize_i8_prog = p_quant_i8,
    };
}

pub fn deinit(self: *Ops) void {
    self.add_bias_prog.release();
    self.clipped_relu_prog.release();
    self.sigmoid_prog.release();
    self.sparse_accumulate_prog.release();
    self.concat_prog.release();
    self.reduce_sum_prog.release();
    self.squared_sum_prog.release();
    self.fill_prog.release();
    self.matmul_prog.release();
    self.concat_backward_prog.release();
    self.matmul_backward_input_prog.release();
    self.matmul_backward_weight_prog.release();
    self.add_bias_backward_prog.release();
    self.clipped_relu_backward_prog.release();
    self.sigmoid_backward_prog.release();
    self.sparse_accumulate_backward_prog.release();
    self.weighted_add_prog.release();
    self.mse_loss_prog.release();
    self.adam_update_prog.release();
    self.quantize_i8_prog.release();
    self.* = undefined;
}

fn roundUp(n: usize, multiple: usize) usize {
    return ((n + multiple - 1) / multiple) * multiple;
}

/// In-place add bias: output[i] += bias[i % cols]
pub fn addBias(self: *const Ops, ctx: *const Context, output: Buffer, bias: Buffer, rows: u32, cols: u32) Context.Error!void {
    try self.add_bias_prog.setArg(0, output);
    try self.add_bias_prog.setArg(1, bias);
    try self.add_bias_prog.setArg(2, rows);
    try self.add_bias_prog.setArg(3, cols);
    const n: usize = @as(usize, rows) * @as(usize, cols);
    try self.add_bias_prog.dispatch(ctx, &.{n}, null);
}

/// Element-wise clipped ReLU: output[i] = clamp(input[i], 0, max_val)
pub fn clippedRelu(self: *const Ops, ctx: *const Context, input: Buffer, output: Buffer, max_val: f32, n: u32) Context.Error!void {
    try self.clipped_relu_prog.setArg(0, input);
    try self.clipped_relu_prog.setArg(1, output);
    try self.clipped_relu_prog.setArg(2, max_val);
    try self.clipped_relu_prog.setArg(3, n);
    try self.clipped_relu_prog.dispatch(ctx, &.{@as(usize, n)}, null);
}

/// Element-wise sigmoid: output[i] = 1 / (1 + exp(-input[i]))
pub fn sigmoid(self: *const Ops, ctx: *const Context, input: Buffer, output: Buffer, n: u32) Context.Error!void {
    try self.sigmoid_prog.setArg(0, input);
    try self.sigmoid_prog.setArg(1, output);
    try self.sigmoid_prog.setArg(2, n);
    try self.sigmoid_prog.dispatch(ctx, &.{@as(usize, n)}, null);
}

/// Sparse accumulate: out[b][j] = bias[j] + sum weights[active[b][i]][j]
pub fn sparseAccumulate(
    self: *const Ops,
    ctx: *const Context,
    weights: Buffer,
    bias: Buffer,
    active_indices: Buffer,
    num_active: Buffer,
    output: Buffer,
    batch_size: u32,
    max_active: u32,
    out_features: u32,
) Context.Error!void {
    try self.sparse_accumulate_prog.setArg(0, weights);
    try self.sparse_accumulate_prog.setArg(1, bias);
    try self.sparse_accumulate_prog.setArg(2, active_indices);
    try self.sparse_accumulate_prog.setArg(3, num_active);
    try self.sparse_accumulate_prog.setArg(4, output);
    try self.sparse_accumulate_prog.setArg(5, batch_size);
    try self.sparse_accumulate_prog.setArg(6, max_active);
    try self.sparse_accumulate_prog.setArg(7, out_features);
    const ts: usize = 16;
    try self.sparse_accumulate_prog.dispatch(ctx, &.{
        roundUp(@as(usize, batch_size), ts),
        roundUp(@as(usize, out_features), ts),
    }, &.{ ts, ts });
}

/// Concatenate two matrices along the last axis.
pub fn concat(
    self: *const Ops,
    ctx: *const Context,
    a: Buffer,
    b: Buffer,
    output: Buffer,
    rows: u32,
    cols_a: u32,
    cols_b: u32,
) Context.Error!void {
    try self.concat_prog.setArg(0, a);
    try self.concat_prog.setArg(1, b);
    try self.concat_prog.setArg(2, output);
    try self.concat_prog.setArg(3, rows);
    try self.concat_prog.setArg(4, cols_a);
    try self.concat_prog.setArg(5, cols_b);
    const ts: usize = 16;
    try self.concat_prog.dispatch(ctx, &.{
        roundUp(@as(usize, rows), ts),
        roundUp(@as(usize, cols_a) + @as(usize, cols_b), ts),
    }, &.{ ts, ts });
}

/// Split gradient along last axis back to the two concat inputs.
pub fn concatBackward(
    self: *const Ops,
    ctx: *const Context,
    grad_output: Buffer,
    grad_a: Buffer,
    grad_b: Buffer,
    rows: u32,
    cols_a: u32,
    cols_b: u32,
) Context.Error!void {
    try self.concat_backward_prog.setArg(0, grad_output);
    try self.concat_backward_prog.setArg(1, grad_a);
    try self.concat_backward_prog.setArg(2, grad_b);
    try self.concat_backward_prog.setArg(3, rows);
    try self.concat_backward_prog.setArg(4, cols_a);
    try self.concat_backward_prog.setArg(5, cols_b);
    const ts: usize = 16;
    try self.concat_backward_prog.dispatch(ctx, &.{
        roundUp(@as(usize, rows), ts),
        roundUp(@as(usize, cols_a) + @as(usize, cols_b), ts),
    }, &.{ ts, ts });
}

const reduce_wg_size: usize = 256;

/// Sum all elements of a buffer. Returns scalar result.
pub fn reduceSum(self: *const Ops, ctx: *const Context, input: Buffer, allocator: std.mem.Allocator) !f32 {
    const n = input.len;
    if (n == 0) return 0;
    const global = roundUp(n, reduce_wg_size);
    const num_groups = global / reduce_wg_size;

    var partial = try Buffer.alloc(ctx, num_groups);
    defer partial.release();

    try self.reduce_sum_prog.setArg(0, input);
    try self.reduce_sum_prog.setArg(1, partial);
    try self.reduce_sum_prog.setArgLocal(2, reduce_wg_size * @sizeOf(f32));
    try self.reduce_sum_prog.setArg(3, @as(u32, @intCast(n)));
    try self.reduce_sum_prog.dispatch(ctx, &.{global}, &.{reduce_wg_size});

    const partials = try allocator.alloc(f32, num_groups);
    defer allocator.free(partials);
    try partial.download(ctx, partials);

    var sum: f32 = 0;
    for (partials) |v| sum += v;
    return sum;
}

/// Sum of squared elements of a buffer. Returns scalar result.
pub fn squaredSum(self: *const Ops, ctx: *const Context, input: Buffer, allocator: std.mem.Allocator) !f32 {
    const n = input.len;
    if (n == 0) return 0;
    const global = roundUp(n, reduce_wg_size);
    const num_groups = global / reduce_wg_size;

    var partial = try Buffer.alloc(ctx, num_groups);
    defer partial.release();

    try self.squared_sum_prog.setArg(0, input);
    try self.squared_sum_prog.setArg(1, partial);
    try self.squared_sum_prog.setArgLocal(2, reduce_wg_size * @sizeOf(f32));
    try self.squared_sum_prog.setArg(3, @as(u32, @intCast(n)));
    try self.squared_sum_prog.dispatch(ctx, &.{global}, &.{reduce_wg_size});

    const partials = try allocator.alloc(f32, num_groups);
    defer allocator.free(partials);
    try partial.download(ctx, partials);

    var sum: f32 = 0;
    for (partials) |v| sum += v;
    return sum;
}

/// Fill buffer with constant value.
pub fn fill(self: *const Ops, ctx: *const Context, buf: Buffer, value: f32) Context.Error!void {
    try self.fill_prog.setArg(0, buf);
    try self.fill_prog.setArg(1, value);
    try self.fill_prog.setArg(2, @as(u32, @intCast(buf.len)));
    try self.fill_prog.dispatch(ctx, &.{buf.len}, null);
}

/// Matrix multiply: C[MxN] = A[MxK] * B[KxN]
pub fn matmul(
    self: *const Ops,
    ctx: *const Context,
    a: Buffer,
    b: Buffer,
    c: Buffer,
    m: u32,
    n: u32,
    k: u32,
) Context.Error!void {
    try self.matmul_prog.setArg(0, a);
    try self.matmul_prog.setArg(1, b);
    try self.matmul_prog.setArg(2, c);
    try self.matmul_prog.setArg(3, m);
    try self.matmul_prog.setArg(4, n);
    try self.matmul_prog.setArg(5, k);
    const ts: usize = 16;
    try self.matmul_prog.dispatch(ctx, &.{
        roundUp(@as(usize, m), ts),
        roundUp(@as(usize, n), ts),
    }, &.{ ts, ts });
}

/// grad_input[M,K] += grad_output[M,N] * weight^T[N,K]
pub fn matmulBackwardInput(
    self: *const Ops,
    ctx: *const Context,
    grad_output: Buffer,
    weight: Buffer,
    grad_input: Buffer,
    m: u32,
    n: u32,
    k: u32,
) Context.Error!void {
    try self.matmul_backward_input_prog.setArg(0, grad_output);
    try self.matmul_backward_input_prog.setArg(1, weight);
    try self.matmul_backward_input_prog.setArg(2, grad_input);
    try self.matmul_backward_input_prog.setArg(3, m);
    try self.matmul_backward_input_prog.setArg(4, n);
    try self.matmul_backward_input_prog.setArg(5, k);
    const ts: usize = 16;
    try self.matmul_backward_input_prog.dispatch(ctx, &.{
        roundUp(@as(usize, m), ts),
        roundUp(@as(usize, k), ts),
    }, &.{ ts, ts });
}

/// grad_weight[K,N] += input^T[K,M] * grad_output[M,N]
pub fn matmulBackwardWeight(
    self: *const Ops,
    ctx: *const Context,
    input: Buffer,
    grad_output: Buffer,
    grad_weight: Buffer,
    m: u32,
    n: u32,
    k: u32,
) Context.Error!void {
    try self.matmul_backward_weight_prog.setArg(0, input);
    try self.matmul_backward_weight_prog.setArg(1, grad_output);
    try self.matmul_backward_weight_prog.setArg(2, grad_weight);
    try self.matmul_backward_weight_prog.setArg(3, m);
    try self.matmul_backward_weight_prog.setArg(4, n);
    try self.matmul_backward_weight_prog.setArg(5, k);
    const ts: usize = 16;
    try self.matmul_backward_weight_prog.dispatch(ctx, &.{
        roundUp(@as(usize, k), ts),
        roundUp(@as(usize, n), ts),
    }, &.{ ts, ts });
}

/// grad_bias[j] += sum_i grad_output[i * cols + j]
pub fn addBiasBackward(
    self: *const Ops,
    ctx: *const Context,
    grad_output: Buffer,
    grad_bias: Buffer,
    rows: u32,
    cols: u32,
) Context.Error!void {
    try self.add_bias_backward_prog.setArg(0, grad_output);
    try self.add_bias_backward_prog.setArg(1, grad_bias);
    try self.add_bias_backward_prog.setArg(2, rows);
    try self.add_bias_backward_prog.setArg(3, cols);
    try self.add_bias_backward_prog.dispatch(ctx, &.{@as(usize, cols)}, null);
}

/// grad_input[i] += grad_output[i] * (0 < input[i] < max_val ? 1 : 0)
pub fn clippedReluBackward(
    self: *const Ops,
    ctx: *const Context,
    input: Buffer,
    grad_output: Buffer,
    grad_input: Buffer,
    max_val: f32,
    n: u32,
) Context.Error!void {
    try self.clipped_relu_backward_prog.setArg(0, input);
    try self.clipped_relu_backward_prog.setArg(1, grad_output);
    try self.clipped_relu_backward_prog.setArg(2, grad_input);
    try self.clipped_relu_backward_prog.setArg(3, max_val);
    try self.clipped_relu_backward_prog.setArg(4, n);
    try self.clipped_relu_backward_prog.dispatch(ctx, &.{@as(usize, n)}, null);
}

/// grad_input[i] += grad_output[i] * s[i] * (1 - s[i])
pub fn sigmoidBackward(
    self: *const Ops,
    ctx: *const Context,
    sigmoid_output: Buffer,
    grad_output: Buffer,
    grad_input: Buffer,
    n: u32,
) Context.Error!void {
    try self.sigmoid_backward_prog.setArg(0, sigmoid_output);
    try self.sigmoid_backward_prog.setArg(1, grad_output);
    try self.sigmoid_backward_prog.setArg(2, grad_input);
    try self.sigmoid_backward_prog.setArg(3, n);
    try self.sigmoid_backward_prog.dispatch(ctx, &.{@as(usize, n)}, null);
}

/// Scatter-add: grad_weights[active[b][i]][j] += grad_output[b][j]
pub fn sparseAccumulateBackward(
    self: *const Ops,
    ctx: *const Context,
    grad_weights: Buffer,
    grad_output: Buffer,
    active_indices: Buffer,
    num_active: Buffer,
    batch_size: u32,
    max_active: u32,
    out_features: u32,
) Context.Error!void {
    try self.sparse_accumulate_backward_prog.setArg(0, grad_weights);
    try self.sparse_accumulate_backward_prog.setArg(1, grad_output);
    try self.sparse_accumulate_backward_prog.setArg(2, active_indices);
    try self.sparse_accumulate_backward_prog.setArg(3, num_active);
    try self.sparse_accumulate_backward_prog.setArg(4, batch_size);
    try self.sparse_accumulate_backward_prog.setArg(5, max_active);
    try self.sparse_accumulate_backward_prog.setArg(6, out_features);
    const ts: usize = 16;
    try self.sparse_accumulate_backward_prog.dispatch(ctx, &.{
        roundUp(@as(usize, batch_size), ts),
        roundUp(@as(usize, out_features), ts),
    }, &.{ ts, ts });
}

/// out[i] = a[i] * alpha + b[i] * beta
pub fn weightedAdd(self: *const Ops, ctx: *const Context, a: Buffer, b: Buffer, out: Buffer, alpha: f32, beta: f32, n: u32) Context.Error!void {
    try self.weighted_add_prog.setArg(0, a);
    try self.weighted_add_prog.setArg(1, b);
    try self.weighted_add_prog.setArg(2, out);
    try self.weighted_add_prog.setArg(3, alpha);
    try self.weighted_add_prog.setArg(4, beta);
    try self.weighted_add_prog.setArg(5, n);
    try self.weighted_add_prog.dispatch(ctx, &.{@as(usize, n)}, null);
}

/// Compute per-element MSE loss and gradient.
/// scale > 0 applies sigmoid pre-transform; scale == 0 means identity.
pub fn mseLoss(
    self: *const Ops,
    ctx: *const Context,
    pred: Buffer,
    target: Buffer,
    element_loss: Buffer,
    grad: Buffer,
    n: u32,
    scale: f32,
    inv_n: f32,
) Context.Error!void {
    try self.mse_loss_prog.setArg(0, pred);
    try self.mse_loss_prog.setArg(1, target);
    try self.mse_loss_prog.setArg(2, element_loss);
    try self.mse_loss_prog.setArg(3, grad);
    try self.mse_loss_prog.setArg(4, n);
    try self.mse_loss_prog.setArg(5, scale);
    try self.mse_loss_prog.setArg(6, inv_n);
    try self.mse_loss_prog.dispatch(ctx, &.{@as(usize, n)}, null);
}

/// In-place Adam parameter update.
pub fn adamUpdate(
    self: *const Ops,
    ctx: *const Context,
    param: Buffer,
    grad: Buffer,
    m: Buffer,
    v: Buffer,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    beta1_t: f32,
    beta2_t: f32,
    n: u32,
) Context.Error!void {
    try self.adam_update_prog.setArg(0, param);
    try self.adam_update_prog.setArg(1, grad);
    try self.adam_update_prog.setArg(2, m);
    try self.adam_update_prog.setArg(3, v);
    try self.adam_update_prog.setArg(4, lr);
    try self.adam_update_prog.setArg(5, beta1);
    try self.adam_update_prog.setArg(6, beta2);
    try self.adam_update_prog.setArg(7, eps);
    try self.adam_update_prog.setArg(8, beta1_t);
    try self.adam_update_prog.setArg(9, beta2_t);
    try self.adam_update_prog.setArg(10, n);
    try self.adam_update_prog.dispatch(ctx, &.{@as(usize, n)}, null);
}

/// GPU-accelerated quantization: float32 -> int8 with scale and clamping.
/// Allocates a temporary byte buffer on GPU, runs the kernel, and downloads results.
pub fn quantizeI8(self: *const Ops, ctx: *const Context, input: Buffer, output: []i8, scale: f32) Context.Error!void {
    const n: u32 = @intCast(output.len);

    // Allocate byte buffer on GPU
    var err: cl.cl_int = undefined;
    const out_mem = cl.clCreateBuffer(
        ctx.context,
        cl.CL_MEM_WRITE_ONLY,
        @as(usize, n),
        null,
        &err,
    );
    try Context.check(err);
    defer _ = cl.clReleaseMemObject(out_mem);

    try self.quantize_i8_prog.setArg(0, input);
    try self.quantize_i8_prog.setArg(1, out_mem);
    try self.quantize_i8_prog.setArg(2, scale);
    try self.quantize_i8_prog.setArg(3, n);
    try self.quantize_i8_prog.dispatch(ctx, &.{@as(usize, n)}, null);

    // Download quantized bytes
    try Context.check(cl.clEnqueueReadBuffer(
        ctx.queue,
        out_mem,
        cl.CL_TRUE,
        0,
        @as(usize, n),
        @ptrCast(output.ptr),
        0,
        null,
        null,
    ));
}

const cl = Context.cl;

fn uploadU32(ctx: *const Context, data: []const u32) Context.Error!Buffer {
    var err: cl.cl_int = undefined;
    const byte_size = data.len * @sizeOf(u32);
    const mem_obj = cl.clCreateBuffer(
        ctx.context,
        cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,
        byte_size,
        @ptrCast(@constCast(data.ptr)),
        &err,
    );
    try Context.check(err);
    return .{ .mem = mem_obj, .len = data.len };
}

test "add_bias: GPU matches CPU" {
    var ctx = try Context.init();
    defer ctx.deinit();
    var ops = try Ops.init(&ctx);
    defer ops.deinit();

    const rows: u32 = 2;
    const cols: u32 = 3;
    const input_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const bias_data = [_]f32{ 0.1, 0.2, 0.3 };

    var buf = try Buffer.upload(&ctx, &input_data);
    defer buf.release();
    var bias_buf = try Buffer.upload(&ctx, &bias_data);
    defer bias_buf.release();

    try ops.addBias(&ctx, buf, bias_buf, rows, cols);

    var result: [6]f32 = undefined;
    try buf.download(&ctx, &result);

    const expected = [_]f32{ 1.1, 2.2, 3.3, 4.1, 5.2, 6.3 };
    for (0..6) |i| {
        try std.testing.expectApproxEqAbs(expected[i], result[i], 1e-6);
    }
}

test "clipped_relu: GPU matches CPU" {
    var ctx = try Context.init();
    defer ctx.deinit();
    var ops = try Ops.init(&ctx);
    defer ops.deinit();

    const input = [_]f32{ -2, -0.5, 0, 0.5, 1, 3, 6, 10 };
    var in_buf = try Buffer.upload(&ctx, &input);
    defer in_buf.release();
    var out_buf = try Buffer.alloc(&ctx, 8);
    defer out_buf.release();

    try ops.clippedRelu(&ctx, in_buf, out_buf, 6.0, 8);

    var result: [8]f32 = undefined;
    try out_buf.download(&ctx, &result);

    const expected = [_]f32{ 0, 0, 0, 0.5, 1, 3, 6, 6 };
    for (0..8) |i| {
        try std.testing.expectApproxEqAbs(expected[i], result[i], 1e-6);
    }
}

test "sigmoid: GPU matches CPU" {
    var ctx = try Context.init();
    defer ctx.deinit();
    var ops = try Ops.init(&ctx);
    defer ops.deinit();

    const input = [_]f32{ 0, 1, -1, 10, -10 };
    var in_buf = try Buffer.upload(&ctx, &input);
    defer in_buf.release();
    var out_buf = try Buffer.alloc(&ctx, 5);
    defer out_buf.release();

    try ops.sigmoid(&ctx, in_buf, out_buf, 5);

    var result: [5]f32 = undefined;
    try out_buf.download(&ctx, &result);

    try std.testing.expectApproxEqAbs(@as(f32, 0.5), result[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7310586), result[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2689414), result[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result[3], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[4], 1e-4);
}

test "sparse_accumulate: GPU matches CPU" {
    var ctx = try Context.init();
    defer ctx.deinit();
    var ops = try Ops.init(&ctx);
    defer ops.deinit();

    const batch_size: u32 = 2;
    const out_features: u32 = 3;
    const max_active: u32 = 2;

    const weights = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const bias = [_]f32{ 0.1, 0.2, 0.3 };
    const indices = [_]u32{ 0, 2, 1, 0 };
    const num_active_data = [_]u32{ 2, 1 };

    var w_buf = try Buffer.upload(&ctx, &weights);
    defer w_buf.release();
    var b_buf = try Buffer.upload(&ctx, &bias);
    defer b_buf.release();
    var idx_buf = try uploadU32(&ctx, &indices);
    defer idx_buf.release();
    var na_buf = try uploadU32(&ctx, &num_active_data);
    defer na_buf.release();
    var out_buf = try Buffer.alloc(&ctx, batch_size * out_features);
    defer out_buf.release();

    try ops.sparseAccumulate(&ctx, w_buf, b_buf, idx_buf, na_buf, out_buf, batch_size, max_active, out_features);

    var result: [6]f32 = undefined;
    try out_buf.download(&ctx, &result);

    const expected = [_]f32{ 8.1, 10.2, 12.3, 4.1, 5.2, 6.3 };
    for (0..6) |i| {
        try std.testing.expectApproxEqAbs(expected[i], result[i], 1e-6);
    }
}

test "concat: GPU matches CPU" {
    var ctx = try Context.init();
    defer ctx.deinit();
    var ops = try Ops.init(&ctx);
    defer ops.deinit();

    const rows: u32 = 2;
    const cols_a: u32 = 2;
    const cols_b: u32 = 3;

    const a_data = [_]f32{ 1, 2, 3, 4 };
    const b_data = [_]f32{ 5, 6, 7, 8, 9, 10 };

    var a_buf = try Buffer.upload(&ctx, &a_data);
    defer a_buf.release();
    var b_buf = try Buffer.upload(&ctx, &b_data);
    defer b_buf.release();
    var out_buf = try Buffer.alloc(&ctx, rows * (cols_a + cols_b));
    defer out_buf.release();

    try ops.concat(&ctx, a_buf, b_buf, out_buf, rows, cols_a, cols_b);

    var result: [10]f32 = undefined;
    try out_buf.download(&ctx, &result);

    const expected = [_]f32{ 1, 2, 5, 6, 7, 3, 4, 8, 9, 10 };
    for (0..10) |i| {
        try std.testing.expectApproxEqAbs(expected[i], result[i], 1e-6);
    }
}

test "reduce_sum: GPU matches CPU" {
    var ctx = try Context.init();
    defer ctx.deinit();
    var ops = try Ops.init(&ctx);
    defer ops.deinit();

    var input: [512]f32 = undefined;
    var cpu_sum: f32 = 0;
    for (0..512) |i| {
        input[i] = @floatFromInt(i + 1);
        cpu_sum += input[i];
    }

    var in_buf = try Buffer.upload(&ctx, &input);
    defer in_buf.release();

    const gpu_sum = try ops.reduceSum(&ctx, in_buf, std.testing.allocator);

    try std.testing.expectApproxEqAbs(cpu_sum, gpu_sum, 1e-6);
}

test "reduce_sum: non-aligned count" {
    var ctx = try Context.init();
    defer ctx.deinit();
    var ops = try Ops.init(&ctx);
    defer ops.deinit();

    var input: [300]f32 = undefined;
    var cpu_sum: f32 = 0;
    for (0..300) |i| {
        input[i] = @floatFromInt(i + 1);
        cpu_sum += input[i];
    }

    var in_buf = try Buffer.upload(&ctx, &input);
    defer in_buf.release();

    const gpu_sum = try ops.reduceSum(&ctx, in_buf, std.testing.allocator);

    try std.testing.expectApproxEqAbs(cpu_sum, gpu_sum, 1e-6);
}

test "matmul_backward_input: GPU matches CPU" {
    var ctx = try Context.init();
    defer ctx.deinit();
    var ops = try Ops.init(&ctx);
    defer ops.deinit();

    // C[2x3] = A[2x4] * B[4x3], so dL/dA[2x4] = dL/dC[2x3] * B^T[3x4]
    const M: u32 = 2;
    const N: u32 = 3;
    const K: u32 = 4;

    const grad_out_data = [_]f32{ 1, 0.5, -1, 2, -0.5, 0.3 };
    const weight_data = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2 };

    var go_buf = try Buffer.upload(&ctx, &grad_out_data);
    defer go_buf.release();
    var w_buf = try Buffer.upload(&ctx, &weight_data);
    defer w_buf.release();
    var gi_buf = try Buffer.alloc(&ctx, M * K);
    defer gi_buf.release();
    try ops.fill(&ctx, gi_buf, 0.0);

    try ops.matmulBackwardInput(&ctx, go_buf, w_buf, gi_buf, M, N, K);

    var result: [M * K]f32 = undefined;
    try gi_buf.download(&ctx, &result);

    // CPU: grad_input[i][k] = sum_j grad_output[i][j] * weight[k][j]
    var expected: [M * K]f32 = undefined;
    for (0..M) |i| {
        for (0..K) |k| {
            var sum: f32 = 0;
            for (0..N) |j| {
                sum += grad_out_data[i * N + j] * weight_data[k * N + j];
            }
            expected[i * K + k] = sum;
        }
    }

    for (0..M * K) |i| {
        try std.testing.expectApproxEqAbs(expected[i], result[i], 1e-4);
    }
}

test "matmul_backward_weight: GPU matches CPU" {
    var ctx = try Context.init();
    defer ctx.deinit();
    var ops = try Ops.init(&ctx);
    defer ops.deinit();

    // C[2x3] = A[2x4] * B[4x3], so dL/dB[4x3] = A^T[4x2] * dL/dC[2x3]
    const M: u32 = 2;
    const N: u32 = 3;
    const K: u32 = 4;

    const input_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const grad_out_data = [_]f32{ 1, 0.5, -1, 2, -0.5, 0.3 };

    var in_buf = try Buffer.upload(&ctx, &input_data);
    defer in_buf.release();
    var go_buf = try Buffer.upload(&ctx, &grad_out_data);
    defer go_buf.release();
    var gw_buf = try Buffer.alloc(&ctx, K * N);
    defer gw_buf.release();
    try ops.fill(&ctx, gw_buf, 0.0);

    try ops.matmulBackwardWeight(&ctx, in_buf, go_buf, gw_buf, M, N, K);

    var result: [K * N]f32 = undefined;
    try gw_buf.download(&ctx, &result);

    // CPU: grad_weight[k][n] = sum_i input[i][k] * grad_output[i][n]
    var expected: [K * N]f32 = undefined;
    for (0..K) |k| {
        for (0..N) |n| {
            var sum: f32 = 0;
            for (0..M) |i| {
                sum += input_data[i * K + k] * grad_out_data[i * N + n];
            }
            expected[k * N + n] = sum;
        }
    }

    for (0..K * N) |i| {
        try std.testing.expectApproxEqAbs(expected[i], result[i], 1e-4);
    }
}

test "add_bias_backward: GPU matches CPU" {
    var ctx = try Context.init();
    defer ctx.deinit();
    var ops = try Ops.init(&ctx);
    defer ops.deinit();

    const rows: u32 = 3;
    const cols: u32 = 4;
    const grad_out_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };

    var go_buf = try Buffer.upload(&ctx, &grad_out_data);
    defer go_buf.release();
    var gb_buf = try Buffer.alloc(&ctx, cols);
    defer gb_buf.release();
    try ops.fill(&ctx, gb_buf, 0.0);

    try ops.addBiasBackward(&ctx, go_buf, gb_buf, rows, cols);

    var result: [4]f32 = undefined;
    try gb_buf.download(&ctx, &result);

    // CPU: grad_bias[j] = sum_i grad_output[i * cols + j]
    const expected = [_]f32{ 1 + 5 + 9, 2 + 6 + 10, 3 + 7 + 11, 4 + 8 + 12 };
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(expected[i], result[i], 1e-5);
    }
}

test "clipped_relu_backward: GPU matches CPU" {
    var ctx = try Context.init();
    defer ctx.deinit();
    var ops = try Ops.init(&ctx);
    defer ops.deinit();

    const input_data = [_]f32{ -1, 0, 0.5, 3, 6, 8 };
    const grad_out_data = [_]f32{ 1, 1, 1, 1, 1, 1 };

    var in_buf = try Buffer.upload(&ctx, &input_data);
    defer in_buf.release();
    var go_buf = try Buffer.upload(&ctx, &grad_out_data);
    defer go_buf.release();
    var gi_buf = try Buffer.alloc(&ctx, 6);
    defer gi_buf.release();
    try ops.fill(&ctx, gi_buf, 0.0);

    try ops.clippedReluBackward(&ctx, in_buf, go_buf, gi_buf, 6.0, 6);

    var result: [6]f32 = undefined;
    try gi_buf.download(&ctx, &result);

    // mask: x > 0 && x < 6 → false, false, true, true, false, false
    const expected = [_]f32{ 0, 0, 1, 1, 0, 0 };
    for (0..6) |i| {
        try std.testing.expectApproxEqAbs(expected[i], result[i], 1e-6);
    }
}

test "sigmoid_backward: GPU matches CPU" {
    var ctx = try Context.init();
    defer ctx.deinit();
    var ops = try Ops.init(&ctx);
    defer ops.deinit();

    const sig_out = [_]f32{ 0.5, 0.7310586, 0.2689414 };
    const grad_out_data = [_]f32{ 1, 2, 0.5 };

    var so_buf = try Buffer.upload(&ctx, &sig_out);
    defer so_buf.release();
    var go_buf = try Buffer.upload(&ctx, &grad_out_data);
    defer go_buf.release();
    var gi_buf = try Buffer.alloc(&ctx, 3);
    defer gi_buf.release();
    try ops.fill(&ctx, gi_buf, 0.0);

    try ops.sigmoidBackward(&ctx, so_buf, go_buf, gi_buf, 3);

    var result: [3]f32 = undefined;
    try gi_buf.download(&ctx, &result);

    // CPU: grad_input = grad_output * s * (1-s)
    for (0..3) |i| {
        const expected = grad_out_data[i] * sig_out[i] * (1.0 - sig_out[i]);
        try std.testing.expectApproxEqAbs(expected, result[i], 1e-5);
    }
}

test "sparse_accumulate_backward: GPU matches CPU" {
    var ctx = try Context.init();
    defer ctx.deinit();
    var ops = try Ops.init(&ctx);
    defer ops.deinit();

    const batch_size: u32 = 2;
    const out_features: u32 = 3;
    const max_active: u32 = 2;
    const in_features: u32 = 4;

    // grad_output[2][3]
    const grad_out_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    // Sample 0: active = [0, 2], count = 2
    // Sample 1: active = [1, _], count = 1
    const indices = [_]u32{ 0, 2, 1, 0 };
    const num_active_data = [_]u32{ 2, 1 };

    var go_buf = try Buffer.upload(&ctx, &grad_out_data);
    defer go_buf.release();
    var idx_buf = try uploadU32(&ctx, &indices);
    defer idx_buf.release();
    var na_buf = try uploadU32(&ctx, &num_active_data);
    defer na_buf.release();
    var gw_buf = try Buffer.alloc(&ctx, in_features * out_features);
    defer gw_buf.release();
    try ops.fill(&ctx, gw_buf, 0.0);

    try ops.sparseAccumulateBackward(&ctx, gw_buf, go_buf, idx_buf, na_buf, batch_size, max_active, out_features);

    var result: [12]f32 = undefined;
    try gw_buf.download(&ctx, &result);

    // CPU reference:
    // Row 0: sample 0 active → += grad_out[0] = [1, 2, 3]
    // Row 1: sample 1 active → += grad_out[1] = [4, 5, 6]
    // Row 2: sample 0 active → += grad_out[0] = [1, 2, 3]
    // Row 3: not active → [0, 0, 0]
    const expected = [_]f32{ 1, 2, 3, 4, 5, 6, 1, 2, 3, 0, 0, 0 };
    for (0..12) |i| {
        try std.testing.expectApproxEqAbs(expected[i], result[i], 1e-5);
    }
}

test "weighted_add: GPU matches CPU" {
    var ctx = try Context.init();
    defer ctx.deinit();
    var ops = try Ops.init(&ctx);
    defer ops.deinit();

    const a_data = [_]f32{ 1, 2, 3, 4 };
    const b_data = [_]f32{ 10, 20, 30, 40 };

    var a_buf = try Buffer.upload(&ctx, &a_data);
    defer a_buf.release();
    var b_buf = try Buffer.upload(&ctx, &b_data);
    defer b_buf.release();
    var out_buf = try Buffer.alloc(&ctx, 4);
    defer out_buf.release();

    try ops.weightedAdd(&ctx, a_buf, b_buf, out_buf, 0.5, 2.0, 4);

    var result: [4]f32 = undefined;
    try out_buf.download(&ctx, &result);

    // out = a * 0.5 + b * 2.0
    const expected = [_]f32{ 0.5 + 20, 1 + 40, 1.5 + 60, 2 + 80 };
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(expected[i], result[i], 1e-5);
    }
}

test "weighted_add: in-place scaling" {
    var ctx = try Context.init();
    defer ctx.deinit();
    var ops = try Ops.init(&ctx);
    defer ops.deinit();

    const data = [_]f32{ 2, 4, 6, 8 };
    var buf = try Buffer.upload(&ctx, &data);
    defer buf.release();

    // In-place scale by 0.5: buf = buf * 0.5 + buf * 0
    try ops.weightedAdd(&ctx, buf, buf, buf, 0.5, 0.0, 4);

    var result: [4]f32 = undefined;
    try buf.download(&ctx, &result);

    const expected = [_]f32{ 1, 2, 3, 4 };
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(expected[i], result[i], 1e-6);
    }
}
