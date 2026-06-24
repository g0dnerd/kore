const std = @import("std");
const Tensor = @import("Tensor.zig");
const Shape = @import("Shape.zig");
const tape_mod = @import("tape.zig");
const Context = @import("gpu/Context.zig");
const Buffer = @import("gpu/Buffer.zig");
const Ops = @import("gpu/ops.zig");
const Arena = @import("gpu/Arena.zig");

const Graph = @This();

ctx: *const Context,
ops: *const Ops,
tape: tape_mod.Tape,
arena: Arena,
intermediates: std.ArrayList(*Tensor),
allocator: std.mem.Allocator,

pub fn init(allocator: std.mem.Allocator, ctx: *const Context, ops: *const Ops) Graph {
    return .{
        .ctx = ctx,
        .ops = ops,
        .tape = tape_mod.Tape.init(allocator),
        .arena = Arena.init(allocator, ctx),
        .intermediates = .empty,
        .allocator = allocator,
    };
}

pub fn deinit(self: *Graph) void {
    self.reset();
    self.tape.deinit();
    self.arena.deinit();
    self.intermediates.deinit(self.allocator);
    self.* = undefined;
}

/// Free all intermediate tensors and their gradients, reset tape and arena.
pub fn reset(self: *Graph) void {
    for (self.intermediates.items) |t| {
        // Grad tensor struct (buffer is arena-managed or cleaned up below)
        if (t.grad) |g| {
            self.allocator.destroy(g);
        }
        // Tensor struct (buffer is arena-managed)
        self.allocator.destroy(t);
    }
    self.intermediates.clearRetainingCapacity();
    self.tape.reset();
    self.arena.reset();
}

/// Zero the gradient buffers of parameter tensors.
pub fn zeroGrads(self: *Graph, params: []const *Tensor) !void {
    for (params) |p| {
        if (p.grad) |g| {
            try self.ops.fill(self.ctx, g.storage.gpu.buffer, 0.0);
        }
    }
}

/// Set a tensor's gradient to a uniform value (e.g., 1.0 for sum-loss).
pub fn setGrad(self: *Graph, tensor: *Tensor, value: f32) !void {
    const n = tensor.elements();
    const buf = try self.arena.alloc(n);
    try self.ops.fill(self.ctx, buf, value);
    const g = try self.allocator.create(Tensor);
    g.* = .{
        .shape = tensor.shape,
        .storage = .{ .gpu = .{ .buffer = buf } },
        .allocator = self.allocator,
    };
    tensor.grad = g;
}

/// C[M,N] = input[M,K] * weight[K,N]
pub fn matmul(self: *Graph, input: *Tensor, weight: *Tensor) !*Tensor {
    const in_dims = input.shape.dims();
    const w_dims = weight.shape.dims();
    const m: u32 = @intCast(in_dims[0]);
    const k: u32 = @intCast(in_dims[1]);
    const n: u32 = @intCast(w_dims[1]);
    std.debug.assert(k == @as(u32, @intCast(w_dims[0])));

    const out_buf = try self.arena.alloc(@as(usize, m) * @as(usize, n));
    try self.ops.matmul(self.ctx, input.storage.gpu.buffer, weight.storage.gpu.buffer, out_buf, m, n, k);

    var output = try self.createIntermediate(&.{ @as(usize, m), @as(usize, n) }, out_buf);
    output.tape_index = self.tape.entries.items.len;
    try self.tape.append(.{
        .saved = .{ .matmul = .{ .input = input, .weight = weight, .m = m, .n = n, .k = k } },
        .output = output,
    });
    return output;
}

/// output = input + bias (broadcast over rows)
pub fn addBias(self: *Graph, input: *Tensor, bias: *Tensor) !*Tensor {
    const in_dims = input.shape.dims();
    const rows: u32 = @intCast(in_dims[0]);
    const cols: u32 = @intCast(in_dims[1]);
    const n: usize = @as(usize, rows) * @as(usize, cols);

    // Copy input to output buffer, then add bias in-place
    const out_buf = try self.arena.alloc(n);
    try copyBuffer(self.ctx, input.storage.gpu.buffer, out_buf, n);
    try self.ops.addBias(self.ctx, out_buf, bias.storage.gpu.buffer, rows, cols);

    var output = try self.createIntermediate(&.{ @as(usize, rows), @as(usize, cols) }, out_buf);
    output.tape_index = self.tape.entries.items.len;
    try self.tape.append(.{
        .saved = .{ .add_bias = .{ .input = input, .bias = bias, .rows = rows, .cols = cols } },
        .output = output,
    });
    return output;
}

/// output = clamp(input, 0, max_val)
pub fn clippedRelu(self: *Graph, input: *Tensor, max_val: f32) !*Tensor {
    const n: u32 = @intCast(input.elements());
    const out_buf = try self.arena.alloc(@as(usize, n));
    try self.ops.clippedRelu(self.ctx, input.storage.gpu.buffer, out_buf, max_val, n);

    var output = try self.createIntermediate(input.shape.dims(), out_buf);
    output.tape_index = self.tape.entries.items.len;
    try self.tape.append(.{
        .saved = .{ .clipped_relu = .{ .input = input, .max_val = max_val, .n = n } },
        .output = output,
    });
    return output;
}

/// output = clamp(input, 0, max_val)^2
pub fn squaredClippedRelu(self: *Graph, input: *Tensor, max_val: f32) !*Tensor {
    const n: u32 = @intCast(input.elements());
    const out_buf = try self.arena.alloc(@as(usize, n));
    try self.ops.squaredClippedRelu(self.ctx, input.storage.gpu.buffer, out_buf, max_val, n);

    var output = try self.createIntermediate(input.shape.dims(), out_buf);
    output.tape_index = self.tape.entries.items.len;
    try self.tape.append(.{
        .saved = .{ .squared_clipped_relu = .{ .input = input, .max_val = max_val, .n = n } },
        .output = output,
    });
    return output;
}

/// output = 1 / (1 + exp(-input))
pub fn sigmoid(self: *Graph, input: *Tensor) !*Tensor {
    const n: u32 = @intCast(input.elements());
    const out_buf = try self.arena.alloc(@as(usize, n));
    try self.ops.sigmoid(self.ctx, input.storage.gpu.buffer, out_buf, n);

    var output = try self.createIntermediate(input.shape.dims(), out_buf);
    output.tape_index = self.tape.entries.items.len;
    try self.tape.append(.{
        .saved = .{ .sigmoid = .{ .input = input, .output = output, .n = n } },
        .output = output,
    });
    return output;
}

/// Concatenate two tensors along the last axis: [rows, cols_a] ++ [rows, cols_b] -> [rows, cols_a + cols_b]
pub fn concat(self: *Graph, a: *Tensor, b: *Tensor) !*Tensor {
    const a_dims = a.shape.dims();
    const b_dims = b.shape.dims();
    const rows: u32 = @intCast(a_dims[0]);
    std.debug.assert(rows == @as(u32, @intCast(b_dims[0])));
    const cols_a: u32 = @intCast(a_dims[1]);
    const cols_b: u32 = @intCast(b_dims[1]);

    const out_cols: usize = @as(usize, cols_a) + @as(usize, cols_b);
    const out_buf = try self.arena.alloc(@as(usize, rows) * out_cols);
    try self.ops.concat(self.ctx, a.storage.gpu.buffer, b.storage.gpu.buffer, out_buf, rows, cols_a, cols_b);

    var output = try self.createIntermediate(&.{ @as(usize, rows), out_cols }, out_buf);
    output.tape_index = self.tape.entries.items.len;
    try self.tape.append(.{
        .saved = .{ .concat = .{ .input_a = a, .input_b = b, .rows = rows, .cols_a = cols_a, .cols_b = cols_b } },
        .output = output,
    });
    return output;
}

/// out[b][j] = bias[j] + sum weights[active[b][i]][j]
pub fn sparseAccumulate(
    self: *Graph,
    weights: *Tensor,
    bias: *Tensor,
    active_indices: Buffer,
    num_active: Buffer,
    batch_size: u32,
    max_active: u32,
) !*Tensor {
    const out_features: u32 = @intCast(bias.shape.dims()[0]);
    const n: usize = @as(usize, batch_size) * @as(usize, out_features);
    const out_buf = try self.arena.alloc(n);

    try self.ops.sparseAccumulate(
        self.ctx,
        weights.storage.gpu.buffer,
        bias.storage.gpu.buffer,
        active_indices,
        num_active,
        out_buf,
        batch_size,
        max_active,
        out_features,
    );

    var output = try self.createIntermediate(&.{ @as(usize, batch_size), @as(usize, out_features) }, out_buf);
    output.tape_index = self.tape.entries.items.len;
    try self.tape.append(.{
        .saved = .{ .sparse_accumulate = .{
            .weights = weights,
            .bias = bias,
            .active_indices = active_indices,
            .num_active = num_active,
            .batch_size = batch_size,
            .max_active = max_active,
            .out_features = out_features,
        } },
        .output = output,
    });
    return output;
}

/// Run backward pass from the given output tensor (whose .grad must be set).
pub fn backward(self: *Graph, output: *Tensor) !void {
    std.debug.assert(output.grad != null);
    var i = self.tape.entries.items.len;
    while (i > 0) {
        i -= 1;
        try self.backwardEntry(&self.tape.entries.items[i]);
    }
}

fn backwardEntry(self: *Graph, entry: *const tape_mod.TapeEntry) !void {
    const grad_out = entry.output.grad.?.storage.gpu.buffer;

    switch (entry.saved) {
        .matmul => |s| {
            if (needsGrad(s.input)) {
                const gi = try self.ensureGrad(s.input);
                try self.ops.matmulBackwardInput(self.ctx, grad_out, s.weight.storage.gpu.buffer, gi, s.m, s.n, s.k);
            }
            if (s.weight.requires_grad) {
                const gw = try self.ensureGrad(s.weight);
                try self.ops.matmulBackwardWeight(self.ctx, s.input.storage.gpu.buffer, grad_out, gw, s.m, s.n, s.k);
            }
        },
        .add_bias => |s| {
            if (needsGrad(s.input)) {
                const gi = try self.ensureGrad(s.input);
                const n: u32 = s.rows * s.cols;
                // grad_input += grad_output (identity)
                try self.ops.weightedAdd(self.ctx, gi, grad_out, gi, 1.0, 1.0, n);
            }
            if (s.bias.requires_grad) {
                const gb = try self.ensureGrad(s.bias);
                try self.ops.addBiasBackward(self.ctx, grad_out, gb, s.rows, s.cols);
            }
        },
        .clipped_relu => |s| {
            if (needsGrad(s.input)) {
                const gi = try self.ensureGrad(s.input);
                try self.ops.clippedReluBackward(self.ctx, s.input.storage.gpu.buffer, grad_out, gi, s.max_val, s.n);
            }
        },
        .squared_clipped_relu => |s| {
            if (needsGrad(s.input)) {
                const gi = try self.ensureGrad(s.input);
                try self.ops.squaredClippedReluBackward(self.ctx, s.input.storage.gpu.buffer, grad_out, gi, s.max_val, s.n);
            }
        },
        .sigmoid => |s| {
            if (needsGrad(s.input)) {
                const gi = try self.ensureGrad(s.input);
                try self.ops.sigmoidBackward(self.ctx, s.output.storage.gpu.buffer, grad_out, gi, s.n);
            }
        },
        .concat => |s| {
            const ga = if (needsGrad(s.input_a)) try self.ensureGrad(s.input_a) else null;
            const gb = if (needsGrad(s.input_b)) try self.ensureGrad(s.input_b) else null;
            if (ga != null or gb != null) {
                // Both grad buffers must exist for the kernel (it writes to both).
                // Use a temp zero buffer for the side that doesn't need grad.
                const real_ga = ga orelse blk: {
                    const tmp = try self.arena.alloc(@as(usize, s.rows) * @as(usize, s.cols_a));
                    try self.ops.fill(self.ctx, tmp, 0.0);
                    break :blk tmp;
                };
                const real_gb = gb orelse blk: {
                    const tmp = try self.arena.alloc(@as(usize, s.rows) * @as(usize, s.cols_b));
                    try self.ops.fill(self.ctx, tmp, 0.0);
                    break :blk tmp;
                };
                try self.ops.concatBackward(self.ctx, grad_out, real_ga, real_gb, s.rows, s.cols_a, s.cols_b);
            }
        },
        .sparse_accumulate => |s| {
            if (s.weights.requires_grad) {
                const gw = try self.ensureGrad(s.weights);
                try self.ops.sparseAccumulateBackward(self.ctx, gw, grad_out, s.active_indices, s.num_active, s.batch_size, s.max_active, s.out_features);
            }
            if (s.bias.requires_grad) {
                const gb = try self.ensureGrad(s.bias);
                try self.ops.addBiasBackward(self.ctx, grad_out, gb, s.batch_size, s.out_features);
            }
        },
    }
}

/// Clip gradients by global L2 norm. Returns the original norm.
/// If global_norm > max_norm, all gradients are scaled by max_norm / global_norm.
pub fn clipGradNorm(self: *Graph, params: []const *Tensor, max_norm: f32) !f32 {
    var total: f64 = 0;

    for (params) |p| {
        const g = p.grad orelse continue;
        total += @as(f64, try self.ops.squaredSum(self.ctx, g.storage.gpu.buffer, self.allocator));
    }

    const global_norm: f32 = @floatCast(@sqrt(total));

    if (global_norm > max_norm) {
        const scale = max_norm / global_norm;
        for (params) |p| {
            const g = p.grad orelse continue;
            const n: u32 = @intCast(g.elements());
            try self.ops.weightedAdd(self.ctx, g.storage.gpu.buffer, g.storage.gpu.buffer, g.storage.gpu.buffer, scale, 0.0, n);
        }
    }

    return global_norm;
}

fn needsGrad(tensor: *const Tensor) bool {
    return tensor.requires_grad or tensor.tape_index != null;
}

fn ensureGrad(self: *Graph, tensor: *Tensor) !Buffer {
    if (tensor.grad) |g| {
        return g.storage.gpu.buffer;
    }

    const n = tensor.elements();

    if (tensor.requires_grad) {
        // Parameter: persistent gradient buffer (survives arena reset)
        var buf = try Buffer.alloc(self.ctx, n);
        errdefer buf.release();
        try self.ops.fill(self.ctx, buf, 0.0);
        const g = try tensor.allocator.create(Tensor);
        g.* = .{
            .shape = tensor.shape,
            .storage = .{ .gpu = .{ .buffer = buf } },
            .allocator = tensor.allocator,
        };
        tensor.grad = g;
        return buf;
    } else {
        // Intermediate: arena-managed gradient buffer
        const buf = try self.arena.alloc(n);
        try self.ops.fill(self.ctx, buf, 0.0);
        const g = try self.allocator.create(Tensor);
        g.* = .{
            .shape = tensor.shape,
            .storage = .{ .gpu = .{ .buffer = buf } },
            .allocator = self.allocator,
        };
        tensor.grad = g;
        return buf;
    }
}

fn createIntermediate(self: *Graph, shape_dims: []const usize, buf: Buffer) !*Tensor {
    const t = try self.allocator.create(Tensor);
    t.* = .{
        .shape = Shape.init(shape_dims),
        .storage = .{ .gpu = .{ .buffer = buf } },
        .allocator = self.allocator,
    };
    try self.intermediates.append(self.allocator, t);
    return t;
}

const cl = Context.cl;

fn copyBuffer(ctx: *const Context, src: Buffer, dst: Buffer, len: usize) !void {
    try Context.check(cl.clEnqueueCopyBuffer(
        ctx.queue,
        src.mem,
        dst.mem,
        0,
        0,
        len * @sizeOf(f32),
        0,
        null,
        null,
    ));
    try Context.check(cl.clFinish(ctx.queue));
}

fn makeTensor(allocator: std.mem.Allocator, shape_dims: []const usize, buf: Buffer) Tensor {
    return .{
        .shape = Shape.init(shape_dims),
        .storage = .{ .gpu = .{ .buffer = buf } },
        .allocator = allocator,
    };
}

/// Run forward pass (matmul -> addBias -> clippedRelu) on raw buffers, return sum(output).
fn computeDenseLoss(
    allocator: std.mem.Allocator,
    ctx: *const Context,
    ops: *const Ops,
    x: []const f32,
    w: []const f32,
    b: []const f32,
    m: u32,
    n: u32,
    k: u32,
    max_val: f32,
) !f32 {
    var x_buf = try Buffer.upload(ctx, x);
    defer x_buf.release();
    var w_buf = try Buffer.upload(ctx, w);
    defer w_buf.release();
    var b_buf = try Buffer.upload(ctx, b);
    defer b_buf.release();

    var h_buf = try Buffer.alloc(ctx, @as(usize, m) * @as(usize, n));
    defer h_buf.release();
    try ops.matmul(ctx, x_buf, w_buf, h_buf, m, n, k);
    try ops.addBias(ctx, h_buf, b_buf, m, n);

    var out_buf = try Buffer.alloc(ctx, @as(usize, m) * @as(usize, n));
    defer out_buf.release();
    try ops.clippedRelu(ctx, h_buf, out_buf, max_val, m * n);

    return ops.reduceSum(ctx, out_buf, allocator);
}

/// Run forward pass (matmul -> addBias -> squaredClippedRelu) on raw buffers, return sum(output).
fn computeSquaredClippedReluLoss(
    allocator: std.mem.Allocator,
    ctx: *const Context,
    ops: *const Ops,
    x: []const f32,
    w: []const f32,
    b: []const f32,
    m: u32,
    n: u32,
    k: u32,
    max_val: f32,
) !f32 {
    var x_buf = try Buffer.upload(ctx, x);
    defer x_buf.release();
    var w_buf = try Buffer.upload(ctx, w);
    defer w_buf.release();
    var b_buf = try Buffer.upload(ctx, b);
    defer b_buf.release();

    var h_buf = try Buffer.alloc(ctx, @as(usize, m) * @as(usize, n));
    defer h_buf.release();
    try ops.matmul(ctx, x_buf, w_buf, h_buf, m, n, k);
    try ops.addBias(ctx, h_buf, b_buf, m, n);

    var out_buf = try Buffer.alloc(ctx, @as(usize, m) * @as(usize, n));
    defer out_buf.release();
    try ops.squaredClippedRelu(ctx, h_buf, out_buf, max_val, m * n);

    return ops.reduceSum(ctx, out_buf, allocator);
}

fn relError(a: f32, n: f32) f32 {
    const diff = @abs(a - n);
    const scale = @max(@abs(a), @abs(n));
    if (scale < 1e-6) return diff;
    return diff / scale;
}

test "numerical gradient check: matmul + addBias + clippedRelu" {
    var ctx = try Context.init();
    defer ctx.deinit();
    var ops = try Ops.init(&ctx);
    defer ops.deinit();
    const allocator = std.testing.allocator;

    const M: u32 = 2;
    const K: u32 = 4;
    const N: u32 = 3;
    const max_val: f32 = 6.0;

    // Parameters with moderate values away from activation boundaries
    var w_data = [_]f32{ 0.15, -0.22, 0.31, 0.41, -0.12, 0.25, -0.33, 0.18, 0.27, -0.14, 0.36, 0.08 };
    var b_data = [_]f32{ 0.05, -0.03, 0.07 };
    const x_data = [_]f32{ 0.5, 0.3, 0.8, 0.2, 0.9, 0.1, 0.4, 0.6 };

    // -- Analytical gradients via autograd --
    const w_buf = try Buffer.upload(&ctx, &w_data);
    var weight = makeTensor(allocator, &.{ K, N }, w_buf);
    weight.requires_grad = true;
    defer weight.deinit();

    const b_buf = try Buffer.upload(&ctx, &b_data);
    var bias = makeTensor(allocator, &.{N}, b_buf);
    bias.requires_grad = true;
    defer bias.deinit();

    const x_buf = try Buffer.upload(&ctx, &x_data);
    var input = makeTensor(allocator, &.{ M, K }, x_buf);
    defer input.deinit();

    var graph = Graph.init(allocator, &ctx, &ops);
    defer graph.deinit();

    const h1 = try graph.matmul(&input, &weight);
    const h2 = try graph.addBias(h1, &bias);
    const h3 = try graph.clippedRelu(h2, max_val);

    // loss = sum(output), so dL/d(output) = all 1s
    try graph.setGrad(h3, 1.0);
    try graph.backward(h3);

    // Download analytical gradients
    var w_grad: [K * N]f32 = undefined;
    try weight.grad.?.storage.gpu.buffer.download(&ctx, &w_grad);
    var b_grad: [N]f32 = undefined;
    try bias.grad.?.storage.gpu.buffer.download(&ctx, &b_grad);

    // -- Numerical gradients via finite differences --
    const eps: f32 = 1e-3;
    const tol: f32 = 2e-2;

    // Weight gradients
    for (0..K * N) |idx| {
        const orig = w_data[idx];

        w_data[idx] = orig + eps;
        const loss_plus = try computeDenseLoss(allocator, &ctx, &ops, &x_data, &w_data, &b_data, M, N, K, max_val);

        w_data[idx] = orig - eps;
        const loss_minus = try computeDenseLoss(allocator, &ctx, &ops, &x_data, &w_data, &b_data, M, N, K, max_val);

        w_data[idx] = orig;

        const numerical = (loss_plus - loss_minus) / (2.0 * eps);
        const err = relError(w_grad[idx], numerical);
        if (err > tol) {
            std.debug.print("weight grad[{d}]: analytical={d:.6}, numerical={d:.6}, rel_err={d:.6}\n", .{ idx, w_grad[idx], numerical, err });
            return error.TestUnexpectedResult;
        }
    }

    // Bias gradients
    for (0..N) |idx| {
        const orig = b_data[idx];

        b_data[idx] = orig + eps;
        const loss_plus = try computeDenseLoss(allocator, &ctx, &ops, &x_data, &w_data, &b_data, M, N, K, max_val);

        b_data[idx] = orig - eps;
        const loss_minus = try computeDenseLoss(allocator, &ctx, &ops, &x_data, &w_data, &b_data, M, N, K, max_val);

        b_data[idx] = orig;

        const numerical = (loss_plus - loss_minus) / (2.0 * eps);
        const err = relError(b_grad[idx], numerical);
        if (err > tol) {
            std.debug.print("bias grad[{d}]: analytical={d:.6}, numerical={d:.6}, rel_err={d:.6}\n", .{ idx, b_grad[idx], numerical, err });
            return error.TestUnexpectedResult;
        }
    }
}

test "numerical gradient check: matmul + addBias + squaredClippedRelu" {
    var ctx = try Context.init();
    defer ctx.deinit();
    var ops = try Ops.init(&ctx);
    defer ops.deinit();
    const allocator = std.testing.allocator;

    const M: u32 = 2;
    const K: u32 = 4;
    const N: u32 = 3;
    const max_val: f32 = 6.0;

    // Parameters with moderate values away from the activation boundaries (0, max_val)
    // so the squared activation's derivative (2x in-region) is smooth for finite diffs.
    var w_data = [_]f32{ 0.15, -0.22, 0.31, 0.41, -0.12, 0.25, -0.33, 0.18, 0.27, -0.14, 0.36, 0.08 };
    var b_data = [_]f32{ 0.05, -0.03, 0.07 };
    const x_data = [_]f32{ 0.5, 0.3, 0.8, 0.2, 0.9, 0.1, 0.4, 0.6 };

    // -- Analytical gradients via autograd --
    const w_buf = try Buffer.upload(&ctx, &w_data);
    var weight = makeTensor(allocator, &.{ K, N }, w_buf);
    weight.requires_grad = true;
    defer weight.deinit();

    const b_buf = try Buffer.upload(&ctx, &b_data);
    var bias = makeTensor(allocator, &.{N}, b_buf);
    bias.requires_grad = true;
    defer bias.deinit();

    const x_buf = try Buffer.upload(&ctx, &x_data);
    var input = makeTensor(allocator, &.{ M, K }, x_buf);
    defer input.deinit();

    var graph = Graph.init(allocator, &ctx, &ops);
    defer graph.deinit();

    const h1 = try graph.matmul(&input, &weight);
    const h2 = try graph.addBias(h1, &bias);
    const h3 = try graph.squaredClippedRelu(h2, max_val);

    // loss = sum(output), so dL/d(output) = all 1s
    try graph.setGrad(h3, 1.0);
    try graph.backward(h3);

    var w_grad: [K * N]f32 = undefined;
    try weight.grad.?.storage.gpu.buffer.download(&ctx, &w_grad);
    var b_grad: [N]f32 = undefined;
    try bias.grad.?.storage.gpu.buffer.download(&ctx, &b_grad);

    // -- Numerical gradients via finite differences --
    const eps: f32 = 1e-3;
    const tol: f32 = 2e-2;

    for (0..K * N) |idx| {
        const orig = w_data[idx];

        w_data[idx] = orig + eps;
        const loss_plus = try computeSquaredClippedReluLoss(allocator, &ctx, &ops, &x_data, &w_data, &b_data, M, N, K, max_val);

        w_data[idx] = orig - eps;
        const loss_minus = try computeSquaredClippedReluLoss(allocator, &ctx, &ops, &x_data, &w_data, &b_data, M, N, K, max_val);

        w_data[idx] = orig;

        const numerical = (loss_plus - loss_minus) / (2.0 * eps);
        const err = relError(w_grad[idx], numerical);
        if (err > tol) {
            std.debug.print("scrl weight grad[{d}]: analytical={d:.6}, numerical={d:.6}, rel_err={d:.6}\n", .{ idx, w_grad[idx], numerical, err });
            return error.TestUnexpectedResult;
        }
    }

    for (0..N) |idx| {
        const orig = b_data[idx];

        b_data[idx] = orig + eps;
        const loss_plus = try computeSquaredClippedReluLoss(allocator, &ctx, &ops, &x_data, &w_data, &b_data, M, N, K, max_val);

        b_data[idx] = orig - eps;
        const loss_minus = try computeSquaredClippedReluLoss(allocator, &ctx, &ops, &x_data, &w_data, &b_data, M, N, K, max_val);

        b_data[idx] = orig;

        const numerical = (loss_plus - loss_minus) / (2.0 * eps);
        const err = relError(b_grad[idx], numerical);
        if (err > tol) {
            std.debug.print("scrl bias grad[{d}]: analytical={d:.6}, numerical={d:.6}, rel_err={d:.6}\n", .{ idx, b_grad[idx], numerical, err });
            return error.TestUnexpectedResult;
        }
    }
}

/// Run forward pass (sparse_accumulate -> clippedRelu) on raw buffers, return sum(output).
fn computeSparseLoss(
    allocator: std.mem.Allocator,
    ctx: *const Context,
    ops: *const Ops,
    w: []const f32,
    b: []const f32,
    indices: []const u32,
    num_active_data: []const u32,
    batch_size: u32,
    max_active: u32,
    out_features: u32,
    max_val: f32,
) !f32 {
    const cl_ = Context.cl;
    var w_buf = try Buffer.upload(ctx, w);
    defer w_buf.release();
    var b_buf = try Buffer.upload(ctx, b);
    defer b_buf.release();

    var err: cl_.cl_int = undefined;
    const idx_mem = cl_.clCreateBuffer(ctx.context, cl_.CL_MEM_READ_WRITE | cl_.CL_MEM_COPY_HOST_PTR, indices.len * @sizeOf(u32), @ptrCast(@constCast(indices.ptr)), &err);
    try Context.check(err);
    var idx_buf: Buffer = .{ .mem = idx_mem, .len = indices.len };
    defer idx_buf.release();

    const na_mem = cl_.clCreateBuffer(ctx.context, cl_.CL_MEM_READ_WRITE | cl_.CL_MEM_COPY_HOST_PTR, num_active_data.len * @sizeOf(u32), @ptrCast(@constCast(num_active_data.ptr)), &err);
    try Context.check(err);
    var na_buf: Buffer = .{ .mem = na_mem, .len = num_active_data.len };
    defer na_buf.release();

    const n: usize = @as(usize, batch_size) * @as(usize, out_features);
    var sa_buf = try Buffer.alloc(ctx, n);
    defer sa_buf.release();
    try ops.sparseAccumulate(ctx, w_buf, b_buf, idx_buf, na_buf, sa_buf, batch_size, max_active, out_features);

    var out_buf = try Buffer.alloc(ctx, n);
    defer out_buf.release();
    try ops.clippedRelu(ctx, sa_buf, out_buf, max_val, @intCast(n));

    return ops.reduceSum(ctx, out_buf, allocator);
}

test "numerical gradient check: sparseAccumulate + clippedRelu" {
    var ctx = try Context.init();
    defer ctx.deinit();
    var ops = try Ops.init(&ctx);
    defer ops.deinit();
    const allocator = std.testing.allocator;

    const in_features: u32 = 6;
    const out_features: u32 = 4;
    const batch_size: u32 = 2;
    const max_active: u32 = 3;
    const max_val: f32 = 6.0;

    // Weights [6][4], bias [4]
    var w_data = [_]f32{
        0.1,   -0.2,  0.3,   0.15,
        0.25,  0.05,  -0.1,  0.35,
        -0.3,  0.4,   0.2,   -0.15,
        0.12,  0.22,  -0.32, 0.08,
        -0.18, 0.28,  0.13,  -0.23,
        0.33,  -0.13, 0.43,  0.03,
    };
    var b_data = [_]f32{ 0.05, -0.02, 0.04, -0.01 };

    // Sample 0: active = [0, 2, 4], count = 3
    // Sample 1: active = [1, 3, _], count = 2
    const indices = [_]u32{ 0, 2, 4, 1, 3, 0 };
    const num_active_data = [_]u32{ 3, 2 };

    // Upload u32 buffers
    const cl_ = Context.cl;
    var err: cl_.cl_int = undefined;
    const idx_mem = cl_.clCreateBuffer(ctx.context, cl_.CL_MEM_READ_WRITE | cl_.CL_MEM_COPY_HOST_PTR, indices.len * @sizeOf(u32), @ptrCast(@constCast(&indices)), &err);
    try Context.check(err);
    var idx_buf: Buffer = .{ .mem = idx_mem, .len = indices.len };
    defer idx_buf.release();

    const na_mem = cl_.clCreateBuffer(ctx.context, cl_.CL_MEM_READ_WRITE | cl_.CL_MEM_COPY_HOST_PTR, num_active_data.len * @sizeOf(u32), @ptrCast(@constCast(&num_active_data)), &err);
    try Context.check(err);
    var na_buf: Buffer = .{ .mem = na_mem, .len = num_active_data.len };
    defer na_buf.release();

    // -- Analytical gradients --
    const w_buf = try Buffer.upload(&ctx, &w_data);
    var weight = makeTensor(allocator, &.{ in_features, out_features }, w_buf);
    weight.requires_grad = true;
    defer weight.deinit();

    const b_buf_ = try Buffer.upload(&ctx, &b_data);
    var bias = makeTensor(allocator, &.{out_features}, b_buf_);
    bias.requires_grad = true;
    defer bias.deinit();

    var graph = Graph.init(allocator, &ctx, &ops);
    defer graph.deinit();

    const h1 = try graph.sparseAccumulate(&weight, &bias, idx_buf, na_buf, batch_size, max_active);
    const h2 = try graph.clippedRelu(h1, max_val);

    try graph.setGrad(h2, 1.0);
    try graph.backward(h2);

    var w_grad: [in_features * out_features]f32 = undefined;
    try weight.grad.?.storage.gpu.buffer.download(&ctx, &w_grad);
    var b_grad: [out_features]f32 = undefined;
    try bias.grad.?.storage.gpu.buffer.download(&ctx, &b_grad);

    // -- Numerical gradients --
    const eps: f32 = 1e-3;
    const tol: f32 = 2e-2;

    // Weight gradients (only active rows should have non-zero)
    for (0..in_features * out_features) |idx| {
        const orig = w_data[idx];

        w_data[idx] = orig + eps;
        const loss_plus = try computeSparseLoss(allocator, &ctx, &ops, &w_data, &b_data, &indices, &num_active_data, batch_size, max_active, out_features, max_val);

        w_data[idx] = orig - eps;
        const loss_minus = try computeSparseLoss(allocator, &ctx, &ops, &w_data, &b_data, &indices, &num_active_data, batch_size, max_active, out_features, max_val);

        w_data[idx] = orig;

        const numerical = (loss_plus - loss_minus) / (2.0 * eps);
        const err_ = relError(w_grad[idx], numerical);
        if (err_ > tol) {
            std.debug.print("sparse weight grad[{d}]: analytical={d:.6}, numerical={d:.6}, rel_err={d:.6}\n", .{ idx, w_grad[idx], numerical, err_ });
            return error.TestUnexpectedResult;
        }
    }

    // Bias gradients
    for (0..out_features) |idx| {
        const orig = b_data[idx];

        b_data[idx] = orig + eps;
        const loss_plus = try computeSparseLoss(allocator, &ctx, &ops, &w_data, &b_data, &indices, &num_active_data, batch_size, max_active, out_features, max_val);

        b_data[idx] = orig - eps;
        const loss_minus = try computeSparseLoss(allocator, &ctx, &ops, &w_data, &b_data, &indices, &num_active_data, batch_size, max_active, out_features, max_val);

        b_data[idx] = orig;

        const numerical = (loss_plus - loss_minus) / (2.0 * eps);
        const err_ = relError(b_grad[idx], numerical);
        if (err_ > tol) {
            std.debug.print("sparse bias grad[{d}]: analytical={d:.6}, numerical={d:.6}, rel_err={d:.6}\n", .{ idx, b_grad[idx], numerical, err_ });
            return error.TestUnexpectedResult;
        }
    }
}

test "numerical gradient check: matmul + sigmoid" {
    var ctx = try Context.init();
    defer ctx.deinit();
    var ops = try Ops.init(&ctx);
    defer ops.deinit();
    const allocator = std.testing.allocator;

    const M: u32 = 2;
    const K: u32 = 3;
    const N: u32 = 2;

    var w_data = [_]f32{ 0.3, -0.4, 0.2, 0.1, -0.3, 0.5 };
    const x_data = [_]f32{ 0.5, 0.8, 0.2, 0.1, 0.6, 0.9 };

    // -- Analytical --
    const w_buf = try Buffer.upload(&ctx, &w_data);
    var weight = makeTensor(allocator, &.{ K, N }, w_buf);
    weight.requires_grad = true;
    defer weight.deinit();

    const x_buf = try Buffer.upload(&ctx, &x_data);
    var input = makeTensor(allocator, &.{ M, K }, x_buf);
    defer input.deinit();

    var graph = Graph.init(allocator, &ctx, &ops);
    defer graph.deinit();

    const h1 = try graph.matmul(&input, &weight);
    const h2 = try graph.sigmoid(h1);

    try graph.setGrad(h2, 1.0);
    try graph.backward(h2);

    var w_grad: [K * N]f32 = undefined;
    try weight.grad.?.storage.gpu.buffer.download(&ctx, &w_grad);

    // -- Numerical --
    const eps: f32 = 1e-3;
    const tol: f32 = 2e-2;

    for (0..K * N) |idx| {
        const orig = w_data[idx];

        w_data[idx] = orig + eps;
        const lp = try computeSigmoidLoss(allocator, &ctx, &ops, &x_data, &w_data, M, N, K);

        w_data[idx] = orig - eps;
        const lm = try computeSigmoidLoss(allocator, &ctx, &ops, &x_data, &w_data, M, N, K);

        w_data[idx] = orig;

        const numerical = (lp - lm) / (2.0 * eps);
        const err = relError(w_grad[idx], numerical);
        if (err > tol) {
            std.debug.print("sigmoid weight grad[{d}]: analytical={d:.6}, numerical={d:.6}, rel_err={d:.6}\n", .{ idx, w_grad[idx], numerical, err });
            return error.TestUnexpectedResult;
        }
    }
}

test "numerical gradient check: concat + matmul" {
    var ctx = try Context.init();
    defer ctx.deinit();
    var ops = try Ops.init(&ctx);
    defer ops.deinit();
    const allocator = std.testing.allocator;

    const M: u32 = 2;
    const Ca: u32 = 3;
    const Cb: u32 = 2;
    const N: u32 = 2;

    var a_data = [_]f32{ 0.1, -0.2, 0.3, 0.4, 0.15, -0.1 };
    var b_data = [_]f32{ 0.25, -0.35, 0.05, 0.45 };
    var w_data = [_]f32{ 0.1, 0.2, -0.3, 0.4, 0.15, -0.25, -0.1, 0.35, 0.2, -0.15 };

    // -- Analytical --
    const a_buf = try Buffer.upload(&ctx, &a_data);
    var input_a = makeTensor(allocator, &.{ M, Ca }, a_buf);
    input_a.requires_grad = true;
    defer input_a.deinit();

    const b_buf = try Buffer.upload(&ctx, &b_data);
    var input_b = makeTensor(allocator, &.{ M, Cb }, b_buf);
    input_b.requires_grad = true;
    defer input_b.deinit();

    const w_buf = try Buffer.upload(&ctx, &w_data);
    var weight = makeTensor(allocator, &.{ Ca + Cb, N }, w_buf);
    weight.requires_grad = true;
    defer weight.deinit();

    var graph = Graph.init(allocator, &ctx, &ops);
    defer graph.deinit();

    const h1 = try graph.concat(&input_a, &input_b);
    const h2 = try graph.matmul(h1, &weight);

    try graph.setGrad(h2, 1.0);
    try graph.backward(h2);

    var a_grad: [M * Ca]f32 = undefined;
    try input_a.grad.?.storage.gpu.buffer.download(&ctx, &a_grad);
    var b_grad: [M * Cb]f32 = undefined;
    try input_b.grad.?.storage.gpu.buffer.download(&ctx, &b_grad);
    var w_grad: [(Ca + Cb) * N]f32 = undefined;
    try weight.grad.?.storage.gpu.buffer.download(&ctx, &w_grad);

    // -- Numerical --
    const eps: f32 = 1e-3;
    const tol: f32 = 2e-2;

    for (0..M * Ca) |idx| {
        const orig = a_data[idx];
        a_data[idx] = orig + eps;
        const lp = try computeConcatLoss(allocator, &ctx, &ops, &a_data, &b_data, &w_data, M, Ca, Cb, N);
        a_data[idx] = orig - eps;
        const lm = try computeConcatLoss(allocator, &ctx, &ops, &a_data, &b_data, &w_data, M, Ca, Cb, N);
        a_data[idx] = orig;
        const numerical = (lp - lm) / (2.0 * eps);
        const err = relError(a_grad[idx], numerical);
        if (err > tol) {
            std.debug.print("concat input_a grad[{d}]: analytical={d:.6}, numerical={d:.6}, rel_err={d:.6}\n", .{ idx, a_grad[idx], numerical, err });
            return error.TestUnexpectedResult;
        }
    }

    for (0..M * Cb) |idx| {
        const orig = b_data[idx];
        b_data[idx] = orig + eps;
        const lp = try computeConcatLoss(allocator, &ctx, &ops, &a_data, &b_data, &w_data, M, Ca, Cb, N);
        b_data[idx] = orig - eps;
        const lm = try computeConcatLoss(allocator, &ctx, &ops, &a_data, &b_data, &w_data, M, Ca, Cb, N);
        b_data[idx] = orig;
        const numerical = (lp - lm) / (2.0 * eps);
        const err = relError(b_grad[idx], numerical);
        if (err > tol) {
            std.debug.print("concat input_b grad[{d}]: analytical={d:.6}, numerical={d:.6}, rel_err={d:.6}\n", .{ idx, b_grad[idx], numerical, err });
            return error.TestUnexpectedResult;
        }
    }

    for (0..(Ca + Cb) * N) |idx| {
        const orig = w_data[idx];
        w_data[idx] = orig + eps;
        const lp = try computeConcatLoss(allocator, &ctx, &ops, &a_data, &b_data, &w_data, M, Ca, Cb, N);
        w_data[idx] = orig - eps;
        const lm = try computeConcatLoss(allocator, &ctx, &ops, &a_data, &b_data, &w_data, M, Ca, Cb, N);
        w_data[idx] = orig;
        const numerical = (lp - lm) / (2.0 * eps);
        const err = relError(w_grad[idx], numerical);
        if (err > tol) {
            std.debug.print("concat weight grad[{d}]: analytical={d:.6}, numerical={d:.6}, rel_err={d:.6}\n", .{ idx, w_grad[idx], numerical, err });
            return error.TestUnexpectedResult;
        }
    }
}

fn computeConcatLoss(
    allocator: std.mem.Allocator,
    ctx: *const Context,
    ops: *const Ops,
    a: []const f32,
    b: []const f32,
    w: []const f32,
    m: u32,
    cols_a: u32,
    cols_b: u32,
    n: u32,
) !f32 {
    var a_buf = try Buffer.upload(ctx, a);
    defer a_buf.release();
    var b_buf = try Buffer.upload(ctx, b);
    defer b_buf.release();

    const cols_out = cols_a + cols_b;
    var cat_buf = try Buffer.alloc(ctx, @as(usize, m) * @as(usize, cols_out));
    defer cat_buf.release();
    try ops.concat(ctx, a_buf, b_buf, cat_buf, m, cols_a, cols_b);

    var w_buf = try Buffer.upload(ctx, w);
    defer w_buf.release();
    var out_buf = try Buffer.alloc(ctx, @as(usize, m) * @as(usize, n));
    defer out_buf.release();
    try ops.matmul(ctx, cat_buf, w_buf, out_buf, m, n, cols_out);

    return ops.reduceSum(ctx, out_buf, allocator);
}

fn computeSigmoidLoss(
    allocator: std.mem.Allocator,
    ctx: *const Context,
    ops: *const Ops,
    x: []const f32,
    w: []const f32,
    m: u32,
    n: u32,
    k: u32,
) !f32 {
    var x_buf = try Buffer.upload(ctx, x);
    defer x_buf.release();
    var w_buf = try Buffer.upload(ctx, w);
    defer w_buf.release();

    var h_buf = try Buffer.alloc(ctx, @as(usize, m) * @as(usize, n));
    defer h_buf.release();
    try ops.matmul(ctx, x_buf, w_buf, h_buf, m, n, k);

    var out_buf = try Buffer.alloc(ctx, @as(usize, m) * @as(usize, n));
    defer out_buf.release();
    try ops.sigmoid(ctx, h_buf, out_buf, m * n);

    return ops.reduceSum(ctx, out_buf, allocator);
}
