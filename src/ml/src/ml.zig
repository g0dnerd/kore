const std = @import("std");

pub const Shape = @import("Shape.zig");
pub const Storage = @import("Storage.zig").Storage;
pub const Tensor = @import("Tensor.zig");
pub const tape = @import("tape.zig");
pub const Graph = @import("Graph.zig");

pub const Linear = @import("Linear.zig");
pub const SparseLinear = @import("SparseLinear.zig");
pub const ClippedReLU = @import("ClippedReLU.zig");
pub const Layer = @import("Layer.zig");
pub const Sequential = @import("Sequential.zig");
pub const MseLoss = @import("MseLoss.zig");
pub const Adam = @import("Adam.zig");
pub const serialize = @import("serialize.zig");

pub const gpu = struct {
    pub const Context = @import("gpu/Context.zig");
    pub const Buffer = @import("gpu/Buffer.zig");
    pub const Program = @import("gpu/Program.zig");
    pub const Arena = @import("gpu/Arena.zig");
    pub const ops = @import("gpu/ops.zig");

    test {
        std.testing.refAllDecls(@This());
    }
};

pub const cpu = struct {
    pub const ops = @import("cpu/ops.zig");
    pub const quantized = @import("cpu/quantized.zig");

    test {
        std.testing.refAllDecls(@This());
    }
};

// ── Integration tests ───────────────────────────────────────────────

fn uploadU32(ctx: *const gpu.Context, data: []const u32) gpu.Context.Error!gpu.Buffer {
    const cl = gpu.Context.cl;
    var err: cl.cl_int = undefined;
    const mem = cl.clCreateBuffer(
        ctx.context,
        cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,
        data.len * @sizeOf(u32),
        @ptrCast(@constCast(data.ptr)),
        &err,
    );
    try gpu.Context.check(err);
    return .{ .mem = mem, .len = data.len };
}

test "train dense network on y = sin(x)" {
    const allocator = std.testing.allocator;
    var ctx = try gpu.Context.init();
    defer ctx.deinit();
    var ops_ = try gpu.ops.init(&ctx);
    defer ops_.deinit();

    var model = try Sequential.init(allocator, &.{
        Layer.linear(try Linear.init(allocator, &ctx, 1, 32, 42)),
        Layer.clippedRelu(6.0),
        Layer.linear(try Linear.init(allocator, &ctx, 32, 1, 123)),
    });
    defer model.deinit();

    var params_list = try model.parameters();
    defer params_list.deinit(allocator);
    const params = params_list.items;

    var adam = try Adam.init(allocator, &ctx, &ops_, params, .{ .lr = 1e-3 });
    defer adam.deinit();

    var graph_ = Graph.init(allocator, &ctx, &ops_);
    defer graph_.deinit();

    const loss_fn: MseLoss = .{};

    const N = 32;
    var x_data: [N]f32 = undefined;
    var y_data: [N]f32 = undefined;
    for (0..N) |i| {
        const t: f32 = @as(f32, @floatFromInt(i)) / @as(f32, N - 1);
        x_data[i] = -1.0 + 2.0 * t;
        y_data[i] = @sin(x_data[i] * std.math.pi);
    }

    var x_buf = try gpu.Buffer.upload(&ctx, &x_data);
    defer x_buf.release();
    var target_buf = try gpu.Buffer.upload(&ctx, &y_data);
    defer target_buf.release();

    // Stack-allocated input tensor (not owned by graph)
    var input: Tensor = .{
        .shape = Shape.init(&.{ N, 1 }),
        .storage = .{ .gpu = .{ .buffer = x_buf } },
        .allocator = allocator,
    };

    var initial_loss: f32 = 0;
    for (0..1000) |s| {
        try graph_.zeroGrads(params);
        const output = try model.forward(&input, &graph_);
        const loss = try loss_fn.forward(output, target_buf, &graph_);
        if (s == 0) initial_loss = loss;
        try graph_.backward(output);
        _ = try graph_.clipGradNorm(params, 1.0);
        try adam.step(params);
        graph_.reset();
    }

    // Final evaluation
    const final_output = try model.forward(&input, &graph_);
    const final_loss = try loss_fn.forward(final_output, target_buf, &graph_);
    graph_.reset();

    // Loss should decrease substantially over 1000 steps
    try std.testing.expect(final_loss < initial_loss * 0.1);
}

test "sparse gradients flow to active rows only" {
    const allocator = std.testing.allocator;
    var ctx = try gpu.Context.init();
    defer ctx.deinit();
    var ops_ = try gpu.ops.init(&ctx);
    defer ops_.deinit();

    const in_features: u32 = 100;
    const out_features: u32 = 16;
    const max_active: u32 = 5;
    const batch_size: u32 = 2;

    var sparse = try SparseLinear.init(allocator, &ctx, in_features, out_features, max_active, 5, 42);
    defer sparse.deinit();
    var dense = try Linear.init(allocator, &ctx, out_features, 1, 123);
    defer dense.deinit();

    // Sample 0: active = [3, 7, 15, 42, 88], count = 5
    // Sample 1: active = [1, 10, 50, 0, 0],  count = 3
    const indices = [_]u32{ 3, 7, 15, 42, 88, 1, 10, 50, 0, 0 };
    const num_active_data = [_]u32{ 5, 3 };

    var idx_buf = try uploadU32(&ctx, &indices);
    defer idx_buf.release();
    var na_buf = try uploadU32(&ctx, &num_active_data);
    defer na_buf.release();

    sparse.setIndices(idx_buf, na_buf, batch_size);

    // Dummy input (ignored by sparse forward) and target
    var dummy_data = [_]f32{ 0, 0 };
    var dummy_buf = try gpu.Buffer.upload(&ctx, &dummy_data);
    defer dummy_buf.release();
    var dummy: Tensor = .{
        .shape = Shape.init(&.{ batch_size, 1 }),
        .storage = .{ .gpu = .{ .buffer = dummy_buf } },
        .allocator = allocator,
    };

    var graph_ = Graph.init(allocator, &ctx, &ops_);
    defer graph_.deinit();

    // Forward: sparse -> dense (no activation, guarantees gradient flow)
    const h1 = try sparse.forward(&dummy, &graph_);
    const h2 = try dense.forward(h1, &graph_);

    // loss = sum(output), dL/d(output) = 1
    try graph_.setGrad(h2, 1.0);
    try graph_.backward(h2);

    // Download sparse weight gradients
    var w_grad: [in_features * out_features]f32 = undefined;
    try sparse.weight.grad.?.storage.gpu.buffer.download(&ctx, &w_grad);

    // Active rows: {1, 3, 7, 10, 15, 42, 50, 88}
    const active_rows = [_]u32{ 1, 3, 7, 10, 15, 42, 50, 88 };

    // Active rows must have non-zero gradients
    for (active_rows) |row| {
        var row_norm: f32 = 0;
        for (0..out_features) |j| {
            row_norm += @abs(w_grad[row * out_features + j]);
        }
        try std.testing.expect(row_norm > 1e-10);
    }

    // Inactive rows must have exactly zero gradients
    for (0..in_features) |row| {
        var is_active = false;
        for (active_rows) |ar| {
            if (row == ar) {
                is_active = true;
                break;
            }
        }
        if (!is_active) {
            for (0..out_features) |j| {
                try std.testing.expectApproxEqAbs(@as(f32, 0.0), w_grad[row * out_features + j], 1e-10);
            }
        }
    }
}

test "GPU quantize i8 matches CPU and round-trips within tolerance" {
    var ctx = try gpu.Context.init();
    defer ctx.deinit();
    var ops_ = try gpu.ops.init(&ctx);
    defer ops_.deinit();

    const input = [_]f32{ 1.0, -0.5, 0.25, 0.0, -1.0, 0.75, 2.0, -2.0 };
    var in_buf = try gpu.Buffer.upload(&ctx, &input);
    defer in_buf.release();

    const scale: f32 = 127.0;
    var gpu_result: [8]i8 = undefined;
    try ops_.quantizeI8(&ctx, in_buf, &gpu_result, scale);

    // CPU reference
    const cpu_result = cpu.quantized.quantize(i8, 8, &input, scale);

    // GPU must match CPU
    for (0..8) |i| {
        try std.testing.expectEqual(cpu_result[i], gpu_result[i]);
    }

    // Round-trip dequantize must be within tolerance
    const dq = cpu.quantized.dequantize(i8, 8, &gpu_result, scale);
    for (0..8) |i| {
        // Values within [-1, 1] have max error ~1/127 ≈ 0.008; saturated values (2.0, -2.0) differ more
        const expected_err: f32 = if (@abs(input[i]) > 1.0) 1.1 else 0.01;
        try std.testing.expectApproxEqAbs(input[i], dq[i], expected_err);
    }
}

test {
    std.testing.refAllDecls(@This());
}
