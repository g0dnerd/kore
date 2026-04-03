const std = @import("std");
const Tensor = @import("Tensor.zig");
const Graph = @import("Graph.zig");
const Context = @import("gpu/Context.zig");
const Buffer = @import("gpu/Buffer.zig");

const SparseLinear = @This();

weight: *Tensor,
bias: *Tensor,
out_features: u32,
max_active: u32,
// Per-batch state, set via setIndices before forward
active_indices: Buffer = undefined,
num_active: Buffer = undefined,
batch_size: u32 = 0,

pub fn init(
    allocator: std.mem.Allocator,
    ctx: *const Context,
    in_features: u32,
    out_features: u32,
    max_active: u32,
    expected_active: u32,
    seed: u64,
) !SparseLinear {
    const in: usize = in_features;
    const out: usize = out_features;

    // Kaiming uniform scaled for sparse: limit = sqrt(6 / expected_active)
    const limit: f32 = @sqrt(6.0 / @as(f32, @floatFromInt(expected_active)));

    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();

    const w_data = try allocator.alloc(f32, in * out);
    defer allocator.free(w_data);
    for (w_data) |*v| {
        v.* = rng.float(f32) * 2.0 * limit - limit;
    }
    const w_buf = try Buffer.upload(ctx, w_data);

    const weight = try allocator.create(Tensor);
    weight.* = .{
        .shape = @import("Shape.zig").init(&.{ in, out }),
        .storage = .{ .gpu = .{ .buffer = w_buf } },
        .requires_grad = true,
        .allocator = allocator,
    };

    const b_data = try allocator.alloc(f32, out);
    defer allocator.free(b_data);
    @memset(b_data, 0);
    const b_buf = try Buffer.upload(ctx, b_data);

    const bias = try allocator.create(Tensor);
    bias.* = .{
        .shape = @import("Shape.zig").init(&.{out}),
        .storage = .{ .gpu = .{ .buffer = b_buf } },
        .requires_grad = true,
        .allocator = allocator,
    };

    return .{
        .weight = weight,
        .bias = bias,
        .out_features = out_features,
        .max_active = max_active,
    };
}

/// Set the active indices for the current batch. Must be called before forward.
pub fn setIndices(self: *SparseLinear, active_indices: Buffer, num_active: Buffer, batch_size: u32) void {
    self.active_indices = active_indices;
    self.num_active = num_active;
    self.batch_size = batch_size;
}

pub fn forward(self: *const SparseLinear, input: *Tensor, graph: *Graph) !*Tensor {
    _ = input;
    return graph.sparseAccumulate(
        self.weight,
        self.bias,
        self.active_indices,
        self.num_active,
        self.batch_size,
        self.max_active,
    );
}

pub fn parameters(self: *const SparseLinear) [2]*Tensor {
    return .{ self.weight, self.bias };
}

pub fn deinit(self: *SparseLinear) void {
    const alloc = self.weight.allocator;
    self.weight.deinit();
    alloc.destroy(self.weight);
    self.bias.deinit();
    alloc.destroy(self.bias);
    self.* = undefined;
}
