const std = @import("std");
const Tensor = @import("Tensor.zig");
const Graph = @import("Graph.zig");
const Context = @import("gpu/Context.zig");
const Buffer = @import("gpu/Buffer.zig");

const Linear = @This();

weight: *Tensor,
bias: *Tensor,

pub fn init(allocator: std.mem.Allocator, ctx: *const Context, in_features: u32, out_features: u32, seed: u64) !Linear {
    const in: usize = in_features;
    const out: usize = out_features;

    // Kaiming uniform: limit = sqrt(6 / fan_in)
    const limit: f32 = @sqrt(6.0 / @as(f32, @floatFromInt(in_features)));

    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();

    // Weight [in_features x out_features]
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

    // Bias [out_features], zero-initialized
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

    return .{ .weight = weight, .bias = bias };
}

pub fn forward(self: *const Linear, input: *Tensor, graph: *Graph) !*Tensor {
    const h = try graph.matmul(input, self.weight);
    return graph.addBias(h, self.bias);
}

pub fn parameters(self: *const Linear) [2]*Tensor {
    return .{ self.weight, self.bias };
}

pub fn deinit(self: *Linear) void {
    const alloc = self.weight.allocator;
    self.weight.deinit();
    alloc.destroy(self.weight);
    self.bias.deinit();
    alloc.destroy(self.bias);
    self.* = undefined;
}
