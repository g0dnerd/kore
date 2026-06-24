const std = @import("std");
const Tensor = @import("Tensor.zig");
const Graph = @import("Graph.zig");

// Squared clipped ReLU: y = clamp(x, 0, max_val)^2. Used on the NNUE feature
// transformer output (SCReLU); the hidden layers keep plain ClippedReLU.
const SquaredClippedReLU = @This();

max_val: f32,

pub fn init(max_val: f32) SquaredClippedReLU {
    return .{ .max_val = max_val };
}

pub fn forward(self: *const SquaredClippedReLU, input: *Tensor, graph: *Graph) !*Tensor {
    return graph.squaredClippedRelu(input, self.max_val);
}

pub fn deinit(self: *SquaredClippedReLU) void {
    self.* = undefined;
}
