const std = @import("std");
const Tensor = @import("Tensor.zig");
const Graph = @import("Graph.zig");

const ClippedReLU = @This();

max_val: f32,

pub fn init(max_val: f32) ClippedReLU {
    return .{ .max_val = max_val };
}

pub fn forward(self: *const ClippedReLU, input: *Tensor, graph: *Graph) !*Tensor {
    return graph.clippedRelu(input, self.max_val);
}

pub fn deinit(self: *ClippedReLU) void {
    self.* = undefined;
}
