const std = @import("std");
const Tensor = @import("Tensor.zig");
const Graph = @import("Graph.zig");
const Layer = @import("Layer.zig");

const Sequential = @This();

layers: []Layer,
allocator: std.mem.Allocator,

pub fn init(allocator: std.mem.Allocator, layers: []const Layer) !Sequential {
    const owned = try allocator.alloc(Layer, layers.len);
    @memcpy(owned, layers);
    return .{ .layers = owned, .allocator = allocator };
}

pub fn forward(self: *const Sequential, input: *Tensor, graph: *Graph) !*Tensor {
    var x = input;
    for (self.layers) |*layer| {
        x = try layer.forward(x, graph);
    }
    return x;
}

pub fn parameters(self: *const Sequential) !std.ArrayList(*Tensor) {
    var list: std.ArrayList(*Tensor) = .empty;
    for (self.layers) |*layer| {
        try layer.appendParameters(&list, self.allocator);
    }
    return list;
}

pub fn deinit(self: *Sequential) void {
    for (self.layers) |*layer| {
        layer.deinit();
    }
    self.allocator.free(self.layers);
    self.* = undefined;
}
