const std = @import("std");
const Tensor = @import("Tensor.zig");
const Graph = @import("Graph.zig");
const Linear = @import("Linear.zig");
const SparseLinear = @import("SparseLinear.zig");
const ClippedReLU = @import("ClippedReLU.zig");

const Layer = @This();

pub const Tag = enum { linear, sparse_linear, clipped_relu };

value: union(Tag) {
    linear: Linear,
    sparse_linear: SparseLinear,
    clipped_relu: ClippedReLU,
},

pub fn forward(self: *const Layer, input: *Tensor, graph: *Graph) !*Tensor {
    return switch (self.value) {
        .linear => |*l| l.forward(input, graph),
        .sparse_linear => |*l| l.forward(input, graph),
        .clipped_relu => |*l| l.forward(input, graph),
    };
}

pub fn appendParameters(self: *const Layer, list: *std.ArrayList(*Tensor), allocator: std.mem.Allocator) !void {
    switch (self.value) {
        .linear => |l| {
            const p = l.parameters();
            try list.appendSlice(allocator, &p);
        },
        .sparse_linear => |l| {
            const p = l.parameters();
            try list.appendSlice(allocator, &p);
        },
        .clipped_relu => {},
    }
}

pub fn deinit(self: *Layer) void {
    switch (self.value) {
        .linear => |*l| l.deinit(),
        .sparse_linear => |*l| l.deinit(),
        .clipped_relu => |*l| l.deinit(),
    }
    self.* = undefined;
}

pub fn linear(l: Linear) Layer {
    return .{ .value = .{ .linear = l } };
}

pub fn sparseLinear(l: SparseLinear) Layer {
    return .{ .value = .{ .sparse_linear = l } };
}

pub fn clippedRelu(max_val: f32) Layer {
    return .{ .value = .{ .clipped_relu = ClippedReLU.init(max_val) } };
}
