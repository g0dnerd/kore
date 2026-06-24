const std = @import("std");
const Tensor = @import("Tensor.zig");
const Shape = @import("Shape.zig");
const Graph = @import("Graph.zig");
const Buffer = @import("gpu/Buffer.zig");

const MseLoss = @This();

use_sigmoid: bool = false,
sigmoid_scale: f32 = 1.0,
// When set, `predicted` is [batch, num_buckets] and only the column selected by
// bucket[b] contributes to loss/gradient (piece-count output buckets). When null,
// the plain per-element MSE over a [batch, 1] head is used.
bucket: ?Buffer = null,

/// Compute MSE loss and set gradient on predicted tensor for backward pass.
/// Returns the scalar loss value.
pub fn forward(self: *const MseLoss, predicted: *Tensor, target: Buffer, graph: *Graph) !f32 {
    const n: u32 = @intCast(predicted.elements());
    const scale: f32 = if (self.use_sigmoid) self.sigmoid_scale else 0.0;

    const elem_loss_buf = try graph.arena.alloc(@as(usize, n));
    const grad_buf = try graph.arena.alloc(@as(usize, n));

    if (self.bucket) |bucket_buf| {
        // [batch, num_buckets]: loss is per-sample (inv_n = 1/batch).
        const dims = predicted.shape.dims();
        const batch: u32 = @intCast(dims[0]);
        const num_buckets: u32 = @intCast(dims[1]);
        const inv_n: f32 = 1.0 / @as(f32, @floatFromInt(batch));

        try graph.ops.mseLossBucketed(
            graph.ctx,
            predicted.storage.gpu.buffer,
            target,
            bucket_buf,
            elem_loss_buf,
            grad_buf,
            n,
            num_buckets,
            scale,
            inv_n,
        );

        const loss_sum = try graph.ops.reduceSum(graph.ctx, elem_loss_buf, graph.allocator);
        const loss = loss_sum * inv_n;

        const g = try graph.allocator.create(Tensor);
        g.* = .{
            .shape = predicted.shape,
            .storage = .{ .gpu = .{ .buffer = grad_buf } },
            .allocator = graph.allocator,
        };
        predicted.grad = g;

        return loss;
    }

    const inv_n: f32 = 1.0 / @as(f32, @floatFromInt(n));

    try graph.ops.mseLoss(
        graph.ctx,
        predicted.storage.gpu.buffer,
        target,
        elem_loss_buf,
        grad_buf,
        n,
        scale,
        inv_n,
    );

    // Scalar loss = mean of element losses
    const loss_sum = try graph.ops.reduceSum(graph.ctx, elem_loss_buf, graph.allocator);
    const loss = loss_sum * inv_n;

    // Set gradient on predicted for backward pass
    const g = try graph.allocator.create(Tensor);
    g.* = .{
        .shape = predicted.shape,
        .storage = .{ .gpu = .{ .buffer = grad_buf } },
        .allocator = graph.allocator,
    };
    predicted.grad = g;

    return loss;
}
