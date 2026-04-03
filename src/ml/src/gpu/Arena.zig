const std = @import("std");
const Buffer = @import("Buffer.zig");
const Context = @import("Context.zig");

const Arena = @This();

buffers: std.ArrayList(Buffer),
ctx: *const Context,
allocator: std.mem.Allocator,

pub fn init(allocator: std.mem.Allocator, ctx: *const Context) Arena {
    return .{
        .buffers = std.ArrayList(Buffer).empty,
        .ctx = ctx,
        .allocator = allocator,
    };
}

/// Allocate a new GPU buffer tracked by the arena.
pub fn alloc(self: *Arena, len: usize) !Buffer {
    var buf = try Buffer.alloc(self.ctx, len);
    errdefer buf.release();
    try self.buffers.append(self.allocator, buf);
    return buf;
}

/// Release all tracked buffers.
pub fn reset(self: *Arena) void {
    for (self.buffers.items) |*buf| {
        buf.release();
    }
    self.buffers.clearRetainingCapacity();
}

/// Release all buffers and free the tracking list.
pub fn deinit(self: *Arena) void {
    self.reset();
    self.buffers.deinit(self.allocator);
    self.* = undefined;
}

test "alloc and reset" {
    var ctx = try Context.init();
    defer ctx.deinit();

    var arena = Arena.init(std.testing.allocator, &ctx);
    defer arena.deinit();

    _ = try arena.alloc(100);
    _ = try arena.alloc(200);
    try std.testing.expectEqual(@as(usize, 2), arena.buffers.items.len);

    arena.reset();
    try std.testing.expectEqual(@as(usize, 0), arena.buffers.items.len);

    _ = try arena.alloc(50);
    try std.testing.expectEqual(@as(usize, 1), arena.buffers.items.len);
}
