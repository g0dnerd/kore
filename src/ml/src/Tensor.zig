const std = @import("std");
const Shape = @import("Shape.zig");
const Storage = @import("Storage.zig").Storage;

const Tensor = @This();

shape: Shape,
storage: Storage,
grad: ?*Tensor = null,
tape_index: ?usize = null,
requires_grad: bool = false,
allocator: std.mem.Allocator,

/// Allocate a zero-initialized CPU tensor.
pub fn alloc(allocator: std.mem.Allocator, shape_dims: []const usize) !Tensor {
    const shape = Shape.init(shape_dims);
    const n = shape.elements();
    const d = try allocator.alloc(f32, n);
    @memset(d, 0);
    return .{
        .shape = shape,
        .storage = .{ .cpu = .{ .data = d } },
        .allocator = allocator,
    };
}

/// Create a CPU tensor from existing data (copies the slice).
pub fn fromSlice(allocator: std.mem.Allocator, shape_dims: []const usize, src: []const f32) !Tensor {
    const shape = Shape.init(shape_dims);
    std.debug.assert(src.len == shape.elements());
    const d = try allocator.dupe(f32, src);
    return .{
        .shape = shape,
        .storage = .{ .cpu = .{ .data = d } },
        .allocator = allocator,
    };
}

/// Free all owned memory including gradient tensor.
pub fn deinit(self: *Tensor) void {
    if (self.grad) |g| {
        g.deinit();
        self.allocator.destroy(g);
    }
    self.storage.deinit(self.allocator);
    self.* = undefined;
}

/// Access the underlying CPU f32 data.
pub fn data(self: *const Tensor) []f32 {
    return self.storage.data();
}

/// Total number of elements.
pub fn elements(self: *const Tensor) usize {
    return self.shape.elements();
}

/// Comptime-parameterized static tensor for CPU inference.
/// Stack-allocated, zero-allocation. Used with cpu/ops for SIMD inference.
pub fn Static(comptime T: type, comptime shape_dims: []const usize) type {
    return struct {
        const Self = @This();
        pub const dtype = T;
        pub const shape = shape_dims;
        pub const ndims = shape_dims.len;
        pub const len = Shape.comptimeElements(shape_dims);
        pub const stride = Shape.comptimeStrides(shape_dims);

        data: [len]T,

        pub fn zeroes() Self {
            var s: Self = undefined;
            @memset(&s.data, 0);
            return s;
        }

        pub fn init(d: [len]T) Self {
            return .{ .data = d };
        }

        pub fn fill(val: T) Self {
            var s: Self = undefined;
            @memset(&s.data, val);
            return s;
        }

        pub fn get(self: *const Self, indices: [ndims]usize) T {
            var offset: usize = 0;
            inline for (0..ndims) |i| {
                offset += indices[i] * stride[i];
            }
            return self.data[offset];
        }

        pub fn set(self: *Self, indices: [ndims]usize, val: T) void {
            var offset: usize = 0;
            inline for (0..ndims) |i| {
                offset += indices[i] * stride[i];
            }
            self.data[offset] = val;
        }
    };
}

test "alloc and deinit" {
    var t = try Tensor.alloc(std.testing.allocator, &.{ 2, 3 });
    defer t.deinit();

    try std.testing.expectEqual(@as(usize, 6), t.elements());
    for (t.data()) |v| {
        try std.testing.expectEqual(@as(f32, 0), v);
    }
}

test "fromSlice" {
    const src = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var t = try Tensor.fromSlice(std.testing.allocator, &.{ 2, 3 }, &src);
    defer t.deinit();

    try std.testing.expectEqualSlices(f32, &src, t.data());
}

test "Static zeroes" {
    const T = Tensor.Static(f32, &.{ 2, 3 });
    const t = T.zeroes();
    try std.testing.expectEqual(@as(usize, 6), T.len);
    for (t.data) |v| {
        try std.testing.expectEqual(@as(f32, 0), v);
    }
}

test "Static init and access" {
    const T = Tensor.Static(f32, &.{ 2, 3 });
    var t = T.init(.{ 1, 2, 3, 4, 5, 6 });
    try std.testing.expectEqual(@as(f32, 1), t.get(.{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 6), t.get(.{ 1, 2 }));
    t.set(.{ 0, 1 }, 42);
    try std.testing.expectEqual(@as(f32, 42), t.data[1]);
}
