const std = @import("std");

const Shape = @This();

pub const max_dims = 8;

dims_buf: [max_dims]usize = .{0} ** max_dims,
ndims: usize = 0,

/// Create a shape from a slice of dimension sizes.
pub fn init(d: []const usize) Shape {
    std.debug.assert(d.len > 0 and d.len <= max_dims);
    var self: Shape = .{ .ndims = d.len };
    @memcpy(self.dims_buf[0..d.len], d);
    return self;
}

/// A scalar shape: 0 dimensions, 1 element.
pub fn scalar() Shape {
    return .{};
}

pub fn dims(self: *const Shape) []const usize {
    return self.dims_buf[0..self.ndims];
}

pub fn rank(self: *const Shape) usize {
    return self.ndims;
}

pub fn elements(self: *const Shape) usize {
    if (self.ndims == 0) return 1;
    var n: usize = 1;
    for (self.dims()) |d| n *= d;
    return n;
}

/// Row-major strides.
pub fn strides(self: *const Shape) [max_dims]usize {
    var s: [max_dims]usize = .{0} ** max_dims;
    if (self.ndims == 0) return s;
    s[self.ndims - 1] = 1;
    if (self.ndims == 1) return s;
    var i: usize = self.ndims - 1;
    while (i > 0) {
        i -= 1;
        s[i] = s[i + 1] * self.dims_buf[i + 1];
    }
    return s;
}

pub fn eql(self: *const Shape, other: *const Shape) bool {
    if (self.ndims != other.ndims) return false;
    return std.mem.eql(usize, self.dims(), other.dims());
}

/// Compute total element count from comptime-known dimensions.
pub fn comptimeElements(comptime d: []const usize) comptime_int {
    var n: comptime_int = 1;
    for (d) |dim| n *= dim;
    return n;
}

/// Compute row-major strides from comptime-known dimensions.
pub fn comptimeStrides(comptime d: []const usize) [d.len]usize {
    if (d.len == 0) return .{};
    var s: [d.len]usize = undefined;
    s[d.len - 1] = 1;
    if (d.len == 1) return s;
    var i: usize = d.len - 1;
    while (i > 0) {
        i -= 1;
        s[i] = s[i + 1] * d[i + 1];
    }
    return s;
}

test "init and dims" {
    const s = Shape.init(&.{ 2, 3, 4 });
    try std.testing.expectEqual(@as(usize, 3), s.rank());
    try std.testing.expectEqualSlices(usize, &.{ 2, 3, 4 }, s.dims());
}

test "elements" {
    const s1 = Shape.init(&.{ 2, 3, 4 });
    try std.testing.expectEqual(@as(usize, 24), s1.elements());

    const s2 = Shape.scalar();
    try std.testing.expectEqual(@as(usize, 1), s2.elements());

    const s3 = Shape.init(&.{5});
    try std.testing.expectEqual(@as(usize, 5), s3.elements());
}

test "strides" {
    const s = Shape.init(&.{ 2, 3, 4 });
    const st = s.strides();
    try std.testing.expectEqual(@as(usize, 12), st[0]);
    try std.testing.expectEqual(@as(usize, 4), st[1]);
    try std.testing.expectEqual(@as(usize, 1), st[2]);
}

test "eql" {
    const a = Shape.init(&.{ 2, 3 });
    const b = Shape.init(&.{ 2, 3 });
    const c = Shape.init(&.{ 3, 2 });
    try std.testing.expect(a.eql(&b));
    try std.testing.expect(!a.eql(&c));
}

test "comptime helpers" {
    try std.testing.expectEqual(@as(usize, 24), Shape.comptimeElements(&.{ 2, 3, 4 }));
    const s = Shape.comptimeStrides(&.{ 2, 3, 4 });
    try std.testing.expectEqual(@as(usize, 12), s[0]);
    try std.testing.expectEqual(@as(usize, 4), s[1]);
    try std.testing.expectEqual(@as(usize, 1), s[2]);
}
