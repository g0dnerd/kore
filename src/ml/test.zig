const std = @import("std");

const foo: []const u8 = &.{ 8, 4, 3, 20 };

pub fn main() !void {
    const roo: []const f32 = @floatFromInt(foo);
    std.debug.print("{any}\n", .{roo});
}
