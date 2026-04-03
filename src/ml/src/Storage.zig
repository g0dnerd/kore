const std = @import("std");
const GpuBuffer = @import("gpu/Buffer.zig");

pub const Storage = union(enum) {
    cpu: Cpu,
    gpu: Gpu,

    pub const Cpu = struct {
        data: []f32,
    };

    pub const Gpu = struct {
        buffer: GpuBuffer,
    };

    pub fn deinit(self: *Storage, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .cpu => |s| allocator.free(s.data),
            .gpu => |s| {
                var buf = s.buffer;
                buf.release();
            },
        }
        self.* = undefined;
    }

    pub fn data(self: Storage) []f32 {
        return switch (self) {
            .cpu => |s| s.data,
            .gpu => unreachable,
        };
    }
};

test "cpu storage" {
    const allocator = std.testing.allocator;
    const d = try allocator.alloc(f32, 4);
    @memcpy(d, &[_]f32{ 1, 2, 3, 4 });

    var storage: Storage = .{ .cpu = .{ .data = d } };
    defer storage.deinit(allocator);

    try std.testing.expectEqual(@as(f32, 3), storage.data()[2]);
}
