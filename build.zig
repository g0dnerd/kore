const std = @import("std");

pub fn build(b: *std.Build) void {
    const no_gpu = b.option(bool, "no_gpu", "Build CPU-only, without OpenCL/libc") orelse false;
    const args = b.dependency("args", .{}).module("args");
    const io = b.dependency("io", .{}).module("io");
    const ml = b.dependency("ml", .{ .no_gpu = no_gpu }).module("ml");

    const kore = b.addModule("kore", .{ .root_source_file = b.path("src/kore.zig") });
    kore.addImport("args", args);
    kore.addImport("io", io);
    kore.addImport("ml", ml);
}
