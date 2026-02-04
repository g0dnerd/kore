const std = @import("std");

pub fn build(b: *std.Build) void {
    const args = b.dependency("args", .{}).module("args");
    // const net = b.dependency("net", .{}).module("net");

    const kore = b.addModule("kore", .{ .root_source_file = b.path("src/kore.zig") });
    kore.addImport("args", args);
    // kore.addImport("net", net);
}
