const std = @import("std");

pub fn build(b: *std.Build) void {
    const test_step = b.step("test", "");
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    _ = b.addModule("io", .{
        .root_source_file = b.path("src/io.zig"),
        .target = target,
        .optimize = optimize,
    });

    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/io.zig"),
            .target = b.graph.host,
        }),
    });

    const run_test = b.addRunArtifact(tests);
    test_step.dependOn(&run_test.step);
}
