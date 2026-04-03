const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const ml = b.addModule("ml", .{
        .root_source_file = b.path("src/ml.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    ml.linkSystemLibrary("OpenCL", .{});

    const test_step = b.step("test", "");
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/ml.zig"),
            .target = b.graph.host,
            .link_libc = true,
        }),
    });
    tests.root_module.linkSystemLibrary("OpenCL", .{});

    const run_test = b.addRunArtifact(tests);
    test_step.dependOn(&run_test.step);
}
