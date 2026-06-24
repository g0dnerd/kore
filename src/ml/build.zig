const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    // CPU-only build (no OpenCL/libc): for freestanding/wasm targets that only
    // use the SIMD inference ops in cpu/. The gpu/ namespace is never referenced
    // there, so it is not analyzed and the OpenCL link is unnecessary.
    const no_gpu = b.option(bool, "no_gpu", "Build CPU-only, without OpenCL/libc") orelse false;

    const ml = b.addModule("ml", .{
        .root_source_file = b.path("src/ml.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = !no_gpu,
    });
    if (!no_gpu) ml.linkSystemLibrary("OpenCL", .{});

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
