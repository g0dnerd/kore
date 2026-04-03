const std = @import("std");
const Context = @import("Context.zig");
const Buffer = @import("Buffer.zig");
const cl = Context.cl;
const log = std.log.scoped(.opencl);

const Program = @This();

program: cl.cl_program,
kernel: cl.cl_kernel,

/// Compile an OpenCL source string and extract the named kernel.
pub fn create(ctx: *const Context, source: []const u8, name: [:0]const u8) Context.Error!Program {
    var err: cl.cl_int = undefined;
    var src_ptr: [*c]const u8 = source.ptr;
    var src_len = source.len;
    const program = cl.clCreateProgramWithSource(ctx.context, 1, &src_ptr, &src_len, &err);
    try Context.check(err);
    errdefer _ = cl.clReleaseProgram(program);

    Context.check(cl.clBuildProgram(program, 1, &ctx.device, null, null, null)) catch |e| {
        var log_size: usize = 0;
        _ = cl.clGetProgramBuildInfo(
            program,
            ctx.device,
            cl.CL_PROGRAM_BUILD_LOG,
            0,
            null,
            &log_size,
        );
        if (log_size > 0) {
            var buf: [4096]u8 = undefined;
            const capped = @min(log_size, buf.len);
            _ = cl.clGetProgramBuildInfo(
                program,
                ctx.device,
                cl.CL_PROGRAM_BUILD_LOG,
                capped,
                &buf,
                null,
            );
            log.err("OpenCL build failure:\n{s}", .{buf[0..capped]});
        }
        return e;
    };

    const kernel = cl.clCreateKernel(program, name.ptr, &err);
    try Context.check(err);

    return .{
        .program = program,
        .kernel = kernel,
    };
}

/// Set a kernel argument.
/// Accepts a Buffer (passes the `cl_mem` handle) or a scalar value.
pub fn setArg(self: *const Program, index: u32, value: anytype) Context.Error!void {
    const T = @TypeOf(value);
    if (T == Buffer) {
        try Context.check(cl.clSetKernelArg(
            self.kernel,
            @intCast(index),
            @sizeOf(cl.cl_mem),
            @ptrCast(&value.mem),
        ));
    } else {
        try Context.check(cl.clSetKernelArg(
            self.kernel,
            @intCast(index),
            @sizeOf(T),
            @ptrCast(&value),
        ));
    }
}

/// Set a local memory kernel argument (allocates work-group-local memory on the device).
pub fn setArgLocal(self: *const Program, index: u32, size: usize) Context.Error!void {
    try Context.check(cl.clSetKernelArg(
        self.kernel,
        @intCast(index),
        size,
        null,
    ));
}

/// Enqueue the kernel with the given work sizes and block until completion.
pub fn dispatch(
    self: *const Program,
    ctx: *const Context,
    global: []const usize,
    local: ?[]const usize,
) Context.Error!void {
    const local_ptr: ?[*]const usize = if (local) |l| l.ptr else null;
    try Context.check(cl.clEnqueueNDRangeKernel(
        ctx.queue,
        self.kernel,
        @intCast(global.len),
        null,
        global.ptr,
        local_ptr,
        0,
        null,
        null,
    ));
    try Context.check(cl.clFinish(ctx.queue));
}

/// Release the kernel and program objects.
pub fn release(self: *Program) void {
    _ = cl.clReleaseKernel(self.kernel);
    _ = cl.clReleaseProgram(self.program);
    self.* = undefined;
}

test "compile fill kernel" {
    var ctx = try Context.init();
    defer ctx.deinit();

    var prog = try create(&ctx, @embedFile("kernels/fill.cl"), "fill");
    defer prog.release();
}

test "fill kernel writes correct values" {
    var ctx = try Context.init();
    defer ctx.deinit();

    var prog = try create(&ctx, @embedFile("kernels/fill.cl"), "fill");
    defer prog.release();

    const n: u32 = 128;
    var buf = try Buffer.alloc(&ctx, n);
    defer buf.release();

    try prog.setArg(0, buf);
    try prog.setArg(1, @as(f32, 3.14));
    try prog.setArg(2, n);
    try prog.dispatch(&ctx, &.{n}, null);

    var result: [128]f32 = undefined;
    try buf.download(&ctx, &result);

    for (result) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 3.14), v, 1e-6);
    }
}

test "matmul: GPU result matches CPU naive multiply" {
    var ctx = try Context.init();
    defer ctx.deinit();

    var prog = try create(&ctx, @embedFile("kernels/matmul.cl"), "matmul");
    defer prog.release();

    // A [2x3] * B [3x4] = C [2x4]
    const M: u32 = 2;
    const K: u32 = 3;
    const N: u32 = 4;

    // A = [[1, 2, 3],
    //      [4, 5, 6]]
    const a_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    // B = [[7,  8,  9,  10],
    //      [11, 12, 13, 14],
    //      [15, 16, 17, 18]]
    const b_data = [_]f32{ 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 };

    var buf_a = try Buffer.upload(&ctx, &a_data);
    defer buf_a.release();
    var buf_b = try Buffer.upload(&ctx, &b_data);
    defer buf_b.release();
    var buf_c = try Buffer.alloc(&ctx, M * N);
    defer buf_c.release();

    try prog.setArg(0, buf_a);
    try prog.setArg(1, buf_b);
    try prog.setArg(2, buf_c);
    try prog.setArg(3, M);
    try prog.setArg(4, N);
    try prog.setArg(5, K);

    // Global work size must be multiple of local (tile) size 16
    const global = [_]usize{ 16, 16 };
    const local = [_]usize{ 16, 16 };
    try prog.dispatch(&ctx, &global, &local);

    var gpu_result: [M * N]f32 = undefined;
    try buf_c.download(&ctx, &gpu_result);

    // CPU naive matmul for reference
    var expected: [M * N]f32 = undefined;
    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0;
            for (0..K) |k| {
                sum += a_data[i * K + k] * b_data[k * N + j];
            }
            expected[i * N + j] = sum;
        }
    }

    // expected = [[74, 80, 86, 92], [173, 188, 203, 218]]
    for (0..M * N) |i| {
        try std.testing.expectApproxEqAbs(expected[i], gpu_result[i], 1e-4);
    }
}

test "matmul: non-tile-aligned dimensions" {
    var ctx = try Context.init();
    defer ctx.deinit();

    var prog = try create(&ctx, @embedFile("kernels/matmul.cl"), "matmul");
    defer prog.release();

    // A [5x7] * B [7x3] = C [5x3]  (none are multiples of tile size 16)
    const M: u32 = 5;
    const K: u32 = 7;
    const N: u32 = 3;

    // Fill with known pattern: a[i][k] = i+k+1, b[k][j] = k-j+1
    var a_data: [M * K]f32 = undefined;
    var b_data: [K * N]f32 = undefined;
    for (0..M) |i| {
        for (0..K) |k| {
            a_data[i * K + k] = @floatFromInt(i + k + 1);
        }
    }
    for (0..K) |k| {
        for (0..N) |j| {
            b_data[k * N + j] = @floatFromInt(k + 1);
            b_data[k * N + j] -= @as(f32, @floatFromInt(j));
        }
    }

    var buf_a = try Buffer.upload(&ctx, &a_data);
    defer buf_a.release();
    var buf_b = try Buffer.upload(&ctx, &b_data);
    defer buf_b.release();
    var buf_c = try Buffer.alloc(&ctx, M * N);
    defer buf_c.release();

    try prog.setArg(0, buf_a);
    try prog.setArg(1, buf_b);
    try prog.setArg(2, buf_c);
    try prog.setArg(3, M);
    try prog.setArg(4, N);
    try prog.setArg(5, K);

    // Round up to tile size
    const global = [_]usize{ 16, 16 };
    const local = [_]usize{ 16, 16 };
    try prog.dispatch(&ctx, &global, &local);

    var gpu_result: [M * N]f32 = undefined;
    try buf_c.download(&ctx, &gpu_result);

    // CPU reference
    var expected: [M * N]f32 = undefined;
    for (0..M) |i| {
        for (0..N) |j| {
            var sum: f32 = 0;
            for (0..K) |k| {
                sum += a_data[i * K + k] * b_data[k * N + j];
            }
            expected[i * N + j] = sum;
        }
    }

    for (0..M * N) |i| {
        try std.testing.expectApproxEqAbs(expected[i], gpu_result[i], 1e-4);
    }
}
