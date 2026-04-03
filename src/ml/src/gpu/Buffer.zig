const std = @import("std");
const Context = @import("Context.zig");
const cl = Context.cl;

const Buffer = @This();

mem: cl.cl_mem,
len: usize,

/// Allocate an uninitialized f32 buffer of `len` elements on the GPU.
pub fn alloc(ctx: *const Context, len: usize) Context.Error!Buffer {
    var err: cl.cl_int = undefined;
    const mem = cl.clCreateBuffer(
        ctx.context,
        cl.CL_MEM_READ_WRITE,
        len * @sizeOf(f32),
        null,
        &err,
    );
    try Context.check(err);
    return .{ .mem = mem, .len = len };
}

/// Create a GPU buffer and copy `data` into it.
pub fn upload(ctx: *const Context, data: []const f32) Context.Error!Buffer {
    const byte_size = data.len * @sizeOf(f32);
    var err: cl.cl_int = undefined;
    const mem = cl.clCreateBuffer(
        ctx.context,
        cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,
        byte_size,
        @ptrCast(@constCast(data.ptr)),
        &err,
    );
    try Context.check(err);
    return .{ .mem = mem, .len = data.len };
}

/// Blocking read of the buffer contents back to host memory.
pub fn download(self: Buffer, ctx: *const Context, out: []f32) Context.Error!void {
    std.debug.assert(out.len >= self.len);
    try Context.check(cl.clEnqueueReadBuffer(
        ctx.queue,
        self.mem,
        cl.CL_TRUE,
        0,
        self.len * @sizeOf(f32),
        @ptrCast(out.ptr),
        0,
        null,
        null,
    ));
}

/// Blocking write of host memory into the buffer.
pub fn write(self: Buffer, ctx: *const Context, src: []const f32) Context.Error!void {
    std.debug.assert(src.len <= self.len);
    try Context.check(cl.clEnqueueWriteBuffer(
        ctx.queue,
        self.mem,
        cl.CL_TRUE,
        0,
        src.len * @sizeOf(f32),
        @ptrCast(src.ptr),
        0,
        null,
        null,
    ));
}

/// Free the underlying cl_mem object.
pub fn release(self: *Buffer) void {
    _ = cl.clReleaseMemObject(self.mem);
    self.* = undefined;
}

test "alloc, upload, download round-trip" {
    var ctx = try Context.init();
    defer ctx.deinit();

    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var buf = try upload(&ctx, &input);
    defer buf.release();

    var output: [4]f32 = undefined;
    try buf.download(&ctx, &output);

    try std.testing.expectEqualSlices(f32, &input, &output);
}
