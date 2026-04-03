const std = @import("std");
pub const cl = @cImport({
    @cInclude("CL/cl.h");
    @cInclude("CL/cl_ext.h");
});

const Context = @This();

platform: cl.cl_platform_id,
device: cl.cl_device_id,
context: cl.cl_context,
queue: cl.cl_command_queue,

pub const Error = error{
    PlatformNotFound,
    DeviceNotFound,
    DeviceNotAvailable,
    CompilerNotAvailable,
    OutOfResources,
    OutOfHostMemory,
    InvalidValue,
    InvalidDevice,
    InvalidContext,
    InvalidQueueProperties,
    InvalidCommandQueue,
    InvalidPlatform,
    InvalidProgram,
    InvalidProgramExecutable,
    InvalidKernelName,
    InvalidKernelDefinition,
    InvalidKernel,
    InvalidArgIndex,
    InvalidArgValue,
    InvalidArgSize,
    InvalidWorkDimension,
    InvalidWorkGroupSize,
    InvalidWorkItemSize,
    InvalidGlobalOffset,
    InvalidBufferSize,
    InvalidMemObject,
    InvalidOperation,
    InvalidBuildOptions,
    BuildProgramFailure,
    Unexpected,
};

/// Convert an OpenCL `cl_int` status code into a Zig error.
pub fn check(code: cl.cl_int) Error!void {
    if (code == cl.CL_SUCCESS) return;
    return switch (code) {
        cl.CL_DEVICE_NOT_FOUND => error.DeviceNotFound,
        cl.CL_DEVICE_NOT_AVAILABLE => error.DeviceNotAvailable,
        cl.CL_COMPILER_NOT_AVAILABLE => error.CompilerNotAvailable,
        cl.CL_MEM_OBJECT_ALLOCATION_FAILURE, cl.CL_OUT_OF_RESOURCES => error.OutOfResources,
        cl.CL_OUT_OF_HOST_MEMORY => error.OutOfHostMemory,
        cl.CL_INVALID_VALUE => error.InvalidValue,
        cl.CL_INVALID_DEVICE => error.InvalidDevice,
        cl.CL_INVALID_CONTEXT => error.InvalidContext,
        cl.CL_INVALID_QUEUE_PROPERTIES => error.InvalidQueueProperties,
        cl.CL_INVALID_COMMAND_QUEUE => error.InvalidCommandQueue,
        cl.CL_INVALID_PLATFORM => error.InvalidPlatform,
        cl.CL_INVALID_PROGRAM => error.InvalidProgram,
        cl.CL_INVALID_PROGRAM_EXECUTABLE => error.InvalidProgramExecutable,
        cl.CL_INVALID_KERNEL_NAME => error.InvalidKernelName,
        cl.CL_INVALID_KERNEL_DEFINITION => error.InvalidKernelDefinition,
        cl.CL_INVALID_KERNEL => error.InvalidKernel,
        cl.CL_INVALID_ARG_INDEX => error.InvalidArgIndex,
        cl.CL_INVALID_ARG_VALUE => error.InvalidArgValue,
        cl.CL_INVALID_ARG_SIZE => error.InvalidArgSize,
        cl.CL_INVALID_WORK_DIMENSION => error.InvalidWorkDimension,
        cl.CL_INVALID_WORK_GROUP_SIZE => error.InvalidWorkGroupSize,
        cl.CL_INVALID_WORK_ITEM_SIZE => error.InvalidWorkItemSize,
        cl.CL_INVALID_GLOBAL_OFFSET => error.InvalidGlobalOffset,
        cl.CL_INVALID_BUFFER_SIZE => error.InvalidBufferSize,
        cl.CL_INVALID_MEM_OBJECT => error.InvalidMemObject,
        cl.CL_INVALID_OPERATION => error.InvalidOperation,
        cl.CL_INVALID_BUILD_OPTIONS => error.InvalidBuildOptions,
        cl.CL_BUILD_PROGRAM_FAILURE => error.BuildProgramFailure,
        else => error.Unexpected,
    };
}

/// Open the first available GPU via OpenCL and create a command queue.
pub fn init() Error!Context {
    var num_platforms: cl.cl_uint = 0;
    try check(cl.clGetPlatformIDs(0, null, &num_platforms));
    if (num_platforms == 0) return error.PlatformNotFound;

    var platform: cl.cl_platform_id = undefined;
    try check(cl.clGetPlatformIDs(1, &platform, null));

    var num_devices: cl.cl_uint = 0;
    const dev_status = cl.clGetDeviceIDs(platform, cl.CL_DEVICE_TYPE_GPU, 0, null, &num_devices);
    if (dev_status != cl.CL_SUCCESS or num_devices == 0) return error.DeviceNotFound;

    var device: cl.cl_device_id = undefined;
    try check(cl.clGetDeviceIDs(platform, cl.CL_DEVICE_TYPE_GPU, 1, &device, null));

    var err: cl.cl_int = undefined;
    const context = cl.clCreateContext(null, 1, &device, null, null, &err);
    try check(err);
    errdefer _ = cl.clReleaseContext(context);

    const queue = cl.clCreateCommandQueueWithProperties(context, device, null, &err);
    try check(err);

    return .{
        .platform = platform,
        .device = device,
        .context = context,
        .queue = queue,
    };
}

/// Release the command queue and context.
pub fn deinit(self: *Context) void {
    _ = cl.clReleaseCommandQueue(self.queue);
    _ = cl.clReleaseContext(self.context);
    self.* = undefined;
}

test "init and deinit" {
    var ctx = try init();
    defer ctx.deinit();

    try std.testing.expect(ctx.platform != null);
    try std.testing.expect(ctx.device != null);
    try std.testing.expect(ctx.context != null);
    try std.testing.expect(ctx.queue != null);
}
