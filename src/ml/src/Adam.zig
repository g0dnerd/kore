const std = @import("std");
const Tensor = @import("Tensor.zig");
const Context = @import("gpu/Context.zig");
const Buffer = @import("gpu/Buffer.zig");
const Ops = @import("gpu/ops.zig");

const Adam = @This();

pub const Config = struct {
    lr: f32 = 1e-3,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    eps: f32 = 1e-8,
};

const ParamState = struct {
    m: Buffer,
    v: Buffer,
    n: u32,
};

states: []ParamState,
config: Config,
step_count: u32 = 0,
ctx: *const Context,
ops: *const Ops,
allocator: std.mem.Allocator,

pub fn init(allocator: std.mem.Allocator, ctx: *const Context, ops: *const Ops, params: []const *Tensor, config: Config) !Adam {
    const states = try allocator.alloc(ParamState, params.len);
    errdefer allocator.free(states);

    var initialized: usize = 0;
    errdefer for (states[0..initialized]) |*s| {
        s.m.release();
        s.v.release();
    };

    for (params, 0..) |p, i| {
        const n: u32 = @intCast(p.elements());
        var m = try Buffer.alloc(ctx, @as(usize, n));
        errdefer m.release();
        try ops.fill(ctx, m, 0.0);

        var v = try Buffer.alloc(ctx, @as(usize, n));
        errdefer v.release();
        try ops.fill(ctx, v, 0.0);

        states[i] = .{ .m = m, .v = v, .n = n };
        initialized = i + 1;
    }

    return .{
        .states = states,
        .config = config,
        .ctx = ctx,
        .ops = ops,
        .allocator = allocator,
    };
}

pub fn step(self: *Adam, params: []const *Tensor) !void {
    self.step_count += 1;
    const beta1_t = std.math.pow(f32, self.config.beta1, @floatFromInt(self.step_count));
    const beta2_t = std.math.pow(f32, self.config.beta2, @floatFromInt(self.step_count));

    for (params, self.states) |p, s| {
        const grad = p.grad orelse continue;
        try self.ops.adamUpdate(
            self.ctx,
            p.storage.gpu.buffer,
            grad.storage.gpu.buffer,
            s.m,
            s.v,
            self.config.lr,
            self.config.beta1,
            self.config.beta2,
            self.config.eps,
            beta1_t,
            beta2_t,
            s.n,
        );
    }
}

/// Get first moment (m) and second moment (v) buffers for serialization.
pub fn getState(self: *const Adam, index: usize) ParamState {
    return self.states[index];
}

pub fn deinit(self: *Adam) void {
    for (self.states) |*s| {
        s.m.release();
        s.v.release();
    }
    self.allocator.free(self.states);
    self.* = undefined;
}
