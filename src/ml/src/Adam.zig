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
    weight_decay: f32 = 0,
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
        const buf = p.storage.gpu.buffer;
        if (self.config.weight_decay > 0) {
            try self.ops.weightedAdd(self.ctx, buf, buf, buf, 1.0 - self.config.lr * self.config.weight_decay, 0.0, s.n);
        }
        try self.ops.adamUpdate(
            self.ctx,
            buf,
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

test "weight decay shrinks parameters toward zero" {
    const allocator = std.testing.allocator;
    var ctx = try Context.init();
    defer ctx.deinit();
    var ops_ = try Ops.init(&ctx);
    defer ops_.deinit();

    const n = 4;
    const init_vals = [n]f32{ 1.0, -2.0, 3.0, -4.0 };
    const zero_grad = [n]f32{ 0.0, 0.0, 0.0, 0.0 };

    var grad_tensor = try allocator.create(Tensor);
    grad_tensor.* = .{
        .shape = .init(&.{n}),
        .storage = .{ .gpu = .{ .buffer = try Buffer.upload(&ctx, &zero_grad) } },
        .allocator = allocator,
    };

    var param = Tensor{
        .shape = .init(&.{n}),
        .storage = .{ .gpu = .{ .buffer = try Buffer.upload(&ctx, &init_vals) } },
        .grad = grad_tensor,
        .allocator = allocator,
    };
    defer {
        grad_tensor.storage.gpu.buffer.release();
        allocator.destroy(grad_tensor);
        param.grad = null;
        param.storage.gpu.buffer.release();
    }

    const lr: f32 = 0.1;
    const wd: f32 = 0.5;
    var params = [_]*Tensor{&param};

    var adam = try Adam.init(allocator, &ctx, &ops_, &params, .{
        .lr = lr,
        .weight_decay = wd,
        .beta1 = 0.0,
        .beta2 = 0.0,
        .eps = 1.0,
    });
    defer adam.deinit();

    try adam.step(&params);

    var result: [n]f32 = undefined;
    try param.storage.gpu.buffer.download(&ctx, &result);

    // With zero gradient, beta1=0, beta2=0, eps=1:
    //   Adam update = lr * (0 / (0 + 1)) = 0
    //   Only weight decay applies: param *= (1 - lr * wd) = 0.95
    const decay = 1.0 - lr * wd;
    for (0..n) |i| {
        try std.testing.expectApproxEqAbs(init_vals[i] * decay, result[i], 1e-5);
    }
}

test "weight decay zero has no effect" {
    const allocator = std.testing.allocator;
    var ctx = try Context.init();
    defer ctx.deinit();
    var ops_ = try Ops.init(&ctx);
    defer ops_.deinit();

    const n = 4;
    const init_vals = [n]f32{ 1.0, -2.0, 3.0, -4.0 };
    const zero_grad = [n]f32{ 0.0, 0.0, 0.0, 0.0 };

    var grad_tensor = try allocator.create(Tensor);
    grad_tensor.* = .{
        .shape = .init(&.{n}),
        .storage = .{ .gpu = .{ .buffer = try Buffer.upload(&ctx, &zero_grad) } },
        .allocator = allocator,
    };

    var param = Tensor{
        .shape = .init(&.{n}),
        .storage = .{ .gpu = .{ .buffer = try Buffer.upload(&ctx, &init_vals) } },
        .grad = grad_tensor,
        .allocator = allocator,
    };
    defer {
        grad_tensor.storage.gpu.buffer.release();
        allocator.destroy(grad_tensor);
        param.grad = null;
        param.storage.gpu.buffer.release();
    }

    var params = [_]*Tensor{&param};

    var adam = try Adam.init(allocator, &ctx, &ops_, &params, .{
        .lr = 0.1,
        .weight_decay = 0,
        .beta1 = 0.0,
        .beta2 = 0.0,
        .eps = 1.0,
    });
    defer adam.deinit();

    try adam.step(&params);

    var result: [n]f32 = undefined;
    try param.storage.gpu.buffer.download(&ctx, &result);

    for (0..n) |i| {
        try std.testing.expectApproxEqAbs(init_vals[i], result[i], 1e-5);
    }
}
