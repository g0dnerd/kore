const std = @import("std");
const Tensor = @import("Tensor.zig");
const Shape = @import("Shape.zig");
const Adam = @import("Adam.zig");
const Context = @import("gpu/Context.zig");
const Buffer = @import("gpu/Buffer.zig");

const file_magic: [4]u8 = "KTML".*;
const file_version: u32 = 1;

pub const Metadata = struct {
    epoch: u32 = 0,
    step: u32 = 0,
    learning_rate: f32 = 0,
    best_val_loss: f32 = 0,
    adam_step: u32 = 0,
};

pub const NamedParam = struct {
    name: []const u8,
    tensor: *Tensor,
};

pub fn save(
    allocator: std.mem.Allocator,
    ctx: *const Context,
    path: []const u8,
    params: []const NamedParam,
    adam: *const Adam,
    metadata: Metadata,
) !void {
    // Pre-serialize metadata JSON to know its length for the header
    var json_buf: [512]u8 = undefined;
    var json_w = std.Io.Writer.fixed(&json_buf);
    var stringify: std.json.Stringify = .{ .writer = &json_w };
    try stringify.write(metadata);
    const meta_json = json_w.buffered();

    // Open file
    var single_threaded: std.Io.Threaded = .init_single_threaded;
    const io = single_threaded.io();
    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var writer_buf: [4096]u8 = undefined;
    var w = file.writer(io, &writer_buf);
    var writer = &w.interface;

    // Header
    try writer.writeAll(&file_magic);
    try writer.writeInt(u32, file_version, .native);
    try writer.writeInt(u32, @intCast(params.len), .native);
    try writer.writeInt(u32, @intCast(meta_json.len), .native);

    // Metadata
    try writer.writeAll(meta_json);

    // Reusable download buffer
    var max_n: usize = 0;
    for (params) |p| max_n = @max(max_n, p.tensor.elements());
    const tmp = try allocator.alloc(f32, max_n);
    defer allocator.free(tmp);

    for (params, 0..) |p, i| {
        const n = p.tensor.elements();

        // Name
        try writer.writeInt(u32, @intCast(p.name.len), .native);
        try writer.writeAll(p.name);

        // Num elements
        try writer.writeInt(u32, @intCast(n), .native);

        // Weights
        try p.tensor.storage.gpu.buffer.download(ctx, tmp[0..n]);
        try writer.writeAll(std.mem.sliceAsBytes(tmp[0..n]));

        // Adam first moment (m)
        const state = adam.getState(i);
        try state.m.download(ctx, tmp[0..n]);
        try writer.writeAll(std.mem.sliceAsBytes(tmp[0..n]));

        // Adam second moment (v)
        try state.v.download(ctx, tmp[0..n]);
        try writer.writeAll(std.mem.sliceAsBytes(tmp[0..n]));
    }

    try writer.flush();
}

pub fn load(
    allocator: std.mem.Allocator,
    ctx: *const Context,
    path: []const u8,
    params: []const NamedParam,
    adam: *Adam,
) !Metadata {
    var single_threaded: std.Io.Threaded = .init_single_threaded;
    const io = single_threaded.io();
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var reader_buf: [4096]u8 = undefined;
    var r = file.reader(io, &reader_buf);
    var reader = &r.interface;

    // Header
    const magic = try reader.take(4);
    if (!std.mem.eql(u8, magic, &file_magic))
        return error.InvalidMagic;

    const version = try reader.takeInt(u32, .native);
    if (version != file_version) return error.UnsupportedVersion;

    const num_params = try reader.takeInt(u32, .native);
    const meta_size = try reader.takeInt(u32, .native);

    // Metadata
    const meta_raw = try reader.take(meta_size);

    const parsed = try std.json.parseFromSlice(Metadata, allocator, meta_raw, .{});
    defer parsed.deinit();
    const metadata = parsed.value;

    // Reusable buffer
    var max_n: usize = 0;

    for (params) |p| {
        max_n = @max(max_n, p.tensor.elements());
    }
    const tmp = try allocator.alloc(f32, max_n);
    defer allocator.free(tmp);

    for (0..num_params) |_| {
        // Name
        const name_len = try reader.takeInt(u32, .native);
        const name = try reader.take(name_len);

        const n: usize = try reader.takeInt(u32, .native);

        // Find param
        const idx = findParam(params, name) orelse return error.UnknownParameter;
        const p = params[idx];
        if (p.tensor.elements() != n) return error.ShapeMismatch;

        const byte_len = n * @sizeOf(f32);

        // Weights
        const w_bytes = try reader.take(byte_len);
        @memcpy(std.mem.sliceAsBytes(tmp[0..n]), w_bytes);
        try p.tensor.storage.gpu.buffer.write(ctx, tmp[0..n]);

        // Adam m
        const state = adam.getState(idx);
        const m_bytes = try reader.take(byte_len);
        @memcpy(std.mem.sliceAsBytes(tmp[0..n]), m_bytes);
        try state.m.write(ctx, tmp[0..n]);

        // Adam v
        const v_bytes = try reader.take(byte_len);
        @memcpy(std.mem.sliceAsBytes(tmp[0..n]), v_bytes);
        try state.v.write(ctx, tmp[0..n]);
    }

    return metadata;
}

fn findParam(params: []const NamedParam, name: []const u8) ?usize {
    for (params, 0..) |p, i| {
        if (std.mem.eql(u8, p.name, name)) return i;
    }
    return null;
}

test "checkpoint save and load round-trip" {
    const allocator = std.testing.allocator;
    var ctx = try Context.init();
    defer ctx.deinit();
    const Ops = @import("gpu/ops.zig");
    var ops_ = try Ops.init(&ctx);
    defer ops_.deinit();

    // Original weights
    const w_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const b_data = [_]f32{ 7.0, 8.0 };

    const w_buf = try Buffer.upload(&ctx, &w_data);
    const weight = try allocator.create(Tensor);
    weight.* = .{
        .shape = Shape.init(&.{ 3, 2 }),
        .storage = .{ .gpu = .{ .buffer = w_buf } },
        .requires_grad = true,
        .allocator = allocator,
    };
    defer {
        weight.deinit();
        allocator.destroy(weight);
    }

    const b_buf = try Buffer.upload(&ctx, &b_data);
    const bias = try allocator.create(Tensor);
    bias.* = .{
        .shape = Shape.init(&.{2}),
        .storage = .{ .gpu = .{ .buffer = b_buf } },
        .requires_grad = true,
        .allocator = allocator,
    };
    defer {
        bias.deinit();
        allocator.destroy(bias);
    }

    const params_arr = [_]*Tensor{ weight, bias };
    var adam = try Adam.init(allocator, &ctx, &ops_, &params_arr, .{ .lr = 1e-3 });
    defer adam.deinit();

    // Write known non-zero values to Adam state
    const m_w = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 };
    const v_w = [_]f32{ 0.01, 0.02, 0.03, 0.04, 0.05, 0.06 };
    const m_b = [_]f32{ 0.7, 0.8 };
    const v_b = [_]f32{ 0.07, 0.08 };

    try adam.getState(0).m.write(&ctx, &m_w);
    try adam.getState(0).v.write(&ctx, &v_w);
    try adam.getState(1).m.write(&ctx, &m_b);
    try adam.getState(1).v.write(&ctx, &v_b);

    const named = [_]NamedParam{
        .{ .name = "layer0.weight", .tensor = weight },
        .{ .name = "layer0.bias", .tensor = bias },
    };

    // Save
    const path = "/tmp/kore_ml_test_ckpt.bin";
    adam.step_count = 42;

    try save(allocator, &ctx, path, &named, &adam, .{
        .epoch = 5,
        .step = 1000,
        .learning_rate = 1e-3,
        .best_val_loss = 0.05,
        .adam_step = adam.step_count,
    });
    var single_threaded: std.Io.Threaded = .init_single_threaded;
    const io = single_threaded.io();
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    // Create fresh parameters (zeroed)
    const fw_buf = try Buffer.alloc(&ctx, 6);
    try ops_.fill(&ctx, fw_buf, 0.0);
    const fresh_w = try allocator.create(Tensor);
    fresh_w.* = .{
        .shape = Shape.init(&.{ 3, 2 }),
        .storage = .{ .gpu = .{ .buffer = fw_buf } },
        .requires_grad = true,
        .allocator = allocator,
    };
    defer {
        fresh_w.deinit();
        allocator.destroy(fresh_w);
    }

    const fb_buf = try Buffer.alloc(&ctx, 2);
    try ops_.fill(&ctx, fb_buf, 0.0);
    const fresh_b = try allocator.create(Tensor);
    fresh_b.* = .{
        .shape = Shape.init(&.{2}),
        .storage = .{ .gpu = .{ .buffer = fb_buf } },
        .requires_grad = true,
        .allocator = allocator,
    };
    defer {
        fresh_b.deinit();
        allocator.destroy(fresh_b);
    }

    const fresh_params = [_]*Tensor{ fresh_w, fresh_b };
    var fresh_adam = try Adam.init(allocator, &ctx, &ops_, &fresh_params, .{ .lr = 1e-3 });
    defer fresh_adam.deinit();

    const fresh_named = [_]NamedParam{
        .{ .name = "layer0.weight", .tensor = fresh_w },
        .{ .name = "layer0.bias", .tensor = fresh_b },
    };

    // Load
    const meta = try load(allocator, &ctx, path, &fresh_named, &fresh_adam);

    // Verify metadata
    try std.testing.expectEqual(@as(u32, 5), meta.epoch);
    try std.testing.expectEqual(@as(u32, 1000), meta.step);
    try std.testing.expectApproxEqAbs(@as(f32, 1e-3), meta.learning_rate, 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.05), meta.best_val_loss, 1e-7);
    try std.testing.expectEqual(@as(u32, 42), meta.adam_step);

    // Restore adam step_count from checkpoint
    fresh_adam.step_count = meta.adam_step;

    // Verify weights
    var out_w: [6]f32 = undefined;
    try fresh_w.storage.gpu.buffer.download(&ctx, &out_w);
    try std.testing.expectEqualSlices(f32, &w_data, &out_w);

    var out_b: [2]f32 = undefined;
    try fresh_b.storage.gpu.buffer.download(&ctx, &out_b);
    try std.testing.expectEqualSlices(f32, &b_data, &out_b);

    // Verify Adam m state
    var out_m_w: [6]f32 = undefined;
    try fresh_adam.getState(0).m.download(&ctx, &out_m_w);
    try std.testing.expectEqualSlices(f32, &m_w, &out_m_w);

    var out_v_w: [6]f32 = undefined;
    try fresh_adam.getState(0).v.download(&ctx, &out_v_w);
    try std.testing.expectEqualSlices(f32, &v_w, &out_v_w);

    // Verify Adam v state
    var out_m_b: [2]f32 = undefined;
    try fresh_adam.getState(1).m.download(&ctx, &out_m_b);
    try std.testing.expectEqualSlices(f32, &m_b, &out_m_b);

    var out_v_b: [2]f32 = undefined;
    try fresh_adam.getState(1).v.download(&ctx, &out_v_b);
    try std.testing.expectEqualSlices(f32, &v_b, &out_v_b);
}
