const std = @import("std");
const Tensor = @import("Tensor.zig");
const Buffer = @import("gpu/Buffer.zig");

pub const Op = enum {
    matmul,
    add_bias,
    clipped_relu,
    sigmoid,
    sparse_accumulate,
};

pub const SavedContext = union(enum) {
    matmul: Matmul,
    add_bias: AddBias,
    clipped_relu: ClippedReLU,
    sigmoid: Sigmoid,
    sparse_accumulate: SparseAccumulate,

    pub const Matmul = struct {
        input: *Tensor,
        weight: *Tensor,
        m: u32,
        n: u32,
        k: u32,
    };

    pub const AddBias = struct {
        input: *Tensor,
        bias: *Tensor,
        rows: u32,
        cols: u32,
    };

    pub const ClippedReLU = struct {
        input: *Tensor,
        max_val: f32,
        n: u32,
    };

    pub const Sigmoid = struct {
        input: *Tensor,
        output: *Tensor,
        n: u32,
    };

    pub const SparseAccumulate = struct {
        weights: *Tensor,
        bias: *Tensor,
        active_indices: Buffer,
        num_active: Buffer,
        batch_size: u32,
        max_active: u32,
        out_features: u32,
    };
};

pub const TapeEntry = struct {
    saved: SavedContext,
    output: *Tensor,
};

pub const Tape = struct {
    entries: std.ArrayList(TapeEntry),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Tape {
        return .{
            .entries = .empty,
            .allocator = allocator,
        };
    }

    pub fn append(self: *Tape, entry: TapeEntry) !void {
        try self.entries.append(self.allocator, entry);
    }

    pub fn reset(self: *Tape) void {
        self.entries.clearRetainingCapacity();
    }

    pub fn deinit(self: *Tape) void {
        self.entries.deinit(self.allocator);
        self.* = undefined;
    }
};

test "tape append and reset" {
    const allocator = std.testing.allocator;
    var t = Tape.init(allocator);
    defer t.deinit();

    // Just test the container mechanics (no GPU needed)
    try std.testing.expectEqual(@as(usize, 0), t.entries.items.len);
    t.reset();
    try std.testing.expectEqual(@as(usize, 0), t.entries.items.len);
}
