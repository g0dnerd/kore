const std = @import("std");

const Self = @This();

writer: *std.Io.Writer,
current_byte: u8 = 0,
byte_progress: u8 = 0,

pub fn init(writer: *std.Io.Writer) Self {
    return .{ .writer = writer };
}

pub fn write(self: *Self, data: u8, num_bits: u4) !void {
    const first_chunk_size: u4 = @intCast(@min(8 - self.byte_progress, num_bits));
    const first_chunk_mask: u8 = @intCast((@as(u16, 1) << first_chunk_size) - 1);

    self.current_byte |= (data & first_chunk_mask) << @intCast(self.byte_progress);
    self.byte_progress += first_chunk_size;

    if (self.byte_progress == 8) {
        try self.writer.writeByte(self.current_byte);
        self.byte_progress = 0;
        self.current_byte = 0;
    }

    if (first_chunk_size >= num_bits) return;

    const second_chunk_size: u4 = @intCast(num_bits - first_chunk_size);
    const second_chunk_mask: u8 = @intCast((@as(u16, 1) << second_chunk_size) - 1);

    self.current_byte |= (data >> @intCast(first_chunk_size)) & second_chunk_mask;

    self.byte_progress += second_chunk_size;
}

// TODO: this should invalidate the writer somehow
pub fn finish(self: *Self) !void {
    if (self.byte_progress > 0) {
        try self.writer.writeByte(self.current_byte);
        self.byte_progress = 0;
        self.current_byte = 0;
    }
    try self.flush();
}

pub fn flush(self: *Self) !void {
    try self.writer.flush();
}

test "full byte" {
    var buf: [1]u8 = undefined;

    var writer: std.Io.Writer = .fixed(&buf);
    var bw = Self.init(&writer);
    try bw.write('a', 8);
    try bw.finish();

    try std.testing.expect(std.mem.eql(u8, &buf, "a"));
}

test "sub-byte writes combine into one byte" {
    // Write 3 bits then 5 bits — should produce one byte, LSB-first packing
    var buf: [1]u8 = undefined;

    var writer: std.Io.Writer = .fixed(&buf);
    var bw = Self.init(&writer);
    try bw.write(0b101, 3); // bits [0:3] = 101
    try bw.write(0b10110, 5); // bits [3:8] = 10110
    try bw.finish();

    // Expected: 10110_101 = 0xB5
    try std.testing.expectEqual(0b10110_101, buf[0]);
}

test "single bits" {
    var buf: [1]u8 = undefined;

    var writer: std.Io.Writer = .fixed(&buf);
    var bw = Self.init(&writer);
    // Write 8 individual bits: 1,0,1,0,0,1,1,0 (LSB first → byte = 0b01100101 = 0x65)
    try bw.write(1, 1);
    try bw.write(0, 1);
    try bw.write(1, 1);
    try bw.write(0, 1);
    try bw.write(0, 1);
    try bw.write(1, 1);
    try bw.write(1, 1);
    try bw.write(0, 1);
    try bw.finish();

    try std.testing.expectEqual(0b01100101, buf[0]);
}

test "write spanning byte boundary" {
    var buf: [2]u8 = undefined;

    var writer: std.Io.Writer = .fixed(&buf);
    var bw = Self.init(&writer);
    try bw.write(0b111, 3); // 3 bits into byte 0
    try bw.write(0b11111, 5); // 5 bits fills byte 0
    try bw.write(0b110, 3); // 3 bits into byte 1
    try bw.write(0b10, 2); // 2 bits into byte 1 (total 5)
    try bw.finish();

    // byte 0: 11111_111 = 0xFF
    try std.testing.expectEqual(@as(u8, 0xFF), buf[0]);
    // byte 1: 000_10_110 = 0b00010110 = 0x16
    try std.testing.expectEqual(@as(u8, 0b00010110), buf[1]);
}

test "write that crosses byte boundary in a single call" {
    // Start with 6 bits written, then write 5 bits — should split across two bytes
    var buf: [2]u8 = undefined;

    var writer: std.Io.Writer = .fixed(&buf);
    var bw = Self.init(&writer);
    try bw.write(0b110011, 6); // 6 bits into byte 0
    try bw.write(0b10101, 5); // 2 bits complete byte 0, 3 bits into byte 1
    try bw.finish();

    // byte 0: bits[0:6]=110011, bits[6:8]=01 (low 2 bits of 10101) → 01_110011 = 0b01110011 = 0x73
    try std.testing.expectEqual(@as(u8, 0b01_110011), buf[0]);
    // byte 1: bits[0:3]=101 (high 3 bits of 10101) → 00000_101 = 0x05
    try std.testing.expectEqual(@as(u8, 0b00000_101), buf[1]);
}

test "multiple full bytes" {
    var buf: [3]u8 = undefined;

    var writer: std.Io.Writer = .fixed(&buf);
    var bw = Self.init(&writer);
    try bw.write(0xAA, 8);
    try bw.write(0xBB, 8);
    try bw.write(0xCC, 8);
    try bw.finish();

    try std.testing.expectEqual(@as(u8, 0xAA), buf[0]);
    try std.testing.expectEqual(@as(u8, 0xBB), buf[1]);
    try std.testing.expectEqual(@as(u8, 0xCC), buf[2]);
}

test "finish with partial byte pads with zeros" {
    var buf: [1]u8 = undefined;

    var writer: std.Io.Writer = .fixed(&buf);
    var bw = Self.init(&writer);
    try bw.write(0b101, 3);
    try bw.finish();

    // Only 3 bits written, high 5 bits should be 0
    try std.testing.expectEqual(@as(u8, 0b00000_101), buf[0]);
}

test "finish is idempotent" {
    var buf: [2]u8 = .{ 0, 0 };

    var writer: std.Io.Writer = .fixed(&buf);
    var bw = Self.init(&writer);
    try bw.write(0xFF, 4); // partial byte
    try bw.finish();

    try std.testing.expectEqual(@as(u8, 0x0F), buf[0]);

    // Second finish should not write another byte
    try bw.finish();
    try std.testing.expectEqual(@as(u8, 0x00), buf[1]);
}

test "write zero bits is a no-op" {
    var buf: [1]u8 = undefined;

    var writer: std.Io.Writer = .fixed(&buf);
    var bw = Self.init(&writer);
    try bw.write(0xFF, 0);
    try bw.write(0xAB, 8);
    try bw.finish();

    try std.testing.expectEqual(@as(u8, 0xAB), buf[0]);
}

test "4-bit nibbles" {
    var buf: [1]u8 = undefined;

    var writer: std.Io.Writer = .fixed(&buf);
    var bw = Self.init(&writer);
    try bw.write(0x0A, 4); // low nibble
    try bw.write(0x05, 4); // high nibble
    try bw.finish();

    // byte = 0101_1010 = 0x5A
    try std.testing.expectEqual(@as(u8, 0x5A), buf[0]);
}
