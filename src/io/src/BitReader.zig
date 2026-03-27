const std = @import("std");

const Self = @This();

reader: *std.Io.Reader,
current_byte: u8 = 0,
byte_progress: u8 = 8,

pub fn init(reader: *std.Io.Reader) Self {
    return .{ .reader = reader };
}

pub fn read(self: *Self, num_bits: u4) !u8 {
    if (num_bits == 0) return 0;

    if (self.byte_progress == 8) {
        self.current_byte = try self.reader.takeByte();
        self.byte_progress = 0;
    }

    const first_chunk_size: u4 = @intCast(@min(8 - self.byte_progress, num_bits));
    const first_chunk_mask: u8 = @intCast((@as(u16, 1) << first_chunk_size) - 1);

    var result: u8 = (self.current_byte >> @intCast(self.byte_progress)) & first_chunk_mask;
    self.byte_progress += first_chunk_size;

    if (first_chunk_size >= num_bits) return result;

    self.current_byte = try self.reader.takeByte();
    self.byte_progress = 0;

    const second_chunk_size: u4 = @intCast(num_bits - first_chunk_size);
    const second_chunk_mask: u8 = @intCast((@as(u16, 1) << second_chunk_size) - 1);

    result |= (self.current_byte & second_chunk_mask) << @intCast(first_chunk_size);
    self.byte_progress += second_chunk_size;

    return result;
}

test "full byte" {
    const data = [_]u8{'a'};
    var reader: std.Io.Reader = .fixed(&data);
    var br = Self.init(&reader);

    try std.testing.expectEqual(@as(u8, 'a'), try br.read(8));
}

test "sub-byte reads" {
    // 0b10110_101 was written as: write(0b101, 3), write(0b10110, 5)
    const data = [_]u8{0b10110_101};
    var reader: std.Io.Reader = .fixed(&data);
    var br = Self.init(&reader);

    try std.testing.expectEqual(@as(u8, 0b101), try br.read(3));
    try std.testing.expectEqual(@as(u8, 0b10110), try br.read(5));
}

test "single bits" {
    // 0b01100101 = bits 1,0,1,0,0,1,1,0 (LSB first)
    const data = [_]u8{0b01100101};
    var reader: std.Io.Reader = .fixed(&data);
    var br = Self.init(&reader);

    try std.testing.expectEqual(@as(u8, 1), try br.read(1));
    try std.testing.expectEqual(@as(u8, 0), try br.read(1));
    try std.testing.expectEqual(@as(u8, 1), try br.read(1));
    try std.testing.expectEqual(@as(u8, 0), try br.read(1));
    try std.testing.expectEqual(@as(u8, 0), try br.read(1));
    try std.testing.expectEqual(@as(u8, 1), try br.read(1));
    try std.testing.expectEqual(@as(u8, 1), try br.read(1));
    try std.testing.expectEqual(@as(u8, 0), try br.read(1));
}

test "read spanning byte boundary" {
    // Two bytes: 0xFF, 0b00010110
    const data = [_]u8{ 0xFF, 0b00010110 };
    var reader: std.Io.Reader = .fixed(&data);
    var br = Self.init(&reader);

    try std.testing.expectEqual(@as(u8, 0b111), try br.read(3));
    try std.testing.expectEqual(@as(u8, 0b11111), try br.read(5));
    try std.testing.expectEqual(@as(u8, 0b110), try br.read(3));
    try std.testing.expectEqual(@as(u8, 0b10), try br.read(2));
}

test "read that crosses byte boundary in a single call" {
    // Written as: write(0b110011, 6), write(0b10101, 5)
    const data = [_]u8{ 0b01_110011, 0b00000_101 };
    var reader: std.Io.Reader = .fixed(&data);
    var br = Self.init(&reader);

    try std.testing.expectEqual(@as(u8, 0b110011), try br.read(6));
    try std.testing.expectEqual(@as(u8, 0b10101), try br.read(5));
}

test "multiple full bytes" {
    const data = [_]u8{ 0xAA, 0xBB, 0xCC };
    var reader: std.Io.Reader = .fixed(&data);
    var br = Self.init(&reader);

    try std.testing.expectEqual(@as(u8, 0xAA), try br.read(8));
    try std.testing.expectEqual(@as(u8, 0xBB), try br.read(8));
    try std.testing.expectEqual(@as(u8, 0xCC), try br.read(8));
}

test "read zero bits is a no-op" {
    const data = [_]u8{0xAB};
    var reader: std.Io.Reader = .fixed(&data);
    var br = Self.init(&reader);

    try std.testing.expectEqual(@as(u8, 0), try br.read(0));
    try std.testing.expectEqual(@as(u8, 0xAB), try br.read(8));
}

test "4-bit nibbles" {
    // 0x5A = 0b0101_1010
    const data = [_]u8{0x5A};
    var reader: std.Io.Reader = .fixed(&data);
    var br = Self.init(&reader);

    try std.testing.expectEqual(@as(u8, 0x0A), try br.read(4));
    try std.testing.expectEqual(@as(u8, 0x05), try br.read(4));
}

test "roundtrip with BitWriter" {
    const BitWriter = @import("BitWriter.zig");
    var buf: [3]u8 = undefined;

    var writer: std.Io.Writer = .fixed(&buf);
    var bw = BitWriter.init(&writer);
    try bw.write(0b101, 3);
    try bw.write(0b10110, 5);
    try bw.write(0b110011, 6);
    try bw.write(0b10101, 5);
    try bw.write(0b1, 1);
    try bw.finish();

    var reader: std.Io.Reader = .fixed(&buf);
    var br = Self.init(&reader);
    try std.testing.expectEqual(@as(u8, 0b101), try br.read(3));
    try std.testing.expectEqual(@as(u8, 0b10110), try br.read(5));
    try std.testing.expectEqual(@as(u8, 0b110011), try br.read(6));
    try std.testing.expectEqual(@as(u8, 0b10101), try br.read(5));
    try std.testing.expectEqual(@as(u8, 0b1), try br.read(1));
}
