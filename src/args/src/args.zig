const std = @import("std");

pub const @"comptime" = @import("comptime.zig");
pub const declarative = @import("declarative.zig");

pub const ArgumentParseError = error{
    ExpectedArgument,
    ExpectedValue,
    InvalidValue,
    UnknownArgument,
    MissingRequiredArgument,
};

// Parse an argument value into its corresponding type
pub fn parseValue(comptime T: type, value_raw: []const u8, arg_name: []const u8) !T {
    const info = @typeInfo(T);

    return switch (info) {
        .int => std.fmt.parseInt(T, value_raw, 10) catch {
            std.log.err("Invalid integer value {s} for argument {s}\n", .{ value_raw, arg_name });
            return error.InvalidValue;
        },
        .optional => |opt| @as(T, try parseValue(opt.child, value_raw, arg_name)),
        .pointer => |p| {
            // []const u8
            if (p.size == .slice and p.child == u8 and p.is_const) {
                return value_raw;
            } else return error.Unimplemented;
        },
        .bool => {
            if (std.mem.eql(u8, value_raw, "true")) return true;
            if (std.mem.eql(u8, value_raw, "false")) return false;
            std.log.err("Invalid boolean value {s} for argument {s}\n", .{ value_raw, arg_name });
            return error.InvalidValue;
        },
        else => error.Unimplemented,
    };
}
