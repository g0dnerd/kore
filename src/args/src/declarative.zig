const builtin = @import("builtin");
const std = @import("std");
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const args = @import("args.zig");
const ArgumentParseError = args.ArgumentParseError;
const parseValue = args.parseValue;

pub fn Parser(Args: type) !type {
    if (std.meta.activeTag(@typeInfo(Args)) != .@"struct") return error.InvalidType;

    const all_args = std.meta.fields(Args);
    const num_args = all_args.len;

    return struct {
        pub fn parse(arg_iter: anytype) ArgumentParseError!Args {
            // Track which arguments have been set
            var args_set = [_]bool{false} ** num_args;

            var result: Args = undefined;

            // Skip process name
            _ = arg_iter.skip();

            while (arg_iter.next()) |arg| {
                // Arguments need two leading dashes
                if (arg.len < 3 or !std.mem.eql(u8, arg[0..2], "--")) {
                    std.log.err("Expected argument with two leading dashes ('-'), found {s}\n", .{arg});
                    return error.ExpectedArgument;
                }
                const arg_name = if (arg.len > 1 and arg[1] == '-' and arg[2] == '-') arg[3..] else arg[2..];

                var found = false;

                inline for (0..num_args) |i| {
                    const expected_arg = all_args[i];
                    if (std.mem.eql(u8, arg_name, expected_arg.name)) {
                        found = true;

                        // Optional booleans are treated as flags (arguments without a value)
                        if (expected_arg.type == ?bool) {
                            @field(result, expected_arg.name) = true;
                        } else {
                            // Parse value associated with this argument
                            const value_raw = arg_iter.next() orelse {
                                std.log.err("Expected value for argument {s}, but found none\n", .{expected_arg.name});
                                return error.ExpectedValue;
                            };
                            @field(result, expected_arg.name) = try parseValue(expected_arg.type, value_raw, expected_arg.name);
                        }
                        args_set[i] = true;
                    }
                }

                if (!found) {
                    std.log.err("Unknown argument {s}\n", .{arg[2..]});
                    return error.UnknownArgument;
                }
            }

            inline for (all_args[0..num_args], 0..) |arg, i| {
                if (!args_set[i]) {
                    const type_info = @typeInfo(arg.type);
                    if (type_info == .optional and arg.type != bool) {
                        // Set optional fields to null
                        @field(result, arg.name) = null;
                    } else {
                        return error.MissingRequiredArgument;
                    }
                }
            }

            return result;
        }
    };
}

test "make parser" {
    const Args = struct {
        hi_mom: u8,
        name: []const u8,
    };

    const ArgParser = try Parser(Args);

    const args_with_values = [_][*:0]const u8{ "prog", "--hi_mom", "25", "--name", "steve" };
    var it = if (builtin.os.tag == .windows)
        try std.process.Args.Iterator.initAllocator(.{ .vector = &args_with_values }, std.heap.page_allocator)
    else
        std.process.Args.Iterator.init(.{ .vector = &args_with_values });

    const resolved_args: Args = try ArgParser.parse(&it);

    const hi_mom = resolved_args.hi_mom;
    const name = resolved_args.name;
    try expectEqual(25, hi_mom);
    try expectEqual("steve", name);
}

test "parse string with whitespace" {
    const Args = struct {
        name: []const u8,
    };

    const ArgParser = try Parser(Args);

    const args_with_values = [_][*:0]const u8{ "prog", "--name", "steve oh" };
    var it = if (builtin.os.tag == .windows)
        try std.process.Args.Iterator.initAllocator(.{ .vector = &args_with_values }, std.heap.page_allocator)
    else
        std.process.Args.Iterator.init(.{ .vector = &args_with_values });

    const result: Args = try ArgParser.parse(&it);
    const name = result.name;
    try expectEqual("steve oh", name);
}

test "parse float" {
    const Args = struct {
        value: f32,
    };

    const ArgParser = try Parser(Args);

    const args_with_values = [_][*:0]const u8{ "prog", "--value", "17.49" };
    var it = if (builtin.os.tag == .windows)
        try std.process.Args.Iterator.initAllocator(.{ .vector = &args_with_values }, std.heap.page_allocator)
    else
        std.process.Args.Iterator.init(.{ .vector = &args_with_values });

    const result: Args = try ArgParser.parse(&it);
    const value = result.value;
    try expectEqual(17.49, value);
}

test "invalid args type" {
    const Args = enum {
        this,
        is,
        illegal,
    };

    try std.testing.expectError(error.InvalidType, Parser(Args));
}

test "parse missing optional" {
    const Args = struct {
        required: u8,
        optional: ?u8,
    };
    const ArgParser = try Parser(Args);

    const arguments = [_][*:0]const u8{ "prog", "--required", "42" };
    var it = if (builtin.os.tag == .windows)
        try std.process.Args.Iterator.initAllocator(.{ .vector = &arguments }, std.heap.page_allocator)
    else
        std.process.Args.Iterator.init(.{ .vector = &arguments });

    const resolved_args = try ArgParser.parse(&it);

    try std.testing.expectEqual(@as(?u8, 42), resolved_args.required);
    try std.testing.expectEqual(@as(?u8, null), resolved_args.optional);
}

test "parse missing required fails" {
    const Args = struct {
        required: u8,
    };
    const ArgParser = try Parser(Args);

    const arguments = [_][*:0]const u8{"prog"};
    var it = if (builtin.os.tag == .windows)
        try std.process.Args.Iterator.initAllocator(.{ .vector = &arguments }, std.heap.page_allocator)
    else
        std.process.Args.Iterator.init(.{ .vector = &arguments });

    const result = ArgParser.parse(&it);
    try std.testing.expectError(error.MissingRequiredArgument, result);
}
