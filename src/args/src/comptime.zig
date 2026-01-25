const std = @import("std");
const args = @import("args.zig");
const parseValue = args.parseValue;
const ArgumentParseError = args.ArgumentParseError;

// A comptime argument parser that builds up arguments and can parse them at runtime.
// All configuration must happen at comptime; parsing happens at runtime.
// The max_args parameter sets the maximum number of arguments this parser can hold.
pub fn ComptimeArgParser(comptime max_args: usize) type {
    return struct {
        const Self = @This();

        arguments: [max_args]Argument,
        len: usize,

        pub const empty: Self = .{
            .arguments = undefined,
            .len = 0,
        };

        // Add an argument to the parser. Must be called at comptime.
        pub fn addArgument(comptime self: *Self, comptime arg: Argument) !void {
            // if (std.mem.containsAtLeastScalar(u8, arg.name, 1, '-') or std.mem.containsAtLeast(u8, arg.name, 1, &std.ascii.whitespace)) {
            //     return error.InvalidArgumentName;
            // }

            self.arguments[self.len] = arg;
            self.len += 1;
        }

        // Get the result type - a struct with a field for each argument.
        pub fn ResultType(comptime self: Self) type {
            return ResolvedArgs(self.arguments[0..self.len]);
        }

        // Parse command line arguments at runtime.
        // Returns a struct with fields corresponding to the configured arguments.
        pub fn parse(comptime self: Self, arg_iter: *std.process.Args.Iterator) !self.ResultType() {
            return self.parseFromIterator(arg_iter);
        }

        // Parse from a specific iterator (useful for testing).
        pub fn parseFromIterator(comptime self: Self, arg_iter: anytype) ArgumentParseError!self.ResultType() {
            const Result = self.ResultType();
            var result: Result = undefined;

            // Track which arguments have been set (runtime array)
            var args_set = [_]bool{false} ** self.len;

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

                // Inline for to unroll at comptime and avoid runtime pointer access
                inline for (self.arguments[0..self.len], 0..) |parser_arg, i| {
                    if (std.mem.eql(u8, arg_name, parser_arg.name)) {
                        found = true;

                        if (parser_arg.flag) {
                            // Flags are booleans that don't take a value
                            @field(result, parser_arg.name) = true;
                        } else {
                            // Parse value associated with this argument
                            const value_raw = arg_iter.next() orelse {
                                std.log.err("Expected value for argument {s}, but found none\n", .{parser_arg.name});
                                return error.ExpectedValue;
                            };
                            @field(result, parser_arg.name) = try parseValue(parser_arg.field_type, value_raw, parser_arg.name);
                        }
                        args_set[i] = true;
                    }
                }

                if (!found) {
                    std.log.err("Unknown argument {s}\n", .{arg[2..]});
                    return error.UnknownArgument;
                }
            }

            // Check that all required (non-optional) arguments were provided
            inline for (self.arguments[0..self.len], 0..) |arg, i| {
                if (!args_set[i]) {
                    const type_info = @typeInfo(arg.field_type);
                    if (type_info == .optional) {
                        // Set optional fields to null
                        @field(result, arg.name) = null;
                    } else if (arg.flag) {
                        // Flags default to false if not provided
                        @field(result, arg.name) = false;
                    } else {
                        return error.MissingRequiredArgument;
                    }
                }
            }

            return result;
        }
    };
}

pub const Argument = struct {
    name: []const u8,
    field_type: type,
    optional: bool = false,
    short: bool = false,
    long: bool = true,
    flag: bool = false,
};

// Creates a struct with a field for every argument.
pub fn ResolvedArgs(comptime arguments: []const Argument) type {
    const n = arguments.len;

    var field_names: [n][]const u8 = undefined;
    var field_types: [n]type = undefined;
    var field_attrs: [n]std.builtin.Type.StructField.Attributes = @splat(.{});

    var i: usize = 0;
    while (i < n) {
        const a = arguments[i];
        field_types[i] = a.field_type;
        field_names[i] = a.name;
        // &name_sanitized;

        i += 1;
    }

    return @Struct(
        .auto,
        null,
        &field_names,
        &field_types,
        &field_attrs,
    );
}

test "test resolving args" {
    comptime var arguments: [4]Argument = undefined;
    const arg1 = Argument{ .field_type = u8, .name = "someInt" };
    const arg2 = Argument{ .field_type = []const u8, .name = "someString" };
    const arg3 = Argument{ .field_type = bool, .name = "?questionMark" };
    const arg4 = Argument{ .field_type = ?u8, .name = "noQuestionMark" };

    args[0] = arg1;
    args[1] = arg2;
    args[2] = arg3;
    args[3] = arg4;

    const ArgsDict = ResolvedArgs(&arguments);

    const t_info: std.builtin.Type = @typeInfo(ArgsDict);

    try std.testing.expectEqual(.@"struct", std.meta.activeTag(t_info));
    try std.testing.expectEqual(arg1.field_type, t_info.@"struct".fields[0].type);
    try std.testing.expectEqual(arg1.name, t_info.@"struct".fields[0].name);
    try std.testing.expectEqual(arg2.field_type, t_info.@"struct".fields[1].type);
    try std.testing.expectEqual(arg2.name, t_info.@"struct".fields[1].name);
    try std.testing.expectEqual(arg3.field_type, t_info.@"struct".fields[2].type);
    try std.testing.expectEqual(arg3.name, t_info.@"struct".fields[2].name);
    try std.testing.expectEqual(arg4.field_type, t_info.@"struct".fields[3].type);
    try std.testing.expectEqual(arg4.name, t_info.@"struct".fields[3].name);
}

test "parse" {
    const parser = comptime blk: {
        var p = ComptimeArgParser(3).empty;
        try p.addArgument(.{ .name = "hi-mom", .field_type = u8 });
        try p.addArgument(.{ .name = "i-am-on-tv", .field_type = ?bool, .flag = true });
        try p.addArgument(.{ .name = "this is optional", .field_type = ?u64 });
        break :blk p;
    };

    const args_with_values = [_][*:0]const u8{ "prog", "--hi-mom", "25", "--i-am-on-tv" };
    var it = std.process.Args.Iterator.init(.{ .vector = &args_with_values });

    const resolved_args = try parser.parseFromIterator(&it);

    try std.testing.expectEqual(@as(?u8, 25), resolved_args.@"hi-mom");
    try std.testing.expectEqual(@as(?bool, true), resolved_args.@"i-am-on-tv");
}

test "parse missing optional" {
    const parser = comptime blk: {
        var p = ComptimeArgParser(2).empty;
        try p.addArgument(.{ .field_type = ?u8, .name = "required" });
        try p.addArgument(.{ .field_type = ?u8, .name = "optional" });
        break :blk p;
    };

    const arguments = [_][*:0]const u8{ "prog", "--required", "42" };
    var it = std.process.Args.Iterator.init(.{ .vector = &arguments });

    const resolved_args = try parser.parseFromIterator(&it);

    try std.testing.expectEqual(@as(?u8, 42), resolved_args.required);
    try std.testing.expectEqual(@as(?u8, null), resolved_args.optional);
}

test "parse missing required fails" {
    const parser = comptime blk: {
        var p = ComptimeArgParser(1).empty;
        try p.addArgument(.{ .field_type = u8, .name = "required" });
        break :blk p;
    };

    const arguments = [_][*:0]const u8{"prog"};
    var it = std.process.Args.Iterator.init(.{ .vector = &arguments });

    const result = parser.parseFromIterator(&it);
    try std.testing.expectError(error.MissingRequiredArgument, result);
}
