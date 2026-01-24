const std = @import("std");
const argument = @import("argument.zig");
const Argument = argument.Argument;

// A comptime argument parser that builds up arguments and can parse them at runtime.
// All configuration must happen at comptime; parsing happens at runtime.
// The max_args parameter sets the maximum number of arguments this parser can hold.
pub fn ArgParser(comptime max_args: usize) type {
    return struct {
        const Self = @This();

        arguments: [max_args]Argument,
        len: usize,

        pub const empty: Self = .{
            .arguments = undefined,
            .len = 0,
        };

        // Add an argument to the parser. Must be called at comptime.
        pub fn addArgument(comptime self: *Self, comptime opts: argument.ArgOptions) void {
            self.arguments[self.len] = Argument.initOpts(opts);
            self.len += 1;
        }

        // Get the result type - a struct with a field for each argument.
        pub fn ResultType(comptime self: Self) type {
            return argument.ResolvedArgs(self.arguments[0..self.len]);
        }

        // Parse command line arguments at runtime.
        // Returns a struct with fields corresponding to the configured arguments.
        pub fn parse(comptime self: Self) !self.ResultType() {
            var arg_iter = std.process.args();
            return self.parseFromIterator(&arg_iter);
        }

        // Parse from a specific iterator (useful for testing).
        pub fn parseFromIterator(comptime self: Self, arg_iter: anytype) !self.ResultType() {
            const Result = self.ResultType();
            var result: Result = undefined;

            // Track which arguments have been set (runtime array)
            var args_set = [_]bool{false} ** self.len;

            // Skip process name
            _ = arg_iter.skip();

            while (arg_iter.next()) |a| {
                // Arguments need a leading dash
                if (a.len == 0 or a[0] != '-') return error.ExpectedArgument;

                const arg_name = if (a.len > 1 and a[1] == '-') a[2..] else a[1..];

                // Use inline for to unroll at comptime - this avoids runtime pointer access
                var found = false;
                inline for (self.arguments[0..self.len], 0..) |arg, i| {
                    if (std.mem.eql(u8, arg_name, arg.name)) {
                        found = true;

                        if (arg.flag) {
                            // Flags are booleans that don't take a value
                            @field(result, arg.name) = true;
                        } else {
                            // Parse value associated with this argument
                            const value_raw = arg_iter.next() orelse return error.ExpectedValue;
                            @field(result, arg.name) = parseValue(arg.field_type, value_raw) catch return error.InvalidValue;
                        }
                        args_set[i] = true;
                    }
                }

                if (!found) return error.UnknownArgument;
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

        fn parseValue(comptime T: type, value_raw: []const u8) !T {
            const info = @typeInfo(T);

            return switch (info) {
                .int => std.fmt.parseInt(T, value_raw, 10) catch return error.InvalidValue,
                .optional => |opt| @as(T, try parseValue(opt.child, value_raw)),
                .pointer => |p| {
                    // []const u8
                    if (p.size == .slice and p.child == u8 and p.is_const) {
                        return value_raw;
                    } else return error.Unimplemented;
                },
                .bool => {
                    if (std.mem.eql(u8, value_raw, "true")) return true;
                    if (std.mem.eql(u8, value_raw, "false")) return false;
                    return error.InvalidValue;
                },
                else => error.Unimplemented,
            };
        }
    };
}

test "parse" {
    const parser = comptime blk: {
        var p = ArgParser(2).empty;
        p.addArgument(.{ .field_type = ?u8, .name = "hiMom" });
        p.addArgument(.{ .field_type = ?bool, .name = "amIOnTv" });
        break :blk p;
    };

    const args_with_values = [_][*:0]const u8{ "prog", "-hiMom", "25", "-amIOnTv", "true" };
    var it = std.process.Args.Iterator.init(.{ .vector = &args_with_values });

    const resolved_args = try parser.parseFromIterator(&it);

    try std.testing.expectEqual(@as(?u8, 25), resolved_args.hiMom);
    try std.testing.expectEqual(@as(?bool, true), resolved_args.amIOnTv);
}

test "parse missing optional" {
    const parser = comptime blk: {
        var p = ArgParser(2).empty;
        p.addArgument(.{ .field_type = ?u8, .name = "required" });
        p.addArgument(.{ .field_type = ?u8, .name = "optional" });
        break :blk p;
    };

    const args = [_][*:0]const u8{ "prog", "-required", "42" };
    var it = std.process.Args.Iterator.init(.{ .vector = &args });

    const resolved_args = try parser.parseFromIterator(&it);

    try std.testing.expectEqual(@as(?u8, 42), resolved_args.required);
    try std.testing.expectEqual(@as(?u8, null), resolved_args.optional);
}

test "parse missing required fails" {
    const parser = comptime blk: {
        var p = ArgParser(1).empty;
        p.addArgument(.{ .field_type = u8, .name = "required" });
        break :blk p;
    };

    const args = [_][*:0]const u8{"prog"};
    var it = std.process.Args.Iterator.init(.{ .vector = &args });

    const result = parser.parseFromIterator(&it);
    try std.testing.expectError(error.MissingRequiredArgument, result);
}
