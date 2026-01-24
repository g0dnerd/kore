const std = @import("std");

pub const ArgOptions = struct {
    name: []const u8,
    field_type: type,
    optional: bool = false,
    short: bool = false,
    long: bool = true,
    flag: bool = false,
};

pub const Argument = struct {
    name: []const u8,
    field_type: type,
    optional: bool = false,
    short: bool = false,
    long: bool = true,
    flag: bool = false,

    pub fn initOpts(comptime opts: ArgOptions) Argument {
        return .{ .name = opts.name, .field_type = opts.field_type, .optional = opts.optional, .short = opts.short, .long = opts.long, .flag = opts.flag };
    }
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
    comptime var args: [4]Argument = undefined;
    const arg1 = Argument.initOpts(.{
        .field_type = u8,
        .name = "someInt",
    });
    const arg2 = Argument.initOpts(.{ .field_type = []const u8, .name = "someString" });
    const arg3 = Argument.initOpts(.{ .field_type = bool, .name = "?questionMark" });
    const arg4 = Argument.initOpts(.{ .field_type = ?u8, .name = "noQuestionMark" });

    args[0] = arg1;
    args[1] = arg2;
    args[2] = arg3;
    args[3] = arg4;

    const ArgsDict = ResolvedArgs(&args);

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
