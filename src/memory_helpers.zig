const std = @import("std");
const Allocator = std.mem.Allocator;

// Shorthand for making arena from existing allocator
pub const Dstack = struct {
    const Self = @This();

    // Two ways of initializing a struct with helpers here..
    //
    // 1) Declare variable as var on stack and init with that variable
    // 2) Do init without self and return anonymous struct
    //      This only works if your internals don't refer to any other
    //      values in the internals. These will become invalidated when you
    //      return from the init function!
    //
    // For example, self.arena.allocator() passes the arena itself into
    // the Allocator interface, which points right back to the arena right here.
    //
    // When init() returns and cleans up the local variables, arena and ator
    // will be copied to the variable at callsite. However, the reference address
    // ator has to the arena is now invalid (it was just copied).
    //
    // Lets do 2) for now
    //  -- Have allocator() be called afterwards
    //  -- ArenaAllocator is safe for copy-by-value
    //

    arena: std.heap.ArenaAllocator,

    // Version 1)
    // ator: Allocator,
    // pub fn init(self: *Self, allocator: Allocator) void {
    //     self.arena = std.heap.ArenaAllocator.init(allocator);
    //     self.ator = self.arena.allocator();
    // }

    pub fn init(allocator: Allocator) Dstack {
        return .{
            .arena = std.heap.ArenaAllocator.init(allocator),
        };
    }

    pub fn ator(self: *Self) Allocator {
        return self.arena.allocator();
    }

    pub fn deinit(self: *Self) void {
        self.arena.deinit();
    }
};

pub fn byteSlice(comptime T: type, slice: []T) []u8 {
    const bytes: [*]u8 = @ptrCast(&slice);
    return bytes[0 .. slice.len * @sizeOf(T)];
}

pub fn byteSliceC(comptime T: type, slice: []const T) []const u8 {
    var bytes: [*]const u8 = @ptrCast(&slice);
    return bytes[0 .. slice.len * @sizeOf(T)];
}
