const std = @import("std");
const Allocator = std.mem.Allocator;

// Shorthand for making arena from existing allocator
pub const Arena = struct {
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

    pub fn init(allocator: Allocator) Self {
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

/// Cast arbitrary slice into byte slice
pub fn byteSlice(comptime T: type, slice: []T) []u8 {
    const bytes: [*]u8 = @ptrCast(@alignCast(slice));
    return bytes[0 .. slice.len * @sizeOf(T)];
}

/// Cast arbitrary slice into byte slice
pub fn byteSliceC(comptime T: type, slice: []const T) []const u8 {
    var bytes: [*]const u8 = @ptrCast(@alignCast(slice));
    return bytes[0 .. slice.len * @sizeOf(T)];
}

// ================= Callback Arena Start =================
/// General purpose callback helper designed for an API that uses
/// a create/destroy pair for some resource.
///
/// This specializes on API type T1, and a resource T2 to be operated on
/// by a concrete API of type T1 at deinit() time.
///
/// Original use case (create/destroy pair for arena-style clean up)
///
pub fn CallbackArena(comptime T1: type) type {
    const CleanupEntry = struct {
        cleanup_fn: *const fn (self: *T1, resource: *anyopaque) void,
        resource: *anyopaque,
    };

    return struct {
        const Self = @This();

        ctx: *T1,
        arena: Arena,
        entries: std.ArrayList(CleanupEntry),

        pub fn create(ator: Allocator, ctx: *T1) !*Self {
            var self = try ator.create(Self);
            self.arena = Arena.init(ator);
            self.entries = std.ArrayList(CleanupEntry).init(self.arena.ator());
            self.ctx = ctx;
            return self;
        }

        pub fn add(self: *Self, comptime T2: type, dest_fn: fn (*T1, T2) void, payload: T2) !void {
            // Wraps dest_fn in an type-erased callback for storage and arranges the logic
            // to cast back to original type
            const opaque_cb = struct {
                pub fn cb(ctx: *T1, resource: *anyopaque) void {
                    const res: *T2 = @ptrCast(@alignCast(resource));
                    dest_fn(ctx, res.*);
                }
            }.cb;

            try self.entries.append(CleanupEntry{
                .cleanup_fn = opaque_cb,
                .resource = try self.alloc(T2, payload),
            });
        }

        fn alloc(self: *Self, comptime T: type, data: T) !*anyopaque {
            const ptr = try self.arena.ator().create(T);
            ptr.* = data;
            return ptr;
        }

        pub fn deinit(self: *Self) void {
            // call each cleanup function in reverse order of allocation
            while (self.entries.items.len > 0) {
                const entry = self.entries.pop();
                entry.cleanup_fn(self.ctx, entry.resource);
            }
            self.entries.deinit();
            self.arena.deinit();
        }
    };
}
// ================= Callback Arena End =================
