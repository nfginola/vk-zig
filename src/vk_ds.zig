// GPU data-structures

const std = @import("std");
const Allocator = std.mem.Allocator;
const vkt = @import("vk_types.zig");
const vk = @import("vulkan");
const nvk = @import("vtx.zig");

/// CPU-visible GPU memory stack allocator for per-frame dynamic data
/// Optional round-robin functionality
pub const Stack = struct {
    const Self = @This();

    pub const InitOptions = struct {
        rr_blocks: u32 = 1,
        rr_block_size: u32 = 1_024 * 4,
    };

    opts: InitOptions,
    buf: vkt.Buffer,
    mem: vk.DeviceMemory,
    varena: *nvk.Arena,
    stack: std.heap.FixedBufferAllocator, // Holds the persistently mapped memory

    rr_byte_offset: u32 = 0,

    pub fn init(varena: *nvk.Arena, ctx: *nvk, opts: InitOptions) !Self {
        const total_size = opts.rr_block_size * opts.rr_blocks;

        const mem = try ctx.allocateMemory(varena, .cpu_to_gpu, total_size);
        const buf = try ctx.createBuffer(varena, total_size, .{ .uniform_buffer_bit = true });
        try ctx.dev.bindBufferMemory(buf.hdl, mem, 0);
        var memory: []u8 = undefined;
        if (try ctx.dev.mapMemory(mem, 0, total_size, .{})) |p| {
            memory = @as([*]u8, @ptrCast(p))[0..total_size];
        }
        const stack = std.heap.FixedBufferAllocator.init(memory);

        return .{
            .opts = opts,
            .buf = buf,
            .mem = mem,
            .varena = varena,
            .stack = stack,
        };
    }
    pub fn deinit(self: *Self) void {
        self.varena.deinit();
    }

    /// Switch to next independent stack
    pub fn next_block(self: *Self) void {
        self.rr_byte_offset += (self.opts.rr_block_size);
        self.rr_byte_offset %= (self.opts.rr_block_size * self.opts.rr_blocks); // wrap
    }

    pub fn grab(self: *Self, comptime T: type, alignment: u32) *T {
        self.stack.allocator().allocWithOptions(T, 1, alignment);
        return @ptrCast(try self.stack.allocator().alignedAlloc(T, alignment, 1));
    }

    pub fn reset(self: *Self) void {
        self.stack.reset();
    }
};
