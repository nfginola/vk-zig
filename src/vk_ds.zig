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
        rr_blocks: u32 = 2,
        rr_block_size: u32 = 1_024 * 4,
    };

    opts: InitOptions,
    buf: vkt.Buffer,
    mem: vk.DeviceMemory,
    varena: *nvk.Arena,

    memory: []u8,

    rr_byte_offset: u32 = 0,
    top: u32 = 0, // block stack top

    pub fn init(varena: *nvk.Arena, ctx: *nvk, opts: InitOptions) !Self {
        const total_size = opts.rr_block_size * opts.rr_blocks;

        const mem = try ctx.allocateMemory(varena, .{
            .type = .cpu_to_gpu,
            .size = total_size,
        });
        const buf = try ctx.createBuffer(varena, total_size, .{ .uniform_buffer_bit = true });
        try ctx.dev.bindBufferMemory(buf.hdl, mem, 0);
        var memory: []u8 = undefined;
        if (try ctx.dev.mapMemory(mem, 0, total_size, .{})) |p| {
            memory = @as([*]u8, @ptrCast(p))[0..total_size];
        }

        return .{
            .opts = opts,
            .buf = buf,
            .mem = mem,
            .varena = varena,
            .memory = memory,
        };
    }
    pub fn deinit(self: *Self) void {
        self.varena.deinit();
    }

    pub fn getOffset(self: *Self, block: u32) u32 {
        const offset = self.opts.rr_block_size * block;
        std.debug.assert(block <= self.opts.rr_blocks);
        return offset;
    }
    pub fn getBlockSize(self: *Self) u32 {
        return self.opts.rr_block_size;
    }

    /// Switch to next independent stack
    pub fn next_block(self: *Self) void {
        self.rr_byte_offset += (self.opts.rr_block_size);
        self.rr_byte_offset %= (self.opts.rr_block_size * self.opts.rr_blocks); // wrap
        self.reset();
    }

    pub fn grab(self: *Self, comptime T: type, alignment: u32) !*T {
        var start: u32 = self.top;
        if (alignment != 0) {
            const align_up = (self.top % alignment) != 0;
            start = self.top + (alignment - (self.top % alignment)) * @intFromBool(align_up);
        }
        self.top = start + @sizeOf(T);

        return @ptrCast(@alignCast(self.memory[(self.rr_byte_offset + start)..(self.rr_byte_offset + @sizeOf(T))]));
    }

    pub fn reset(self: *Self) void {
        self.top = 0;
    }
};
