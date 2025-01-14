const std = @import("std");
const nvk = @import("vtx.zig");
const vk = @import("vulkan");
const memh = @import("memory_helpers.zig");
const Allocator = std.mem.Allocator;

pub const Receipt = struct {
    start: usize,
    memory: ?[]u8 = null, // Context may optionally give memory slice for immediate filling
    size: u64 = 0,
};

pub const UploadContext = struct {
    const Self = @This();

    top: usize = 0,
    // memory: [1024]u8 = undefined,
    memory: [*]u8 = undefined,

    vtx: *nvk.Context = undefined,
    vk_mem: vk.DeviceMemory = undefined,
    vk_buf: vk.Buffer = undefined,
    vk_cmdp: nvk.CommandPool = undefined,
    vk_cmdb: nvk.CommandBuffer = undefined,
    queue: nvk.Queue = undefined,
    fence: vk.Fence = undefined,
    fence_on: bool = true,

    // Keep user payloads alive until submission complete
    dstack: memh.Dstack = undefined,

    pub fn push(self: *Self, data: []const u8, alignment: usize) !Receipt {
        try self.host_wait();

        const align_up = (self.top % alignment) != 0;
        const start = self.top + (alignment - (self.top % alignment)) * @intFromBool(align_up);
        @memcpy(self.memory[start..], data[0..data.len]);
        self.top = start + data.len;

        const rec = Receipt{ .start = start, .memory = null, .size = data.len };
        return rec;
    }

    pub fn grab(self: *Self, bytes: usize, alignment: usize) !Receipt {
        try self.host_wait();

        const align_up = (self.top % alignment) != 0;
        const start = self.top + (alignment - (self.top % alignment)) * @intFromBool(align_up);
        self.top = start + bytes;

        const rec = Receipt{ .start = start, .memory = self.memory[start..(start + bytes)], .size = bytes };
        return rec;
    }

    pub fn add_work(self: *Self, comptime T: type, payload: T, cb: fn (nvk.CommandBuffer, src: vk.Buffer, *const T) void) !void {
        const mem = try self.dstack.ator().create(T);
        mem.* = payload;
        cb(self.vk_cmdb, self.vk_buf, mem);
    }

    pub fn submit(self: *Self, sem_out: ?vk.Semaphore) !vk.Fence {
        try self.host_wait();
        self.top = 0;

        try self.vk_cmdb.endCommandBuffer();
        const ci = vk.SubmitInfo{
            .p_command_buffers = &.{self.vk_cmdb.handle},
            .command_buffer_count = 1,
            .signal_semaphore_count = if (sem_out != null) 1 else 0,
            .p_signal_semaphores = if (sem_out != null) &.{sem_out.?} else null,
        };
        _ = try self.queue.api.submit(1, &.{ci}, self.fence);
        self.fence_on = true;

        return self.fence;
    }

    pub fn host_wait(self: *Self) !void {
        if (self.fence_on) {
            _ = try self.vtx.dev.waitForFences(1, &.{self.fence}, vk.TRUE, std.math.maxInt(u64));
            _ = try self.vtx.dev.resetFences(1, &.{self.fence});
            self.fence_on = false; // do once

            _ = self.dstack.arena.reset(.retain_capacity);
            try self.vk_cmdp.reset(.{});
            self.vk_cmdb = try self.vk_cmdp.alloc(.primary, 1);
            try self.vk_cmdb.beginCommandBuffer(&vk.CommandBufferBeginInfo{ .flags = .{ .one_time_submit_bit = true } });
        }
    }

    pub fn create(ator: Allocator, vtx: *nvk.Context) !*Self {
        const self = try ator.create(Self);
        self.* = .{};
        self.vtx = vtx;
        self.queue = self.vtx.getQueue(.transfer);
        self.fence = self.vtx.createFence(.{ .signaled_bit = true });

        const total_size = 64_000;

        self.vk_cmdp = try vtx.createCmdPool(.transfer, .{ .transient_bit = true });
        self.vk_mem = try vtx.allocateMemory(.cpu_to_gpu, total_size);
        self.vk_buf = try vtx.createBuffer(total_size, .{ .transfer_src_bit = true });
        try vtx.dev.bindBufferMemory(self.vk_buf, self.vk_mem, 0);
        if (try vtx.dev.mapMemory(self.vk_mem, 0, total_size, .{})) |p| {
            self.memory = @as([*]u8, @ptrCast(p))[0..total_size];
        }

        self.dstack = memh.Dstack.init(ator);

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.vtx.destroyBuffer(self.vk_buf);
        self.vtx.freeMemory(self.vk_mem);
        self.vtx.destroyCmdPool(self.vk_cmdp);
        self.vtx.destroyFence(self.fence);
    }
};
