const std = @import("std");
const nvk = @import("vtx.zig");
const vk = @import("vulkan");
const memh = @import("memory_helpers.zig");
const Allocator = std.mem.Allocator;

pub const Receipt = struct {
    start: usize,
    memory: ?[]u8 = null, // Context may optionally give memory slice for immediate filling
    size: u64,
};

pub const UploadContext = struct {
    const Self = @This();

    // We can replace managing our own stack
    // with the FixedBufferAllocator.. but lets go manual for now
    //
    // - allocWithOptions for alignment
    // - reset
    top: usize = 0,
    memory: [*]u8 = undefined,

    varena: *nvk.VulkanArena = undefined,

    vtx: *nvk.Context = undefined,
    vk_mem: vk.DeviceMemory = undefined,
    vk_buf: nvk.Buffer = undefined,

    fence_on: bool = true,

    cmdp_transfer: nvk.CommandPool = undefined,
    cmdb_transfer: nvk.CommandBuffer = undefined,
    transfer_queue: nvk.Queue = undefined,
    transfer_fence: vk.Fence = undefined,
    transfer_sem: vk.Semaphore = undefined, // Sync between ownership release and acq submits

    cmdp_target: nvk.CommandPool = undefined,
    cmdb_target: nvk.CommandBuffer = undefined,
    target_queue: nvk.Queue = undefined,
    target_fence: vk.Fence = undefined,

    // Keep user payloads alive until submission complete
    arena: memh.Arena = undefined,

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

    pub fn copy_to_buffer(self: *Self, dst: nvk.Buffer, receipt: Receipt) !void {
        self.cmdb_transfer.copyBuffer(self.vk_buf.hdl, dst.hdl, 1, &.{vk.BufferCopy{ .dst_offset = 0, .src_offset = receipt.start, .size = receipt.size }});

        // Release ownership from this dedicated transfer queue family
        self.cmdb_transfer.pipelineBarrier(.{ .transfer_bit = true }, .{ .top_of_pipe_bit = true }, .{}, 0, null, 1, &.{vk.BufferMemoryBarrier{
            .buffer = dst.hdl,
            .src_queue_family_index = self.transfer_queue.fam.?,
            .dst_queue_family_index = self.target_queue.fam.?,
            .src_access_mask = .{ .transfer_write_bit = true },
            .dst_access_mask = .{},
            .offset = 0,
            .size = receipt.size,
        }}, 0, null);

        // Acquire ownership
        self.cmdb_target.pipelineBarrier(.{ .transfer_bit = true }, .{ .top_of_pipe_bit = true }, .{}, 0, null, 1, &.{vk.BufferMemoryBarrier{
            .buffer = dst.hdl,
            .src_queue_family_index = self.transfer_queue.fam.?,
            .dst_queue_family_index = self.target_queue.fam.?,
            .src_access_mask = .{ .transfer_write_bit = true },
            .dst_access_mask = .{},
            .offset = 0,
            .size = receipt.size,
        }}, 0, null);
    }

    pub fn submit(self: *Self, sem_out: ?vk.Semaphore) !vk.Fence {
        try self.host_wait();
        self.top = 0;

        // Transfer and qf ownership release
        {
            try self.cmdb_transfer.endCommandBuffer();
            const ci = vk.SubmitInfo{
                .p_command_buffers = &.{self.cmdb_transfer.handle},
                .command_buffer_count = 1,
                .signal_semaphore_count = 1,
                .p_signal_semaphores = &.{self.transfer_sem},
            };
            _ = try self.transfer_queue.api.submit(1, &.{ci}, self.transfer_fence);
        }

        // qf ownership acqusition for target family
        {
            try self.cmdb_target.endCommandBuffer();
            const ci = vk.SubmitInfo{
                .p_command_buffers = &.{self.cmdb_target.handle},
                .command_buffer_count = 1,
                .signal_semaphore_count = if (sem_out != null) 1 else 0,
                .p_signal_semaphores = if (sem_out != null) &.{sem_out.?} else null,
            };
            _ = try self.target_queue.api.submit(1, &.{ci}, self.target_fence);
        }

        self.fence_on = true;

        return self.target_fence;
    }

    pub fn host_wait(self: *Self) !void {
        if (self.fence_on) {
            _ = try self.vtx.dev.waitForFences(2, &.{ self.transfer_fence, self.target_fence }, vk.TRUE, std.math.maxInt(u64));
            _ = try self.vtx.dev.resetFences(2, &.{ self.transfer_fence, self.target_fence });
            self.fence_on = false; // do once

            _ = self.arena.arena.reset(.retain_capacity);
            try self.cmdp_transfer.reset(.{});
            try self.cmdp_target.reset(.{});
            self.cmdb_transfer = try self.cmdp_transfer.alloc(.primary, 1);
            self.cmdb_target = try self.cmdp_target.alloc(.primary, 1);
            try self.cmdb_transfer.beginCommandBuffer(&vk.CommandBufferBeginInfo{ .flags = .{ .one_time_submit_bit = true } });
            try self.cmdb_target.beginCommandBuffer(&vk.CommandBufferBeginInfo{ .flags = .{ .one_time_submit_bit = true } });
        }
    }

    pub fn create(ator: Allocator, vtx: *nvk.Context, target_queue: nvk.QueueType) !*Self {
        const total_size = 64_000;

        const self = try ator.create(Self);
        self.* = .{};
        self.arena = memh.Arena.init(ator);
        self.varena = try nvk.Context.createArena(ator);
        self.vtx = vtx;

        self.transfer_queue = self.vtx.getQueue(.transfer);
        self.transfer_fence = try self.vtx.createFence(self.varena, .{ .signaled_bit = true });
        self.transfer_sem = try self.vtx.createSemaphore(self.varena);
        self.cmdp_transfer = try vtx.createCmdPool(self.varena, .transfer, .{ .transient_bit = true });

        // Use dedicated acquire queue for graphics/compute
        const target_queue_type = switch (target_queue) {
            .graphics, .transfer_graphics => .transfer_graphics,
            .compute, .transfer_compute => .transfer_compute,
            else => target_queue,
        };
        self.target_queue = self.vtx.getQueue(target_queue_type);
        self.target_fence = try self.vtx.createFence(self.varena, .{ .signaled_bit = true });
        self.cmdp_target = try vtx.createCmdPool(self.varena, target_queue_type, .{ .transient_bit = true });

        self.vk_mem = try vtx.allocateMemory(self.varena, .cpu_to_gpu, total_size);
        self.vk_buf = try vtx.createBuffer(self.varena, total_size, .{ .transfer_src_bit = true });
        try vtx.dev.bindBufferMemory(self.vk_buf.hdl, self.vk_mem, 0);
        if (try vtx.dev.mapMemory(self.vk_mem, 0, total_size, .{})) |p| {
            self.memory = @as([*]u8, @ptrCast(p))[0..total_size];
        }

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.varena.deinit();
    }
};
