const std = @import("std");
const nvk = @import("vtx.zig");
const vk = @import("vulkan");
const memh = @import("memory_helpers.zig");
const Allocator = std.mem.Allocator;
const vkt = @import("vk_types.zig");

pub const Receipt = struct {
    start: usize,
    memory: ?[]u8 = null, // Context may optionally give memory slice for immediate filling
    size: u64,
};

const Self = @This();

// We can replace managing our own stack
// with the FixedBufferAllocator.. but lets go manual for now
//
// - allocWithOptions for alignment
// - reset

top: usize = 0,
memory: [*]u8 = undefined,

// Keep user payloads alive until submission complete
arena: memh.Arena = undefined,
varena: *nvk.Arena = undefined,

vtx: *nvk = undefined,
vk_mem: vk.DeviceMemory = undefined,
vk_buf: vkt.Buffer = undefined,

fence_on: bool = true,

cmdp_transfer: vkt.CommandPool = undefined,
cmdb_transfer: vkt.CommandBuffer = undefined,
transfer_queue: vkt.Queue = undefined,
transfer_fence: vk.Fence = undefined,
transfer_sem: vk.Semaphore = undefined, // Sync between ownership release and acq submits

cmdp_target: vkt.CommandPool = undefined,
cmdb_target: vkt.CommandBuffer = undefined,
target_fence: vk.Fence = undefined,

curr_target_queue: ?vkt.QueueType = null,

copies: std.ArrayList(CopyKind) = undefined,

const BufferCopy = struct {
    rec: Receipt,
    dst: vkt.Buffer,
    dst_stage: vk.PipelineStageFlags,
};

const ImageCopy = struct {
    // TODO: support images
    a: u32,
};

const CopyKind = union(enum) {
    buffer: BufferCopy,
    image: ImageCopy,
};

pub fn push(self: *Self, data: []const u8, alignment: usize) !Receipt {
    try self.host_wait();

    var start: usize = self.top;
    if (alignment != 0) {
        const align_up = (self.top % alignment) != 0;
        start = self.top + (alignment - (self.top % alignment)) * @intFromBool(align_up);
    }

    @memcpy(self.memory[start..], data[0..data.len]);
    self.top = start + data.len;

    const rec = Receipt{ .start = start, .memory = null, .size = data.len };
    return rec;
}

pub fn grab(self: *Self, bytes: usize, alignment: usize) !Receipt {
    try self.host_wait();

    var start: usize = self.top;
    if (alignment != 0) {
        const align_up = (self.top % alignment) != 0;
        start = self.top + (alignment - (self.top % alignment)) * @intFromBool(align_up);
    }
    self.top = start + bytes;

    const rec = Receipt{ .start = start, .memory = self.memory[start..(start + bytes)], .size = bytes };
    return rec;
}

pub fn copy_to_buffer(self: *Self, dst: vkt.Buffer, dst_stage: vk.PipelineStageFlags, receipt: Receipt) !void {
    const copy = CopyKind{ .buffer = .{ .dst = dst, .dst_stage = dst_stage, .rec = receipt } };
    try self.copies.append(copy);
}

/// User can optionally pass a semaphore that will be
/// signaled once transfer and qf ownership transfer have
/// both completed
pub fn submit(self: *Self, target: vkt.QueueType, sem_out: ?vk.Semaphore) !void {
    try self.host_wait();

    // Note or ownership transfer:
    // If an application does not need the contents of a resource to remain valid
    // when transferring from one queue family to another, then the ownership transfer
    // should be skipped.

    // Update internal state if user requests queue target change,
    // Otherwise the last target is cached.
    const replace_queue = if (self.curr_target_queue == null or (self.curr_target_queue != null and target != self.curr_target_queue)) true else false;
    var target_queue: vkt.Queue = undefined;
    if (replace_queue) {
        // Use dedicated acquire queue for graphics/compute
        self.curr_target_queue = switch (target) {
            .graphics, .transfer_graphics => .transfer_graphics,
            .compute, .transfer_compute => .transfer_compute,
            else => target,
        };
        target_queue = self.vtx.getQueue(self.curr_target_queue.?);

        // Recreate command pool for new queue target
        if (self.curr_target_queue != null) self.vtx.destroyCmdPool(self.cmdp_target);
        self.cmdp_target = try self.vtx.createCmdPool(null, self.curr_target_queue.?, .{ .transient_bit = true });
        self.cmdb_target = try self.cmdp_target.alloc(.primary, 1);
    } else {}

    // Record copies on transfer queue and prep ownership release and acquisition barriers
    var buf_releases = try std.ArrayList(vk.BufferMemoryBarrier).initCapacity(self.arena.ator(), 10);
    var buf_acqs = try std.ArrayList(vk.BufferMemoryBarrier).initCapacity(self.arena.ator(), 10);
    for (self.copies.items) |copy| {
        switch (copy) {
            .buffer => |item| {
                self.cmdb_transfer.copyBuffer(self.vk_buf.hdl, item.dst.hdl, 1, &.{vk.BufferCopy{
                    .dst_offset = 0,
                    .src_offset = item.rec.start,
                    .size = item.rec.size,
                }});

                try buf_releases.append(vk.BufferMemoryBarrier{
                    .buffer = item.dst.hdl,
                    .src_queue_family_index = self.transfer_queue.fam.?,
                    .dst_queue_family_index = target_queue.fam.?,
                    .src_access_mask = .{ .transfer_write_bit = true },
                    .dst_access_mask = .{},
                    .offset = 0,
                    .size = item.rec.size,
                });
                try buf_acqs.append(vk.BufferMemoryBarrier{
                    .buffer = item.dst.hdl,
                    .src_queue_family_index = self.transfer_queue.fam.?,
                    .dst_queue_family_index = target_queue.fam.?,
                    .src_access_mask = .{ .transfer_write_bit = true },
                    .dst_access_mask = .{},
                    .offset = 0,
                    .size = item.rec.size,
                });
            },
            .image => |item| {
                // TODO
                _ = item;
            },
        }
    }

    // Transfer and qf ownership release
    {
        self.cmdb_transfer.pipelineBarrier(
            .{ .transfer_bit = true },
            .{ .top_of_pipe_bit = true },
            .{},
            0,
            null,
            @intCast(buf_releases.items.len),
            @ptrCast(buf_releases.items.ptr),
            0,
            null,
        );

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
        try self.cmdb_target.beginCommandBuffer(&vk.CommandBufferBeginInfo{ .flags = .{ .one_time_submit_bit = true } });
        self.cmdb_target.pipelineBarrier(.{ .transfer_bit = true }, .{ .top_of_pipe_bit = true }, .{}, 0, null, @intCast(buf_acqs.items.len), @ptrCast(buf_acqs.items.ptr), 0, null);
        try self.cmdb_target.endCommandBuffer();
        const ci = vk.SubmitInfo{
            .p_command_buffers = &.{self.cmdb_target.handle},
            .command_buffer_count = 1,
            .signal_semaphore_count = if (sem_out != null) 1 else 0,
            .p_signal_semaphores = if (sem_out != null) &.{sem_out.?} else null,
            .wait_semaphore_count = 1,
            .p_wait_semaphores = &.{self.transfer_sem},
            .p_wait_dst_stage_mask = &.{.{ .transfer_bit = true }},
        };
        _ = try target_queue.api.submit(1, &.{ci}, self.target_fence);
    }

    // CPU payload submitted, can release
    self.top = 0;
    self.copies.clearAndFree();
    _ = self.arena.arena.reset(.retain_capacity);

    self.fence_on = true;
}

pub fn host_wait(self: *Self) !void {
    if (self.fence_on) {
        try self.vtx.waitResetFences(&[_]vk.Fence{ self.transfer_fence, self.target_fence });
        self.fence_on = false; // do once

        try self.cmdp_transfer.reset(.{});
        self.cmdb_transfer = try self.cmdp_transfer.alloc(.primary, 1);
        try self.cmdb_transfer.beginCommandBuffer(&vk.CommandBufferBeginInfo{ .flags = .{ .one_time_submit_bit = true } });
    }
}

pub fn create(ator: Allocator, vtx: *nvk) !*Self {
    const total_size = 64_000;

    const self = try ator.create(Self);
    self.* = .{};
    self.arena = memh.Arena.init(ator);
    self.varena = try vtx.createArena(ator);
    self.vtx = vtx;

    self.copies = try std.ArrayList(CopyKind).initCapacity(self.arena.ator(), 100);

    self.transfer_queue = self.vtx.getQueue(.transfer);
    self.transfer_fence = try self.vtx.createFence(self.varena, .{ .signaled_bit = true });
    self.transfer_sem = try self.vtx.createSemaphore(self.varena);
    self.cmdp_transfer = try vtx.createCmdPool(self.varena, .transfer, .{ .transient_bit = true });

    self.target_fence = try self.vtx.createFence(self.varena, .{ .signaled_bit = true });

    self.vk_mem = try vtx.allocateMemory(self.varena, .cpu_to_gpu, total_size);
    self.vk_buf = try vtx.createBuffer(self.varena, total_size, .{ .transfer_src_bit = true });
    try vtx.dev.bindBufferMemory(self.vk_buf.hdl, self.vk_mem, 0);
    if (try vtx.dev.mapMemory(self.vk_mem, 0, total_size, .{})) |p| {
        self.memory = @as([*]u8, @ptrCast(p))[0..total_size];
    }

    return self;
}

pub fn deinit(self: *Self) void {
    if (self.curr_target_queue != null) self.vtx.destroyCmdPool(self.cmdp_target);
    self.varena.deinit();
}
