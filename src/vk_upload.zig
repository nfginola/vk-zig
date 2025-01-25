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
memory: []u8 = undefined,

// Keep user payloads alive until submission complete
arena: memh.Arena = undefined,
varena: *nvk.Arena = undefined,

vtx: *nvk = undefined,
vk_buf: vkt.Buffer = undefined,

host_waitable: bool = true,

cmdp_transfer: vkt.CommandPool = undefined,
cmdb_transfer: vkt.CommandBuffer = undefined,
transfer_queue: vkt.Queue = undefined,
transfer_sem: vkt.Semaphore = undefined, // Sync between ownership release and acq submits

cmdp_target: vkt.CommandPool = undefined,
cmdb_target: vkt.CommandBuffer = undefined,
target_sem: vkt.Semaphore = undefined, // Sync between ownership release and acq submits

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

    @memcpy(self.memory[start..(start + data.len)], data[0..]);
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
pub fn submit(self: *Self, target: vkt.QueueType) !vkt.Semaphore {
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
        const timeline = vk.TimelineSemaphoreSubmitInfo{
            .signal_semaphore_value_count = 1,
            .p_signal_semaphore_values = &.{self.transfer_sem.next()},
        };
        const ci = vk.SubmitInfo{
            .p_command_buffers = &.{self.cmdb_transfer.handle},
            .command_buffer_count = 1,
            .signal_semaphore_count = 1,
            .p_signal_semaphores = &.{self.transfer_sem.hdl},
            .p_next = &timeline,
        };
        _ = try self.transfer_queue.api.submit(1, &.{ci}, .null_handle);
    }

    // qf ownership acqusition for target family
    {
        try self.cmdb_target.beginCommandBuffer(&vk.CommandBufferBeginInfo{ .flags = .{ .one_time_submit_bit = true } });
        self.cmdb_target.pipelineBarrier(
            .{ .transfer_bit = true },
            .{ .top_of_pipe_bit = true },
            .{},
            0,
            null,
            @intCast(buf_acqs.items.len),
            @ptrCast(buf_acqs.items.ptr),
            0,
            null,
        );
        try self.cmdb_target.endCommandBuffer();
        const timeline = vk.TimelineSemaphoreSubmitInfo{
            .wait_semaphore_value_count = 1,
            .p_wait_semaphore_values = &.{self.transfer_sem.value},
            .signal_semaphore_value_count = 1,
            .p_signal_semaphore_values = &.{self.target_sem.next()},
        };
        const ci = vk.SubmitInfo{
            .command_buffer_count = 1,
            .p_command_buffers = &.{self.cmdb_target.handle},
            .p_wait_dst_stage_mask = &.{.{ .transfer_bit = true }},
            .wait_semaphore_count = 1,
            .p_wait_semaphores = &.{self.transfer_sem.hdl},
            .signal_semaphore_count = 1,
            .p_signal_semaphores = &.{self.target_sem.hdl},
            .p_next = &timeline,
        };
        _ = try target_queue.api.submit(1, &.{ci}, .null_handle);
    }

    // CPU payload submitted, can release transient info
    self.top = 0;
    self.copies.clearAndFree();
    _ = self.arena.arena.reset(.retain_capacity);
    self.host_waitable = true;

    return self.target_sem;
}

pub fn host_wait(self: *Self) !void {
    if (self.host_waitable) {
        _ = try self.vtx.dev.waitSemaphores(
            &.{
                .semaphore_count = 2,
                .p_semaphores = &.{ self.transfer_sem.hdl, self.target_sem.hdl },
                .p_values = &.{ self.transfer_sem.value, self.target_sem.value },
            },
            std.math.maxInt(u64),
        );

        self.host_waitable = false; // do once

        try self.cmdp_transfer.reset(.{});
        self.cmdb_transfer = try self.cmdp_transfer.alloc(.primary, 1);
        try self.cmdb_transfer.beginCommandBuffer(&vk.CommandBufferBeginInfo{ .flags = .{ .one_time_submit_bit = true } });
    }
}

pub fn create(ator: Allocator, vtx: *nvk, total_size: u32) !*Self {
    const self = try ator.create(Self);
    self.* = .{};
    self.arena = memh.Arena.init(ator);
    self.varena = try vtx.createArena(ator);
    self.vtx = vtx;

    self.copies = try std.ArrayList(CopyKind).initCapacity(self.arena.ator(), 100);

    self.transfer_queue = self.vtx.getQueue(.transfer);
    self.transfer_sem = try self.vtx.createSemaphoreT(self.varena);
    self.cmdp_transfer = try vtx.createCmdPool(self.varena, .transfer, .{ .transient_bit = true });

    self.target_sem = try self.vtx.createSemaphoreT(self.varena);

    self.vk_buf = try vtx.createBufferWithMemory(self.varena, .{
        .size = total_size,
        .usage = .{ .transfer_src_bit = true },
        .mem_type = .cpu_to_gpu,
    });
    self.memory = try vtx.mapBuffer(u8, self.vk_buf);

    return self;
}

pub fn deinit(self: *Self) void {
    if (self.curr_target_queue != null) self.vtx.destroyCmdPool(self.cmdp_target);
    self.varena.deinit();
}
