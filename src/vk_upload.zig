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
target_sem: vkt.Semaphore = undefined, // Sync between ownership acq and (optionally) outside

curr_target_queue: ?vkt.QueueType = null,

copies: std.ArrayList(CopyKind) = undefined,

const BufferCopy = struct {
    rec: Receipt,
    dst: vkt.Buffer,
};

const ImageCopy = struct {
    // TODO: support images
    //
    rec: Receipt,
    img: vkt.Image,
    inf: ImageCopyInfo,
};

const CopyKind = union(enum) {
    buffer: BufferCopy,
    image: ImageCopy,
};

pub const ImageCopyInfo = struct {
    offset: vk.Offset3D = .{ .x = 0, .y = 0, .z = 0 },
    subres: vk.ImageSubresourceLayers = .{
        .aspect_mask = .{ .color_bit = true },
        .mip_level = 0, // Target mip
        .base_array_layer = 0, // Target start layer
        .layer_count = 1,
    },
    extent: vk.Extent3D,
    layout: vk.ImageLayout = .shader_read_only_optimal,
    gen_mips: bool = true,
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

pub fn copy_to_buffer(self: *Self, dst: vkt.Buffer, receipt: Receipt) !void {
    const copy = CopyKind{ .buffer = .{ .dst = dst, .rec = receipt } };
    try self.copies.append(copy);
}

pub fn copy_to_image(self: *Self, dst: vkt.Image, inf: ImageCopyInfo, receipt: Receipt) !void {
    const copy = CopyKind{ .image = .{ .img = dst, .inf = inf, .rec = receipt } };
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

    // Batch barriers (queue ownership release, acquisition and image layout prep)
    var buf_releases = try std.ArrayList(vk.BufferMemoryBarrier).initCapacity(self.arena.ator(), 10);
    var buf_acqs = try std.ArrayList(vk.BufferMemoryBarrier).initCapacity(self.arena.ator(), 10);
    var img_layout_prep = try std.ArrayList(vk.ImageMemoryBarrier).initCapacity(self.arena.ator(), 10);
    var img_releases = try std.ArrayList(vk.ImageMemoryBarrier).initCapacity(self.arena.ator(), 10);
    var img_acqs = try std.ArrayList(vk.ImageMemoryBarrier).initCapacity(self.arena.ator(), 10);
    for (self.copies.items) |copy| {
        switch (copy) {
            .buffer => |item| {
                // Validation requires us to give src/dst family on both the releaser and acquirer
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
                    .dst_access_mask = .{ .memory_read_bit = true },
                    .offset = 0,
                    .size = item.rec.size,
                });
            },
            .image => |item| {
                // Put images into right layout for optimal transfer
                try img_layout_prep.append(vk.ImageMemoryBarrier{
                    .src_queue_family_index = self.transfer_queue.fam.?,
                    .dst_queue_family_index = self.transfer_queue.fam.?,
                    .src_access_mask = .{},
                    .dst_access_mask = .{ .transfer_write_bit = true },
                    .image = item.img.hdl,
                    .old_layout = .undefined,
                    .new_layout = .transfer_dst_optimal,
                    .subresource_range = .{
                        .aspect_mask = item.inf.subres.aspect_mask,
                        .base_array_layer = item.inf.subres.base_array_layer, // Sync from target subresouce and up (remaining)
                        .base_mip_level = 0,
                        .layer_count = vk.REMAINING_ARRAY_LAYERS,
                        .level_count = vk.REMAINING_MIP_LEVELS,
                    },
                });

                try img_releases.append(vk.ImageMemoryBarrier{
                    .src_queue_family_index = self.transfer_queue.fam.?,
                    .dst_queue_family_index = target_queue.fam.?,
                    .src_access_mask = .{ .transfer_write_bit = true },
                    .dst_access_mask = .{ .transfer_write_bit = true },
                    .image = item.img.hdl,
                    .old_layout = .transfer_dst_optimal,
                    .new_layout = .general, // general purpose layout intermediary
                    .subresource_range = .{
                        .aspect_mask = item.inf.subres.aspect_mask,
                        .base_array_layer = item.inf.subres.base_array_layer,
                        .base_mip_level = 0,
                        .layer_count = vk.REMAINING_ARRAY_LAYERS,
                        .level_count = vk.REMAINING_MIP_LEVELS,
                    },
                });

                try img_acqs.append(vk.ImageMemoryBarrier{
                    .src_queue_family_index = self.transfer_queue.fam.?,
                    .dst_queue_family_index = target_queue.fam.?,
                    .src_access_mask = .{ .transfer_write_bit = true },
                    .dst_access_mask = .{ .memory_read_bit = true },
                    .image = item.img.hdl,

                    // validation layer bug?: for some reason validation still thinks its in transfer dst optimal
                    // must be that API just doesnt transition from tranfer dst optimal to general on img_release?
                    // it doesn't matter much since Graphics, Compute and Transfer all support transfer_dst_optimal,
                    // and for Sparse Queue everything goes into there as undefined anyways
                    // .old_layout = .general,
                    .old_layout = .transfer_dst_optimal,

                    .new_layout = item.inf.layout, // final desired layout
                    .subresource_range = .{
                        .aspect_mask = item.inf.subres.aspect_mask,
                        .base_array_layer = item.inf.subres.base_array_layer,
                        .base_mip_level = 0,
                        .layer_count = vk.REMAINING_ARRAY_LAYERS,
                        .level_count = vk.REMAINING_MIP_LEVELS,
                    },
                });
            },
        }
    }

    // Record image layout prep before copies
    self.cmdb_transfer.pipelineBarrier(
        .{ .bottom_of_pipe_bit = true },
        .{ .transfer_bit = true },
        .{},
        0,
        null,
        0,
        null,
        @intCast(img_layout_prep.items.len),
        @ptrCast(img_layout_prep.items.ptr),
    );

    // Record copies
    for (self.copies.items) |copy| {
        switch (copy) {
            .buffer => |item| {
                self.cmdb_transfer.copyBuffer(self.vk_buf.hdl, item.dst.hdl, 1, &.{vk.BufferCopy{
                    .dst_offset = 0,
                    .src_offset = item.rec.start,
                    .size = item.rec.size,
                }});
            },
            .image => |item| {

                // FUTURE: We can coalesce multiple copies into a single
                //         API call, but we need to group per final layout.
                //         Keep it simple with singles for now.
                self.cmdb_transfer.copyBufferToImage(
                    self.vk_buf.hdl,
                    item.img.hdl,
                    .transfer_dst_optimal,
                    1,
                    &.{vk.BufferImageCopy{
                        .buffer_offset = item.rec.start,
                        // Buffer row/image 0 --> Assume data tightly packed
                        // based on Extent
                        .buffer_row_length = 0,
                        .buffer_image_height = 0,
                        .image_offset = item.inf.offset,
                        .image_extent = item.inf.extent,
                        .image_subresource = item.inf.subres,
                    }},
                );

                // try vkt.Utils.generateMips(item.img, self.cmdb_transfer, .{
                //     .aspect = .{ .color_bit = true },
                //     .width = item.inf.extent.width,
                //     .height = item.inf.extent.height,
                //     .layout = .transfer_dst_optimal,
                // });
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
            @intCast(img_releases.items.len),
            @ptrCast(img_releases.items.ptr),
        );

        try self.cmdb_transfer.endCommandBuffer();

        try self.transfer_queue.submit(.{
            .cmdbs = &[_]vk.CommandBuffer{self.cmdb_transfer.handle},
            .signals = &[_]*vkt.Semaphore{&self.transfer_sem},
        });
    }

    // qf ownership acqusition for target family
    {
        try self.cmdb_target.beginCommandBuffer(&vk.CommandBufferBeginInfo{ .flags = .{ .one_time_submit_bit = true } });
        self.cmdb_target.pipelineBarrier(
            .{ .bottom_of_pipe_bit = true },
            .{ .top_of_pipe_bit = true },
            .{},
            0,
            null,
            @intCast(buf_acqs.items.len),
            @ptrCast(buf_acqs.items.ptr),
            @intCast(img_acqs.items.len),
            @ptrCast(img_acqs.items.ptr),
        );

        // Record generate mips
        for (self.copies.items) |copy| {
            switch (copy) {
                .buffer => {},
                .image => |item| {
                    if (item.inf.gen_mips) {
                        try vkt.Utils.generateMips(item.img, self.cmdb_target, .{
                            .aspect = .{ .color_bit = true },
                            .width = item.inf.extent.width,
                            .height = item.inf.extent.height,
                            .layout = item.inf.layout, // Acquisition put them in final layout
                        });
                    }
                },
            }
        }

        try self.cmdb_target.endCommandBuffer();

        try target_queue.submit(.{
            .cmdbs = &[_]vk.CommandBuffer{self.cmdb_target.handle},
            .waits = &[_]*vkt.Semaphore{&self.transfer_sem},
            .signals = &[_]*vkt.Semaphore{&self.target_sem},
        });
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

        self.host_waitable = false;

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
    self.transfer_sem = try self.vtx.createSemaphore(self.varena);
    self.cmdp_transfer = try vtx.createCmdPool(self.varena, .transfer, .{ .transient_bit = true });

    self.target_sem = try self.vtx.createSemaphore(self.varena);

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
