const vk = @import("vulkan");
const vkb = @import("vk_base.zig");
const vsc = @import("vk_sc.zig");

// Native VK handles for convenience functions
pub const Device = vkb.Device;
pub const CommandBuffer = vkb.CommandBuffer;

pub const SemaphoreSubmitInfo = struct {
    // If timeline semaphore, monotonically increasing counter automatically updated
    semaphore: *Semaphore,
    // I assume using all commands bit emulates when semaphores get waited/signaled
    // previously (without this option).
    // In other words, when queue waits, it waits at top of pipe.
    // When queue signals, it waits at bottom of pipe (all is done).
    stage_mask: vk.PipelineStageFlags2 = .{ .all_commands_bit = true },
};

pub const Queue = struct {
    const Self = @This();
    api: vkb.Queue = undefined,
    fam: ?u32 = null,
    id: u32 = 0,

    pub const SubmitInfo = struct {
        cmdbs: []const vk.CommandBuffer,
        waits: ?[]const *Semaphore = null,
        signals: ?[]const *Semaphore = null,
    };

    pub const SubmitInfo2 = struct {
        cmdbs: []const vk.CommandBuffer,
        waits: ?[]const SemaphoreSubmitInfo = null,
        signals: ?[]const SemaphoreSubmitInfo = null,
    };

    /// Submit without stage masks
    pub fn submit(self: *Self, inf: SubmitInfo) !void {
        var cmdbs: [8]vk.CommandBufferSubmitInfo = undefined;
        var waits: [8]vk.SemaphoreSubmitInfo = undefined;
        var signals: [8]vk.SemaphoreSubmitInfo = undefined;

        for (inf.cmdbs, 0..) |cmdb, i| {
            cmdbs[i] = vk.CommandBufferSubmitInfo{
                .command_buffer = cmdb,
                .device_mask = 0,
            };
        }

        if (inf.waits != null) {
            for (inf.waits.?, 0..) |wait, i| {
                waits[i] = vk.SemaphoreSubmitInfo{
                    .semaphore = wait.hdl,
                    .stage_mask = .{ .all_commands_bit = true },
                    .value = if (wait.timeline) wait.value else 0,
                    .device_index = 0,
                };
            }
        }

        if (inf.signals != null) {
            for (inf.signals.?, 0..) |signal, i| {
                signals[i] = vk.SemaphoreSubmitInfo{
                    .semaphore = signal.hdl,
                    .stage_mask = .{ .all_commands_bit = true },
                    .value = if (signal.timeline) signal.next() else 0,
                    .device_index = 0,
                };
            }
        }

        try self.api.submit2(1, &.{
            vk.SubmitInfo2{
                .command_buffer_info_count = @intCast(inf.cmdbs.len),
                .p_command_buffer_infos = &cmdbs,
                .wait_semaphore_info_count = if (inf.waits != null) @intCast(inf.waits.?.len) else 0,
                .p_wait_semaphore_infos = &waits,
                .signal_semaphore_info_count = if (inf.signals != null) @intCast(inf.signals.?.len) else 0,
                .p_signal_semaphore_infos = &signals,
            },
        }, .null_handle);
    }

    /// Submit with stage masks
    pub fn submit2(self: *Self, inf: SubmitInfo2) !void {
        var cmdbs: [8]vk.CommandBufferSubmitInfo = undefined;
        var waits: [8]vk.SemaphoreSubmitInfo = undefined;
        var signals: [8]vk.SemaphoreSubmitInfo = undefined;

        for (inf.cmdbs, 0..) |cmdb, i| {
            cmdbs[i] = vk.CommandBufferSubmitInfo{
                .command_buffer = cmdb,
                .device_mask = 0,
            };
        }

        if (inf.waits != null) {
            for (inf.waits.?, 0..) |wait, i| {
                waits[i] = vk.SemaphoreSubmitInfo{
                    .semaphore = wait.semaphore.hdl,
                    .stage_mask = wait.stage_mask,
                    .value = if (wait.semaphore.timeline) wait.semaphore.value else 0,
                    .device_index = 0,
                };
            }
        }

        if (inf.signals != null) {
            for (inf.signals.?, 0..) |signal, i| {
                signals[i] = vk.SemaphoreSubmitInfo{
                    .semaphore = signal.semaphore.hdl,
                    .stage_mask = signal.stage_mask,
                    .value = if (signal.semaphore.timeline) signal.semaphore.next() else 0,
                    .device_index = 0,
                };
            }
        }

        try self.api.submit2(1, &.{
            vk.SubmitInfo2{
                .command_buffer_info_count = @intCast(inf.cmdbs.len),
                .p_command_buffer_infos = &cmdbs,
                .wait_semaphore_info_count = if (inf.waits != null) @intCast(inf.waits.?.len) else 0,
                .p_wait_semaphore_infos = &waits,
                .signal_semaphore_info_count = if (inf.signals != null) @intCast(inf.signals.?.len) else 0,
                .p_signal_semaphore_infos = &signals,
            },
        }, .null_handle);
    }

    // pub fn submitMulti, if we would ever need..
};
pub const Buffer = struct {
    hdl: vk.Buffer,
    size: u64,

    memory: ?DeviceMemory = null,
    gpu_adr: ?u64 = 0, // Used only for buffer device address
    mem_offset: ?u64 = 0,
};
pub const Image = struct {
    hdl: vk.Image,
    memory: ?DeviceMemory = null,
    view: ?vk.ImageView = null, // default full-view if requested
};
pub const DeviceMemory = struct {
    hdl: vk.DeviceMemory,
    dev_addressable: bool,
};

pub const Semaphore = struct {
    const Self = @This();

    hdl: vk.Semaphore,
    value: u64 = 0,
    timeline: bool,

    pub fn next(self: *Self) u64 {
        self.value += 1;
        return self.value;
    }
};

pub const QueueType = enum {
    graphics,
    compute,
    transfer,

    // Dedicated queues used for queue ownership transfer to the
    // appropriate queue family
    transfer_graphics,
    transfer_compute,
};

pub const MemoryType = enum {
    gpu,
    cpu_to_gpu,
    gpu_to_cpu,
};

pub const DescriptorSet = struct {
    const Self = @This();

    dev: Device,
    hdl: vk.DescriptorSet,

    const BufferWrite = struct {
        buf: Buffer,
        buf_offset: u32,
        buf_range: u32,

        dst_binding: u32,
        dst_array_el: u32,
        dst_type: vk.DescriptorType,
    };

    const ImageWrite = struct {
        layout: vk.ImageLayout,
        view: vk.ImageView,

        dst_binding: u32,
        dst_array_el: u32,
        dst_type: vk.DescriptorType,
    };

    pub fn writeBuffer(self: *Self, write: BufferWrite) void {
        const binfo = vk.DescriptorBufferInfo{
            .buffer = write.buf.hdl,
            .offset = write.buf_offset,
            .range = write.buf_range,
        };

        self.dev.updateDescriptorSets(1, &.{vk.WriteDescriptorSet{
            .dst_set = self.hdl,
            .dst_binding = write.dst_binding,
            .dst_array_element = write.dst_array_el,
            .p_image_info = &.{vk.DescriptorImageInfo{
                .image_layout = .undefined,
                .image_view = .null_handle,
                .sampler = .null_handle,
            }},
            .p_texel_buffer_view = &.{.null_handle},
            .descriptor_count = 1,
            .descriptor_type = write.dst_type,
            .p_buffer_info = &.{binfo},
        }}, 0, null);
    }
    pub fn writeImage(self: *Self, write: ImageWrite) void {
        const iinfo = vk.DescriptorImageInfo{
            .image_layout = write.layout,
            .image_view = write.view,
            .sampler = .null_handle,
        };

        self.dev.updateDescriptorSets(1, &.{vk.WriteDescriptorSet{
            .dst_set = self.hdl,
            .dst_binding = write.dst_binding,
            .dst_array_element = write.dst_array_el,
            .descriptor_count = 1,
            .descriptor_type = write.dst_type,
            .p_image_info = &.{iinfo},
            .p_texel_buffer_view = &.{.null_handle},
            .p_buffer_info = &.{vk.DescriptorBufferInfo{
                .offset = 0,
                .buffer = .null_handle,
                .range = 0,
            }},
        }}, 0, null);
    }
};

pub const DescriptorPool = struct {
    const Self = @This();

    pub const AllocInfo = struct {
        layout: vk.DescriptorSetLayout,

        // Maximum descriptors allowed for this binding
        // For descriptor indexing
        variable_descriptors: ?u32,
    };

    dev: Device,
    hdl: vk.DescriptorPool,

    // Allocate single
    pub fn alloc(self: *Self, allocInfo: AllocInfo) !DescriptorSet {
        var dset: DescriptorSet = undefined;
        _ = try self.dev.allocateDescriptorSets(&vk.DescriptorSetAllocateInfo{
            .descriptor_pool = self.hdl,
            .descriptor_set_count = 1,
            .p_set_layouts = &.{allocInfo.layout},
            .p_next = if (allocInfo.variable_descriptors != null) &vk.DescriptorSetVariableDescriptorCountAllocateInfo{
                .descriptor_set_count = 1,
                .p_descriptor_counts = &.{allocInfo.variable_descriptors.?}, // will fail if dlayout has descriptor count < this one
            } else null,
        }, @ptrCast(&dset.hdl));

        dset.dev = self.dev;

        return dset;
    }

    pub fn reset(self: *Self) void {
        self.dev.resetDescriptorPool(self.hdl, .{});
    }
};

pub const CommandPool = struct {
    const Self = @This();

    devd: vkb.DeviceDispatch,
    dev: Device,
    hdl: vk.CommandPool,

    pub fn alloc(
        self: *Self,
        level: vk.CommandBufferLevel,
        count: u32,
    ) !CommandBuffer {
        var cmdb: vk.CommandBuffer = undefined;
        try self.dev.allocateCommandBuffers(&.{ .command_pool = self.hdl, .command_buffer_count = count, .level = level }, @ptrCast(&cmdb));
        return CommandBuffer.init(cmdb, &self.devd);
    }

    pub fn reset(self: *Self, flags: vk.CommandPoolResetFlags) !void {
        try self.dev.resetCommandPool(self.hdl, flags);
    }
};

// Thin translation layer here because we want Swapchain to be
// able to consume vkt types (Queue) at interface level.
// Otherwise circular dependency since vsc.Swapchain would
// need vkt (Queue), and vkt would include vsc.Swapchain as public interface
pub const Swapchain = struct {
    const Self = @This();
    native: vsc.Swapchain,

    pub fn getNext(self: *Self, img_acq_sem: vk.Semaphore, img_acq_fence: ?vk.Fence) !?vsc.Swapchain.Next {
        return try self.native.getNext(img_acq_sem, img_acq_fence);
    }

    pub fn present(self: *Self, present_queue: Queue, wait_sem: vk.Semaphore) !void {
        try self.native.present(present_queue.api, wait_sem);
    }

    pub fn getExtent(self: *Self) vk.Extent2D {
        return self.native.extent;
    }

    pub fn getFormat(self: *Self) vk.Format {
        return self.native.format.format;
    }
};

pub const DescriptorSetLayoutInfo = struct {
    bindings: []const DescriptorSetLayoutBinding,
    flags: vk.DescriptorSetLayoutCreateFlags,
};

pub const DescriptorSetLayoutBinding = struct {
    binding: vk.DescriptorSetLayoutBinding,

    // DDI (pNext)
    flags: ?vk.DescriptorBindingFlags,
};

pub const MemoryAllocateInfo = struct {
    type: MemoryType,
    size: u64,
    device_adr: bool = false,
};

// For shorthand buffer/memory pair creation
pub const MemoryBufferInfo = struct {
    size: u64,
    usage: vk.BufferUsageFlags,
    mem_type: MemoryType,
};

pub const ImageInfo = struct {
    type: vk.ImageType,
    format: vk.Format,
    width: u32,
    height: u32,
    usage: vk.ImageUsageFlags,
    depth: u32 = 1,
    mips: u32 = 1,
    array_layers: u32 = 1,
    samples: vk.SampleCountFlags = .{ .@"1_bit" = true },
    sharing_mode: vk.SharingMode = .exclusive,
    tiling: vk.ImageTiling = .optimal,
    view_type: ?vk.ImageViewType = null,
};

pub const Utils = struct {
    pub fn fullSubres(aspect: vk.ImageAspectFlags) vk.ImageSubresourceRange {
        return vk.ImageSubresourceRange{
            .aspect_mask = aspect,
            .base_array_layer = 0,
            .base_mip_level = 0,
            .level_count = vk.REMAINING_ARRAY_LAYERS,
            .layer_count = vk.REMAINING_MIP_LEVELS,
        };
    }
};
