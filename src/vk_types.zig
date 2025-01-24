const vk = @import("vulkan");
const vkb = @import("vk_base.zig");
const vsc = @import("vk_sc.zig");

// Native VK handles for convenience functions
pub const Device = vkb.Device;
pub const CommandBuffer = vkb.CommandBuffer;

pub const Queue = struct { api: vkb.Queue = undefined, fam: ?u32 = null, id: u32 = 0 };
pub const Buffer = struct {
    hdl: vk.Buffer,
    size: u64,

    memory: ?DeviceMemory = null,
    gpu_adr: ?u64 = 0, // Used only for buffer device address
    mem_offset: ?u64 = 0,
};
pub const DeviceMemory = struct {
    hdl: vk.DeviceMemory,
    dev_addressable: bool,
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

    pub fn writeBuffer(self: *Self, write: BufferWrite) void {
        const binfo = vk.DescriptorBufferInfo{
            .buffer = write.buf.hdl,
            .offset = write.buf_offset,
            .range = write.buf_range,
        };

        self.dev.updateDescriptorSets(1, &.{vk.WriteDescriptorSet{
            .dst_set = self.hdl,
            .dst_binding = 0,
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
    pub fn writeImage() void {}
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

    pub fn getNext(self: *Self, img_acq_sem: vk.Semaphore, img_acq_fence: ?vk.Fence) !vsc.Swapchain.Next {
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
