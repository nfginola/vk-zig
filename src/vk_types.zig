const vk = @import("vulkan");
const vkb = @import("vk_base.zig");
const vsc = @import("vk_sc.zig");

// Native VK handles for convenience functions
pub const Device = vkb.Device;
pub const CommandBuffer = vkb.CommandBuffer;

pub const Queue = struct { api: vkb.Queue = undefined, fam: ?u32 = null, id: u32 = 0 };
pub const Buffer = struct { hdl: vk.Buffer };

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

/// TODO:
/// CPU-visible GPU memory stack allocator for per-frame dynamic data
pub const GPUStack = struct {
    buf: Buffer,
    mem: vk.DeviceMemory,

    // push
    // reset
};
