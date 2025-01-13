const bconf = @import("VK_CONF"); // Build configs
const std = @import("std");
const glfw = @import("mach-glfw");
const vk = @import("vulkan");
const Allocator = std.mem.Allocator;
const memh = @import("memory_helpers.zig");

pub const InitOptions = struct {
    name: [*:0]const u8 = "Default Name",
    window: glfw.Window,
};

// Each API info gets merged, so this is just a convenient way to group
// Versions are additive, so we need previous ones if we enable more.
const apis: []const vk.ApiInfo = &.{
    // vk.features.version_1_3,
    // vk.features.version_1_2,
    vk.features.version_1_1,
    vk.features.version_1_0,
    vk.extensions.khr_surface,
    vk.extensions.khr_swapchain,

    if (bconf.validation_layer)
        vk.extensions.ext_debug_utils
    else
        .{},
};
const api_version = apis[0];

// Dispatch tables
const BaseDispatch = vk.BaseWrapper(apis); // Non-instance functions
const InstanceDispatch = vk.InstanceWrapper(apis); // Instance functions
const DeviceDispatch = vk.DeviceWrapper(apis); // Device functions (and sub-structures like queue)
const QueueProxy = vk.QueueProxy(apis);

// Types
const Instance = vk.InstanceProxy(apis);
pub const Device = vk.DeviceProxy(apis);
pub const CommandBuffer = vk.CommandBufferProxy(apis);
pub const QueueType = enum { graphics, transfer, compute };
pub const Queue = struct { api: QueueProxy, fam: u32 };
pub const CommandPool = struct {
    const Self = @This();

    ctx: *Context,
    hdl: vk.CommandPool,

    pub fn alloc(
        self: *Self,
        level: vk.CommandBufferLevel,
        count: u32,
    ) !CommandBuffer {
        var cmdb: vk.CommandBuffer = undefined;
        try self.ctx.*.dev.allocateCommandBuffers(&.{ .command_pool = self.hdl, .command_buffer_count = count, .level = level }, @ptrCast(&cmdb));
        return CommandBuffer.init(cmdb, &self.ctx.*.devd);
    }

    pub fn reset(self: *Self, flags: vk.CommandPoolResetFlags) !void {
        try self.ctx.*.dev.resetCommandPool(self.hdl, flags);
    }
};
pub const Swapchain = struct {
    const Self = @This();
    pub const Next = struct {
        idx: u32,
        image: vk.Image,
    };

    surf: vk.SurfaceKHR,
    hdl: vk.SwapchainKHR,
    format: vk.SurfaceFormatKHR,
    images: std.ArrayList(vk.Image),
    views: std.ArrayList(vk.ImageView),
    dev: Device,

    next: Next,

    pub fn getNext(self: *Self, img_acq_sem: vk.Semaphore, img_acq_fence: vk.Fence) !Next {
        const res = try self.dev.acquireNextImageKHR(self.hdl, std.math.maxInt(u64), img_acq_sem, img_acq_fence);

        self.next = .{
            .image = self.images.items[res.image_index],
            .idx = res.image_index,
        };
        return self.next;
    }

    pub fn present(self: *Self, present_queue: Queue, wait_sem: vk.Semaphore) !void {
        _ = try present_queue.api.presentKHR(&.{
            .p_image_indices = &.{self.next.idx},
            .p_swapchains = &.{self.hdl},
            .swapchain_count = 1,
            .p_wait_semaphores = &.{wait_sem},
            .wait_semaphore_count = 1,
        });
    }

    fn init(self: *Self, ator: Allocator, pd: vk.PhysicalDevice, dev: Device, win: glfw.Window) !void {
        const inst = Context.inst;
        self.dev = dev;

        var dstack = memh.Dstack.init(ator);
        defer dstack.deinit();

        if (glfw.createWindowSurface(inst.handle, win, null, &self.surf) != @intFromEnum(vk.Result.success))
            return error.SurfaceInitFailed;

        self.format = blk: {
            const surf_fmts = try inst.getPhysicalDeviceSurfaceFormatsAllocKHR(pd, self.surf, dstack.ator());
            for (surf_fmts) |fmt| {
                if (fmt.format == .b8g8r8a8_srgb and fmt.color_space == .srgb_nonlinear_khr) {
                    break :blk fmt;
                }
            }
            return error.SurfFormatNotFound;
        };

        const sel_pmode: vk.PresentModeKHR = blk: {
            const present_modes = try inst.getPhysicalDeviceSurfacePresentModesAllocKHR(pd, self.surf, dstack.ator());
            for (present_modes) |pmode| {
                if (pmode == .mailbox_khr)
                    break :blk pmode;
            }

            // Fallback
            // FIFO: Show on next vertical blank (vsync) - guaranteed
            // Immediate: May cause tearing (No vsync)
            break :blk vk.PresentModeKHR.fifo_khr;
        };

        const sc_extent: vk.Extent2D = blk: {
            const surf_caps = try inst.getPhysicalDeviceSurfaceCapabilitiesKHR(pd, self.surf);
            // Spec: (0xFFFFFFFF, 0xFFFFFFFF) indicates that the surface size will be determined by the extent of a swapchain
            // We assign the window client dimensions.
            if (surf_caps.current_extent.width == std.math.maxInt(u32)) {
                const draw_area = win.getFramebufferSize();
                break :blk .{
                    .width = std.math.clamp(draw_area.width, surf_caps.current_extent.width, surf_caps.current_extent.width),
                    .height = std.math.clamp(draw_area.height, surf_caps.current_extent.height, surf_caps.current_extent.height),
                };
            }

            break :blk surf_caps.current_extent;
        };

        self.hdl = try dev.createSwapchainKHR(&vk.SwapchainCreateInfoKHR{
            .surface = self.surf,
            .image_format = self.format.format,
            .image_extent = sc_extent,
            .image_array_layers = 1,
            .image_usage = .{ .color_attachment_bit = true, .transfer_dst_bit = true }, // allow clear color with transfer dst
            .image_color_space = .srgb_nonlinear_khr,
            .min_image_count = 3,
            .present_mode = sel_pmode,
            // TODO: must transfer ownership if queue family is different
            // branch depending on qf
            .image_sharing_mode = .exclusive,
            .pre_transform = .{ .identity_bit_khr = true },
            .composite_alpha = .{ .opaque_bit_khr = true },
            .clipped = vk.TRUE,
        }, null);

        const images = try dev.getSwapchainImagesAllocKHR(self.hdl, dstack.ator());
        self.images = try std.ArrayList(vk.Image).initCapacity(ator, images.len);
        self.views = try std.ArrayList(vk.ImageView).initCapacity(ator, images.len);
        try self.images.appendSlice(images);

        for (self.images.items) |img| {
            const ci = vk.ImageViewCreateInfo{ .image = img, .view_type = .@"2d", .format = self.format.format, .components = .{
                .r = .identity,
                .g = .identity,
                .b = .identity,
                .a = .identity,
            }, .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_array_layer = 0,
                .base_mip_level = 0,
                .layer_count = 1,
                .level_count = 1,
            } };
            const v = try self.dev.createImageView(&ci, null);
            try self.views.append(v);
        }
    }

    fn deinit(self: *Self) void {
        for (self.views.items) |v| {
            self.dev.destroyImageView(v, null);
        }
        self.dev.destroySwapchainKHR(self.hdl, null);
        Context.inst.destroySurfaceKHR(self.surf, null);
    }
};

pub const Context = struct {
    const Self = @This();

    // Statics
    var vkb: BaseDispatch = undefined;
    var vki: InstanceDispatch = undefined;
    var inst: Instance = undefined;
    var debug_msgr: ?vk.DebugUtilsMessengerEXT = null;

    // Per-context
    pd: vk.PhysicalDevice,
    pd_props: vk.PhysicalDeviceProperties,
    mem_props: vk.PhysicalDeviceMemoryProperties,

    devd: DeviceDispatch,
    dev: Device,

    // One queue of each type, no practical reason to have more.
    queues: [@typeInfo(QueueType).@"enum".fields.len]Queue,

    sc: Swapchain,

    pub fn create(
        ator: Allocator,
        opts: InitOptions,
    ) !*Context {
        const self = try ator.create(Self);

        try createInstance(ator, opts.name);
        try setupDbgMsgr();

        try self.getPhysicalDevice(ator);
        try self.createDevice(ator);
        try self.sc.init(ator, self.pd, self.dev, opts.window);

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.dev.deviceWaitIdle() catch unreachable;
        self.sc.deinit();
        self.dev.destroyDevice(null);
        if (bconf.validation_layer) {
            if (debug_msgr) |msgr| {
                inst.destroyDebugUtilsMessengerEXT(msgr, null);
            }
        }
        inst.destroyInstance(null);
    }

    pub fn getQueue(self: *Self, qtype: QueueType) Queue {
        return self.queues[@intFromEnum(qtype)];
    }

    pub fn createCmdPool(self: *Self, ci: vk.CommandPoolCreateInfo) !CommandPool {
        return .{
            .ctx = self,
            .hdl = try self.dev.createCommandPool(&ci, null),
        };
    }

    pub fn destroyCmdPool(self: *Self, pool: CommandPool) void {
        self.dev.destroyCommandPool(pool.hdl, null);
    }

    pub fn createFence(self: *Self, flags: vk.FenceCreateFlags) vk.Fence {
        return self.dev.createFence(&.{ .flags = flags }, null) catch unreachable;
    }

    pub fn createSemaphore(self: *Self) vk.Semaphore {
        return self.dev.createSemaphore(&.{}, null) catch unreachable;
    }

    pub fn destroyFence(self: *Self, hdl: vk.Fence) void {
        self.dev.destroyFence(hdl, null);
    }

    pub fn destroySemaphore(self: *Self, hdl: vk.Semaphore) void {
        self.dev.destroySemaphore(hdl, null);
    }

    fn createInstance(ator: Allocator, app_name: [*:0]const u8) !void {
        // Vulkan entrypoint to obtain a VkInstance provided by GLFW
        // https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetInstanceProcAddr.html
        vkb = try BaseDispatch.load(@as(vk.PfnGetInstanceProcAddr, @ptrCast(&glfw.getInstanceProcAddress)));

        var inst_exts = std.ArrayList([*:0]const u8).init(ator);
        var inst_layers = std.ArrayList([*:0]const u8).init(ator);

        // Extensions required by GLFW
        try inst_exts.appendSlice(glfw.getRequiredInstanceExtensions() orelse {
            std.log.err("Failed to get required VK instance extensions for GLFW. Error = {s}", .{glfw.mustGetError().description});
            return error.MissingInstanceExtensions;
        });

        const layer_props = try vkb.enumerateInstanceLayerPropertiesAlloc(ator);
        // Add optional NVIDIA optimus if on laptop
        for (layer_props) |prop| {
            const target = "VK_LAYER_NV_optimus";
            if (std.mem.eql(u8, (&prop.layer_name)[0..target.len], target))
                try inst_layers.append(target);
        }

        // Add validation layer and associated extensios
        if (bconf.validation_layer) {
            const validation_layer_present = for (layer_props) |prop| {
                const target = "VK_LAYER_KHRONOS_validation";
                if (std.mem.eql(u8, (&prop.layer_name)[0..target.len], target))
                    break true;
            } else false;

            if (!validation_layer_present) {
                std.log.err("Debug Layer requested but Validationa Layer not present", .{});
                return error.ValidationLayerNotPresent;
            }
            try inst_exts.append("VK_EXT_debug_utils");
            try inst_layers.append("VK_LAYER_KHRONOS_validation");
        }

        for (inst_layers.items) |name| {
            std.debug.print("Layer Name: {s}\n", .{name});
        }
        for (inst_exts.items) |name| {
            std.debug.print("Ext Name: {s}\n", .{name});
        }

        const inst_hdl = try vkb.createInstance(&vk.InstanceCreateInfo{
            .p_application_info = &vk.ApplicationInfo{
                .p_application_name = app_name,
                .application_version = vk.makeApiVersion(0, 0, 1, 0),
                .p_engine_name = app_name,
                .engine_version = vk.makeApiVersion(0, 0, 1, 0),
                .api_version = api_version.version,
            },
            .enabled_layer_count = @intCast(inst_layers.items.len),
            .pp_enabled_layer_names = @ptrCast(inst_layers.items),
            .enabled_extension_count = @intCast(inst_exts.items.len),
            .pp_enabled_extension_names = @ptrCast(inst_exts.items),
        }, null);

        vki = try InstanceDispatch.load(inst_hdl, vkb.dispatch.vkGetInstanceProcAddr);
        inst = Instance.init(inst_hdl, &vki);
    }

    fn setupDbgMsgr() !void {
        if (!bconf.validation_layer)
            return;

        debug_msgr = try inst.createDebugUtilsMessengerEXT(&.{
            .message_type = .{ .validation_bit_ext = true, .performance_bit_ext = true },
            .message_severity = .{ .error_bit_ext = true, .warning_bit_ext = true, .verbose_bit_ext = true, .info_bit_ext = true },
            .pfn_user_callback = debugCallback,
        }, null);
    }

    fn getPhysicalDevice(self: *Self, ator: Allocator) !void {
        const pds = try inst.enumeratePhysicalDevicesAlloc(ator);
        for (pds) |pd| {
            const props = inst.getPhysicalDeviceProperties(pd);
            if (props.device_type == .discrete_gpu) {
                self.pd = pd;
                self.pd_props = props;
                break;
            }
        }
        std.debug.print("Using physical device: {s}\n", .{self.pd_props.device_name});
    }

    fn createDevice(self: *Self, ator: Allocator) !void {
        // TODO: Redo queue search function
        //
        // 1. Setup configuration
        //      - Desired Queue Family
        //

        var dstack = memh.Dstack.init(ator);
        defer dstack.deinit();

        // Find suitable queue families
        const qf_props = try inst.getPhysicalDeviceQueueFamilyPropertiesAlloc(self.pd, dstack.ator());
        for (qf_props, 0..) |qf, qf_idx| {
            if (qf.queue_flags.graphics_bit == true) {
                self.queues[@intFromEnum(QueueType.graphics)].fam = @intCast(qf_idx);
            }
            if (qf.queue_flags.compute_bit == true) {
                self.queues[@intFromEnum(QueueType.compute)].fam = @intCast(qf_idx);
            }
            if (qf.queue_flags.transfer_bit == true) {
                self.queues[@intFromEnum(QueueType.transfer)].fam = @intCast(qf_idx);
            }
        }

        // Get unique set of queue families
        var set = std.AutoHashMap(u32, void).init(dstack.ator());
        for (self.queues) |q| {
            try set.put(q.fam, {});
        }

        // Setup queue create infos per unique queue family
        var q_cinfos = std.ArrayList(vk.DeviceQueueCreateInfo).init(dstack.ator());
        var it = set.iterator();
        while (it.next()) |entry| {
            try q_cinfos.append(.{
                .queue_count = 1,
                .queue_family_index = entry.key_ptr.*,
                .p_queue_priorities = &.{1.0},
            });
        }

        // Turn on/off features available in the physical device
        const feats = inst.getPhysicalDeviceFeatures(self.pd);

        // Queue capabilities are dependent on queue family identified by index
        const dev = try inst.createDevice(self.pd, &vk.DeviceCreateInfo{
            .queue_create_info_count = @intCast(q_cinfos.items.len),
            .p_queue_create_infos = @ptrCast(q_cinfos.items),
            .enabled_extension_count = 1,
            .pp_enabled_extension_names = &.{vk.extensions.khr_swapchain.name},
            .p_enabled_features = &feats,
        }, null);

        self.devd = try DeviceDispatch.load(dev, vki.dispatch.vkGetDeviceProcAddr);
        self.dev = Device.init(dev, &self.devd);

        // Retrieve queues. Queue proxy needs Device dispatch table because
        // the QueueSubmit and similar functions are loaded into the table there
        for (&self.queues) |*q| {
            q.api = QueueProxy.init(self.dev.getDeviceQueue(q.fam, 0), &self.devd);
        }
    }
};

fn debugCallback(
    message_severity: vk.DebugUtilsMessageSeverityFlagsEXT,
    message_types: vk.DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: ?*const vk.DebugUtilsMessengerCallbackDataEXT,
    p_user_data: ?*anyopaque,
    // _: vk.DebugUtilsMessageSeverityFlagsEXT,
    // _: vk.DebugUtilsMessageTypeFlagsEXT,
    // _: ?*const vk.DebugUtilsMessengerCallbackDataEXT,
    // _: ?*anyopaque,
) callconv(vk.vulkan_call_conv) vk.Bool32 {
    _ = message_severity;
    _ = message_types;
    _ = p_user_data;
    std.log.err("{s}\n", .{p_callback_data.?.p_message.?});

    return vk.FALSE;
}
