const bconf = @import("VK_CONF"); // Build configs
const std = @import("std");
const glfw = @import("mach-glfw");
const vk = @import("vulkan");
const Allocator = std.mem.Allocator;

pub const InitOptions = struct {
    name: [*:0]const u8 = "Default Name",
    window: glfw.Window,
    debug: bool = true,
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

const Instance = vk.InstanceProxy(apis);
const Device = vk.DeviceProxy(apis);
const Queue = vk.QueueProxy(apis);

pub const Context = struct {
    const Self = @This();

    // Statics
    var vkb: BaseDispatch = undefined;
    var vki: InstanceDispatch = undefined;
    var inst: Instance = undefined;
    var debug_msgr: ?vk.DebugUtilsMessengerEXT = null;

    // Per-context
    surf: vk.SurfaceKHR = undefined,
    pd: vk.PhysicalDevice = undefined,
    pd_props: vk.PhysicalDeviceProperties = undefined,
    mem_props: vk.PhysicalDeviceMemoryProperties = undefined,

    devd: DeviceDispatch = undefined,
    dev: Device = undefined,

    gq: Queue = undefined,
    pq: Queue = undefined,
    tq: Queue = undefined,
    sc: vk.SwapchainKHR = undefined,

    pub fn init(
        ext_ator: Allocator,
        opts: InitOptions,
    ) !Context {
        var self: Self = undefined;
        var arena = std.heap.ArenaAllocator.init(ext_ator);
        const ator = arena.allocator();
        defer arena.deinit();

        try create_instance(ator, opts.name);
        try setup_debug_msgr();

        if (glfw.createWindowSurface(inst.handle, opts.window, null, &self.surf) != @intFromEnum(vk.Result.success))
            return error.SurfaceInitFailed;

        try self.get_physical_device(ator);
        try self.create_device(ator);
        try self.create_swapchain(ator, opts.window);

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.dev.deviceWaitIdle() catch unreachable;

        self.dev.destroySwapchainKHR(self.sc, null);
        self.dev.destroyDevice(null);
        inst.destroySurfaceKHR(self.surf, null);
        if (bconf.validation_layer) {
            if (debug_msgr) |msgr| {
                inst.destroyDebugUtilsMessengerEXT(msgr, null);
            }
        }
        inst.destroyInstance(null);
    }

    fn create_instance(ator: Allocator, app_name: [*:0]const u8) !void {
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

    fn setup_debug_msgr() !void {
        if (!bconf.validation_layer)
            return;

        debug_msgr = try inst.createDebugUtilsMessengerEXT(&.{
            .message_type = .{ .validation_bit_ext = true, .performance_bit_ext = true },
            .message_severity = .{ .error_bit_ext = true, .warning_bit_ext = true, .verbose_bit_ext = true, .info_bit_ext = true },
            .pfn_user_callback = debugCallback,
        }, null);
    }

    fn get_physical_device(self: *Self, ator: Allocator) !void {
        const pds = try inst.enumeratePhysicalDevicesAlloc(ator);
        for (pds) |pd| {
            const props = inst.getPhysicalDeviceProperties(pd);
            if (props.device_type == .discrete_gpu) {
                self.pd = pd;
                self.pd_props = props;
                break;
            }
        }
    }

    fn create_device(self: *Self, ator: Allocator) !void {
        // Find suitable queue families
        var gq_qf: u32 = undefined;
        var pq_qf: u32 = undefined;
        const qf_props = try inst.getPhysicalDeviceQueueFamilyPropertiesAlloc(self.pd, ator);
        for (qf_props, 0..) |qf, qf_idx| {
            if (qf.queue_flags.graphics_bit == true) {
                gq_qf = @intCast(qf_idx);
                pq_qf = gq_qf; // graphics queue can also be used for presenting
            }
        }

        // Get unique set of queue families
        var set = std.AutoHashMap(u32, void).init(ator);
        try set.put(gq_qf, {});
        try set.put(pq_qf, {});

        // Setup queue create infos per unique queue family
        var q_cinfos = std.ArrayList(vk.DeviceQueueCreateInfo).init(ator);
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
        self.gq = Queue.init(self.dev.getDeviceQueue(gq_qf, 0), &self.devd);
        self.pq = Queue.init(self.dev.getDeviceQueue(pq_qf, 0), &self.devd);
    }

    fn create_swapchain(self: *Self, ator: Allocator, win: glfw.Window) !void {
        const sel_fmt: vk.SurfaceFormatKHR = blk: {
            const surf_fmts = try inst.getPhysicalDeviceSurfaceFormatsAllocKHR(self.pd, self.surf, ator);
            for (surf_fmts) |fmt| {
                if (fmt.format == .b8g8r8a8_srgb and fmt.color_space == .srgb_nonlinear_khr) {
                    break :blk fmt;
                }
            }
            return error.SurfFormatNotFound;
        };

        const sel_pmode: vk.PresentModeKHR = blk: {
            const present_modes = try inst.getPhysicalDeviceSurfacePresentModesAllocKHR(self.pd, self.surf, ator);
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
            const surf_caps = try inst.getPhysicalDeviceSurfaceCapabilitiesKHR(self.pd, self.surf);
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

        self.sc = try self.dev.createSwapchainKHR(&vk.SwapchainCreateInfoKHR{
            .surface = self.surf,
            .image_format = sel_fmt.format,
            .image_extent = sc_extent,
            .image_array_layers = 1,
            .image_usage = .{ .color_attachment_bit = true },
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
