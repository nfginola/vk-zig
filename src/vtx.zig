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
// Version on top used as relevant API
const api_version = apis[0];

// API consists of enabling commands for three types:
// Base, Instance, and Device functions. These map to the below wrappers!

// Wrapper for functions that are loaded by vkGetInstanceProcAddr without an instance ()
// vkCreateInstance, vkEnumerateInstanceVersion, etc.
const BaseDispatch = vk.BaseWrapper(apis);

// Wrapper for instance functions that are loaded by vkGetInstanceProcAddr
const InstanceDispatch = vk.InstanceWrapper(apis);
// Convenience wrapper for above
const Instance = vk.InstanceProxy(apis);

// Wrapper for device functions that are loaded by vkGetDeviceProcAddr
const DeviceDispatch = vk.DeviceWrapper(apis);
// Convenience wrapper for above
const Device = vk.DeviceProxy(apis);

const Queue = vk.QueueProxy(apis);

pub const Context = struct {
    const Self = @This();

    // Dispatch tables, file globals
    var vkb: BaseDispatch = undefined; // Holds dispatch table for functions that don't need instance
    var vki: InstanceDispatch = undefined; // Holds dispatch table for functions that need instance
    var inst: Instance = undefined; // Convenience wrapper for InstanceDispatch
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

        try setup_dispatch_tbls(ator, opts);
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

    fn setup_dispatch_tbls(ator: Allocator, opts: InitOptions) !void {
        // 1. Get our Vulkan entrypoint which is provided by GLFW: function that allows us to obtain our VkInstance
        // https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetInstanceProcAddr.html
        vkb = try BaseDispatch.load(@as(vk.PfnGetInstanceProcAddr, @ptrCast(&glfw.getInstanceProcAddress)));

        // 2. Get necessary extensions for GLFW
        var inst_exts = std.ArrayList([*:0]const u8).init(ator);
        var inst_layers = std.ArrayList([*:0]const u8).init(ator);

        try inst_exts.appendSlice(glfw.getRequiredInstanceExtensions() orelse {
            std.log.err("Failed to get required VK instance extensions for GLFW. Error = {s}", .{glfw.mustGetError().description});
            return error.MissingInstanceExtensions;
        });

        // 2. Query some more layers and extensions
        const layer_props = try vkb.enumerateInstanceLayerPropertiesAlloc(ator);

        // On laptop, check availability of NV optimus
        for (layer_props) |prop| {
            const target = "VK_LAYER_NV_optimus";
            if (std.mem.eql(u8, (&prop.layer_name)[0..target.len], target))
                try inst_layers.append("VK_LAYER_NV_optimus");
        }

        if (bconf.validation_layer) {
            // Check for validation layer availability
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

        // 3. Create our Vulkan Instance
        const app_info = vk.ApplicationInfo{
            .p_application_name = opts.name,
            .application_version = vk.makeApiVersion(0, 0, 1, 0),
            .p_engine_name = opts.name,
            .engine_version = vk.makeApiVersion(0, 0, 1, 0),
            .api_version = api_version.version,
        };

        for (inst_layers.items) |name| {
            std.debug.print("Layer Name: {s}\n", .{name});
        }
        for (inst_exts.items) |name| {
            std.debug.print("Ext Name: {s}\n", .{name});
        }

        const inst_hdl = try vkb.createInstance(&vk.InstanceCreateInfo{
            .p_application_info = &app_info,
            .enabled_layer_count = @intCast(inst_layers.items.len),
            .pp_enabled_layer_names = @ptrCast(inst_layers.items),
            .enabled_extension_count = @intCast(inst_exts.items.len),
            .pp_enabled_extension_names = @ptrCast(inst_exts.items),
        }, null);

        // 4.   Initialize 'proxying wrappers': Bundles 'handle' and associated 'functions' into a struct for convenience!
        // 4.1  Populates dispatch table functions using GetInstanceProc (loader) and function names
        vki = try InstanceDispatch.load(inst_hdl, vkb.dispatch.vkGetInstanceProcAddr);

        // 4.2  Just a proxying wrapper, holds the handle.
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
            const props = vki.getPhysicalDeviceProperties(pd);
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
        const qf_props = try vki.getPhysicalDeviceQueueFamilyPropertiesAlloc(self.pd, ator);
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
        const feats = vki.getPhysicalDeviceFeatures(self.pd);

        // Queue capabilities are dependent on queue family identified by index
        const dev = try vki.createDevice(self.pd, &vk.DeviceCreateInfo{
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
            const surf_fmts = try vki.getPhysicalDeviceSurfaceFormatsAllocKHR(self.pd, self.surf, ator);
            for (surf_fmts) |fmt| {
                if (fmt.format == .b8g8r8a8_srgb and fmt.color_space == .srgb_nonlinear_khr) {
                    break :blk fmt;
                }
            }
            return error.SurfFormatNotFound;
        };

        const sel_pmode: vk.PresentModeKHR = blk: {
            const present_modes = try vki.getPhysicalDeviceSurfacePresentModesAllocKHR(self.pd, self.surf, ator);
            for (present_modes) |pmode| {
                if (pmode == .mailbox_khr)
                    break :blk pmode;
            }

            // Fallback
            break :blk vk.PresentModeKHR.fifo_khr;
            // FIFO: Show on next vertical blank (vsync)
            // Immediate: May cause tearing (No vsync)
        };

        const sc_extent: vk.Extent2D = blk: {
            const surf_caps = try vki.getPhysicalDeviceSurfaceCapabilitiesKHR(self.pd, self.surf);
            // VkSurfaceCapabilitiesKHR spec:
            // Special value (0xFFFFFFFF, 0xFFFFFFFF) indicating that the surface size will be determined by the extent of a swapchain
            // In this case, we will use the given window client dimensions to specify the extent
            if (surf_caps.current_extent.width == std.math.maxInt(u32)) {
                const draw_area = win.getFramebufferSize();
                break :blk .{
                    .width = std.math.clamp(draw_area.width, surf_caps.current_extent.width, surf_caps.current_extent.width),
                    .height = std.math.clamp(draw_area.height, surf_caps.current_extent.height, surf_caps.current_extent.height),
                };
            }

            break :blk surf_caps.current_extent;
        };

        const sc_ci = vk.SwapchainCreateInfoKHR{
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
        };
        self.sc = try self.dev.createSwapchainKHR(&sc_ci, null);
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
