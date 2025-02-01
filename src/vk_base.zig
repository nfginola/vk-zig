const bconf = @import("VK_CONF");
const std = @import("std");
const Allocator = std.mem.Allocator;
const vk = @import("vulkan");

// Each API info gets merged, so this is just a convenient way to group
// Versions are additive, so we need previous ones if we enable more.
const apis: []const vk.ApiInfo = &.{
    vk.features.version_1_3,
    vk.features.version_1_2,
    vk.features.version_1_1,
    vk.features.version_1_0,
    vk.extensions.khr_surface,
    vk.extensions.khr_swapchain,
    vk.extensions.khr_dynamic_rendering,

    if (bconf.validation_layer)
        vk.extensions.ext_debug_utils
    else
        .{},
};
pub const api_version = apis[0];

// Dispatch tables
pub const BaseDispatch = vk.BaseWrapper(apis); // Non-instance functions
pub const InstanceDispatch = vk.InstanceWrapper(apis); // Instance functions
pub const DeviceDispatch = vk.DeviceWrapper(apis); // Device functions (and sub-structures like queue)
pub const Instance = vk.InstanceProxy(apis);

pub const Device = vk.DeviceProxy(apis);
pub const CommandBuffer = vk.CommandBufferProxy(apis);
pub const Queue = vk.QueueProxy(apis);

var vkb: BaseDispatch = undefined;
var debug_msgr: ?vk.DebugUtilsMessengerEXT = null;

pub var vki: InstanceDispatch = undefined;
pub var inst: Instance = undefined;

pub var get_instance_proc_adr: vk.PfnGetInstanceProcAddr = undefined;

pub fn init(
    ator: Allocator,
    app_name: [*:0]const u8,
    get_inst_fn: vk.PfnGetInstanceProcAddr,
    ext_req_exts: [][*:0]const u8,
) !void {
    // Vulkan entrypoint to obtain a VkInstance provided by GLFW
    // https://registry.khronos.org/vulkan/specs/latest/man/html/vkGetInstanceProcAddr.html
    vkb = try BaseDispatch.load(get_inst_fn);
    get_instance_proc_adr = get_inst_fn;

    var inst_exts = std.ArrayList([*:0]const u8).init(ator);
    var inst_layers = std.ArrayList([*:0]const u8).init(ator);

    // Extensions required by GLFW
    try inst_exts.appendSlice(ext_req_exts);

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

    try setupDbgMsgr();
}

pub fn deinit() void {
    if (bconf.validation_layer) {
        if (debug_msgr) |msgr| {
            inst.destroyDebugUtilsMessengerEXT(msgr, null);
        }
    }
    inst.destroyInstance(null);
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

fn debugCallback(
    message_severity: vk.DebugUtilsMessageSeverityFlagsEXT,
    message_types: vk.DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: ?*const vk.DebugUtilsMessengerCallbackDataEXT,
    p_user_data: ?*anyopaque,
) callconv(vk.vulkan_call_conv) vk.Bool32 {
    _ = message_severity;
    _ = message_types;
    _ = p_user_data;
    std.log.err("{s}\n", .{p_callback_data.?.p_message.?});

    return vk.FALSE;
}
