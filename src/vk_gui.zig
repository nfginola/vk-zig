const std = @import("std");
const Allocator = std.mem.Allocator;
const zgui = @import("zgui");
const vk = @import("vulkan");
const vkt = @import("vk_types.zig");
const vtx = @import("vtx.zig");
const vkb = @import("vk_base.zig");

var dpool: vkt.DescriptorPool = undefined;
var ctx: *vtx = undefined;

pub fn init(ator: Allocator, vk_ctx: *vtx) !void {
    ctx = vk_ctx;

    dpool = try ctx.createDescPool(null, 10, &[_]vk.DescriptorPoolSize{
        vk.DescriptorPoolSize{
            .descriptor_count = 10,
            .type = .combined_image_sampler,
        },
    }, .{ .free_descriptor_set_bit = true });

    const prend: zgui.backend.VkPipelineRenderingCreateInfo = .{
        .s_type = @intFromEnum(vk.StructureType.pipeline_rendering_create_info),
        .color_attachment_count = 1,
        .p_color_attachment_formats = &[_]i32{@intFromEnum(ctx.sc.getFormat())},
        .view_mask = 0,
        .depth_attachment_format = 0,
        .stencil_attachment_format = 0,
    };

    const vk_init: zgui.backend.ImGui_ImplVulkan_InitInfo = .{
        .instance = @intFromEnum(vkb.inst.handle),
        .physical_device = @intFromEnum(ctx.pd),
        .device = @intFromEnum(ctx.dev.handle),
        .queue_family = ctx.getQueue(.graphics).fam.?,
        .queue = @intFromEnum(ctx.getQueue(.graphics).api.handle),
        .descriptor_pool = @intFromEnum(dpool.hdl),
        .min_image_count = 2,
        .image_count = 2,
        .render_pass = 0,
        .use_dynamic_rendering = true,
        .pipeline_rendering_create_info = prend,
    };

    const vk_loader = struct {
        pub fn load(function_name: [*:0]const u8, user_data: ?*anyopaque) callconv(.C) ?*anyopaque {
            if (user_data == null)
                std.debug.assert(false);

            const instance: *vk.Instance = @ptrCast(@alignCast(user_data.?));
            return @constCast(@ptrCast(vkb.get_instance_proc_adr(instance.*, function_name)));
        }
    }.load;

    zgui.init(ator);
    zgui.io.setConfigFlags(.{ .dock_enable = true, .is_srgb = true });
    const style = zgui.getStyle();
    zgui.Style.setColorsBuiltin(style, .dark);

    _ = zgui.backend.loadFunctions(vk_loader, &vkb.inst.handle);
    zgui.backend.init(vk_init, ctx.sc.native.win.handle);
}

pub fn deinit() void {
    zgui.backend.deinit();
    zgui.deinit();
    ctx.destroyDescPool(dpool);
}
