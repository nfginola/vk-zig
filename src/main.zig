const std = @import("std");
const glfw = @import("mach-glfw");
const zimg = @import("zigimg");
const nvk = @import("vtx.zig");
const vk = @import("vulkan");
const memh = @import("memory_helpers.zig");
const utx = @import("vk_upload.zig");
const vkt = @import("vk_types.zig");

const WIDTH = 1600;
const HEIGHT = 900;

/// GLFW error handling callback
fn errorCallback(error_code: glfw.ErrorCode, description: [:0]const u8) void {
    std.log.err("glfw: {}: {s}\n", .{ error_code, description });
}

fn initGLFW() glfw.Window {
    glfw.setErrorCallback(errorCallback);
    if (!glfw.init(.{})) {
        std.log.err("Failed to initialize GLFW: {?s}", .{glfw.getErrorString()});
        std.process.exit(1);
    }

    // Create our window
    return glfw.Window.create(WIDTH, HEIGHT, "Graphics Application", null, null, .{ .client_api = .no_api, .position_x = 700, .position_y = 300 }) orelse {
        std.log.err("Failed to create Window: {?s}", .{glfw.getErrorString()});
        std.process.exit(1);
    };
}

fn deinitGLFW(window: *const glfw.Window) void {
    window.destroy();
    glfw.terminate();
}

pub fn main() !void {
    const window: glfw.Window = initGLFW();
    defer deinitGLFW(&window);

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        if (gpa.deinit() == .leak) {
            std.log.err("Program leaked memory!\n", .{});
        }
    }

    var arena = memh.Arena.init(gpa.allocator());
    defer arena.deinit();

    var ctx = try nvk.create(arena.ator(), .{ .name = "Vulkan Engine", .window = window });
    const dev = ctx.dev;
    const gq = ctx.getQueue(.graphics);
    defer ctx.deinit();

    var upload = try utx.UploadContext.create(arena.ator(), ctx, .graphics);
    defer upload.deinit();

    // Top level VK resources lifetime arena
    var varena = try ctx.createArena(arena.ator());
    defer varena.deinit();

    var cmdp = try ctx.createCmdPool(varena, .graphics, .{ .transient_bit = true });
    const clear_finished_sem = try ctx.createSemaphore(varena);
    const sc_img_acquired_sem = try ctx.createSemaphore(varena);
    const cmdb_finished_fence = try ctx.createFence(varena, .{ .signaled_bit = true });

    const vb = try ctx.createBuffer(varena, 32_000, .{ .vertex_buffer_bit = true, .transfer_dst_bit = true });
    const ib = try ctx.createBuffer(varena, 32_000, .{ .index_buffer_bit = true, .transfer_dst_bit = true });
    const Vertex = struct {
        x: f32,
        y: f32,
        z: f32,
        u: f32,
        v: f32,
        w: f32,
    };
    {
        // Grab buffer memory
        const vmem = try ctx.allocateMemory(varena, .gpu, 64_000);
        const imem = try ctx.allocateMemory(varena, .gpu, 64_000);
        _ = try ctx.dev.bindBufferMemory(vb.hdl, vmem, 0);
        _ = try ctx.dev.bindBufferMemory(ib.hdl, imem, 0);

        // Upload tri and indices
        const vertices = [_]Vertex{
            .{ .x = 0.0, .y = -0.5, .z = 0.0, .u = 1.0, .v = 0.0, .w = 0.0 },
            .{ .x = -0.5, .y = 0.5, .z = 0.0, .u = 0.0, .v = 1.0, .w = 0.0 },
            .{ .x = 0.5, .y = 0.5, .z = 0.0, .u = 0.0, .v = 0.0, .w = 1.0 },
        };
        const indices = [_]u32{ 0, 1, 2 };
        try upload.copy_to_buffer(vb, try upload.push(memh.byteSliceC(Vertex, vertices[0..]), 0));
        try upload.copy_to_buffer(ib, try upload.push(memh.byteSliceC(u32, indices[0..]), 0));
        try upload.submit(null);
        try upload.host_wait();
    }

    // TODO:
    //  x Grab glslc
    //  x Write triangle vs and fs shader in GLSL
    //      - vs just passthrough NDC verts and uvs
    //      - fs just outputs uv as colors
    //  x Compile shaders to SPIR-V
    //  x Create shader module
    //      - Verifies that shaders are O.K! :)
    //  x Make pipeline for triangle
    //  x Support multi-queue and multi-family. (One queue per distinct family, or fallback to gfx queues)
    //
    //  x Support automatic qf ownership transfer from dedicated transfer queue
    //
    //  x Replace vk.Buffer with custom nvk.Buffer
    //      x Prep for VMA
    //  x Garbage can (arena for vk resources)
    //
    //  x Use input layout and VB/IB to render triangle
    //
    //  x Move swapchain to vsc.zig
    //  x Refactor vtx.zig to not have to need a struct for variables and functions?
    //      (Look at Allocator as example)
    //
    //  x Refactored garbage can to be generic (nicer callsite)
    //  x Refactored and split vk init into base for instance
    //
    //  - Add VMA support
    //
    //  - Make math helpers
    //      - Vector2,3,4
    //          - vadd, vsub, vdivFac, vmulFac, dot, cross
    //      - Mat2,3,4
    //          - store row-major for easy mul calc (n row*column)
    //          - mat2 x v2
    //          - mat3 x v3
    //          - mat4 x v4
    //      - Use Right Handed coordinate system with Z up and Y into screen
    //
    //  - Some simple ass render graph to avoid manual barriers
    //

    const p_layout = try ctx.dev.createPipelineLayout(&vk.PipelineLayoutCreateInfo{}, null);
    defer ctx.dev.destroyPipelineLayout(p_layout, null);

    const target_details = vk.PipelineRenderingCreateInfoKHR{
        .color_attachment_count = 1,
        .p_color_attachment_formats = &.{ctx.sc.native.format.format},
        .view_mask = 0,
        .depth_attachment_format = .undefined,
        .stencil_attachment_format = .undefined,
    };
    var pipes: [1]vk.Pipeline = undefined;
    _ = try ctx.dev.createGraphicsPipelines(.null_handle, pipes.len, &.{
        vk.GraphicsPipelineCreateInfo{
            .p_next = &target_details,
            .stage_count = 2,
            .p_stages = &.{
                vk.PipelineShaderStageCreateInfo{
                    .module = try ctx.createShaderModuleFromFile(arena.ator(), varena, "res/shaders/compiled/tri.spv"),
                    .stage = .{ .vertex_bit = true },
                    .p_name = "main",
                },
                vk.PipelineShaderStageCreateInfo{
                    .module = try ctx.createShaderModuleFromFile(arena.ator(), varena, "res/shaders/compiled/pass.spv"),
                    .stage = .{ .fragment_bit = true },
                    .p_name = "main",
                },
            },
            .p_vertex_input_state = &vk.PipelineVertexInputStateCreateInfo{
                .vertex_binding_description_count = 1,
                .p_vertex_binding_descriptions = &.{
                    vk.VertexInputBindingDescription{
                        .binding = 0,
                        .input_rate = .vertex,
                        .stride = @sizeOf(Vertex),
                    },
                },
                .vertex_attribute_description_count = 2,
                .p_vertex_attribute_descriptions = &.{
                    vk.VertexInputAttributeDescription{
                        .binding = 0,
                        .location = 0,
                        .format = .r32g32b32_sfloat,
                        .offset = @offsetOf(Vertex, "x"),
                    },
                    vk.VertexInputAttributeDescription{
                        .binding = 0,
                        .location = 1,
                        .format = .r32g32b32_sfloat,
                        .offset = @offsetOf(Vertex, "u"),
                    },
                },
            },
            .p_input_assembly_state = &vk.PipelineInputAssemblyStateCreateInfo{
                .topology = .triangle_list,
                .primitive_restart_enable = vk.FALSE,
            },
            .p_multisample_state = &vk.PipelineMultisampleStateCreateInfo{
                .rasterization_samples = .{ .@"1_bit" = true },
                .sample_shading_enable = vk.FALSE,
                .min_sample_shading = 0,
                .alpha_to_coverage_enable = vk.FALSE,
                .alpha_to_one_enable = vk.FALSE,
            },
            .p_depth_stencil_state = &vk.PipelineDepthStencilStateCreateInfo{
                .depth_test_enable = vk.FALSE,
                .depth_write_enable = vk.FALSE,
                .depth_compare_op = .always,
                .depth_bounds_test_enable = vk.FALSE,
                .stencil_test_enable = vk.FALSE,
                .front = std.mem.zeroInit(vk.StencilOpState, .{}),
                .back = std.mem.zeroInit(vk.StencilOpState, .{}),
                .min_depth_bounds = 0.0,
                .max_depth_bounds = 1.0,
            },
            .p_rasterization_state = &vk.PipelineRasterizationStateCreateInfo{
                .cull_mode = .{ .back_bit = true },
                .front_face = .counter_clockwise,
                .polygon_mode = .fill,
                .depth_clamp_enable = vk.TRUE,
                .rasterizer_discard_enable = vk.FALSE,
                .depth_bias_enable = vk.FALSE,
                .depth_bias_clamp = 0.0,
                .depth_bias_constant_factor = 0.0,
                .depth_bias_slope_factor = 0.0,
                .line_width = 1.0,
            },
            .p_color_blend_state = &vk.PipelineColorBlendStateCreateInfo{
                .logic_op_enable = vk.FALSE,
                .logic_op = .clear,
                .attachment_count = 1,
                .blend_constants = .{ 0.0, 0.0, 0.0, 0.0 },
                .p_attachments = &.{
                    vk.PipelineColorBlendAttachmentState{
                        .blend_enable = vk.FALSE,
                        .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true },
                        .src_alpha_blend_factor = .one,
                        .alpha_blend_op = .add,
                        .color_blend_op = .add,
                        .dst_alpha_blend_factor = .one,
                        .dst_color_blend_factor = .one,
                        .src_color_blend_factor = .one,
                    },
                },
            },
            .p_viewport_state = &vk.PipelineViewportStateCreateInfo{
                .scissor_count = 1,
                .viewport_count = 1,
                .p_viewports = &.{
                    vk.Viewport{
                        .x = 0,
                        .y = 0,
                        .width = @floatFromInt(ctx.sc.getExtent().width),
                        .height = @floatFromInt(ctx.sc.getExtent().height),
                        .min_depth = 0.0,
                        .max_depth = 1.0,
                    },
                },
                .p_scissors = &.{
                    vk.Rect2D{
                        .offset = .{
                            .x = 0,
                            .y = 0,
                        },
                        .extent = ctx.sc.getExtent(),
                    },
                },
            },
            .layout = p_layout,
            .subpass = 0,
            .base_pipeline_handle = .null_handle,
            .base_pipeline_index = 0,
        },
    }, null, &pipes);
    defer ctx.dev.destroyPipeline(pipes[0], null);

    while (!window.shouldClose()) {
        if (window.getKey(glfw.Key.escape) == glfw.Action.press) {
            break;
        }

        try ctx.waitResetFences(&[_]vk.Fence{cmdb_finished_fence});
        const sc_next = try ctx.sc.getNext(sc_img_acquired_sem, null);

        try cmdp.reset(.{});
        var cmdb = try cmdp.alloc(.primary, 1);
        try cmdb.beginCommandBuffer(&.{ .flags = .{ .one_time_submit_bit = true } });

        cmdb.pipelineBarrier(.{ .bottom_of_pipe_bit = true }, .{ .color_attachment_output_bit = true }, .{}, 0, null, 0, null, 1, &.{
            vk.ImageMemoryBarrier{
                .old_layout = .undefined,
                .new_layout = .color_attachment_optimal,
                .src_access_mask = .{},
                .dst_access_mask = .{ .color_attachment_write_bit = true },
                .image = sc_next.image,
                .subresource_range = vkt.Utils.fullSubres(.{ .color_bit = true }),
                .src_queue_family_index = gq.fam.?,
                .dst_queue_family_index = gq.fam.?,
            },
        });

        const color_att: vk.RenderingAttachmentInfoKHR = .{
            .clear_value = .{ .color = .{ .float_32 = .{ 0.3, 0.4, 0.3, 1.0 } } },
            .image_layout = .color_attachment_optimal,
            .image_view = sc_next.view,
            .load_op = .clear,
            .store_op = .store,
            .resolve_mode = .{},
            .resolve_image_layout = .undefined,
        };

        const rinfo: vk.RenderingInfoKHR = .{
            .render_area = .{ .extent = .{ .width = window.getFramebufferSize().width, .height = window.getFramebufferSize().height }, .offset = .{ .x = 0, .y = 0 } },
            .layer_count = 1,
            .color_attachment_count = 1,
            .p_color_attachments = &.{color_att},
            .view_mask = 0,
        };
        cmdb.beginRenderingKHR(&rinfo);
        cmdb.bindPipeline(.graphics, pipes[0]);
        cmdb.bindVertexBuffers(0, 1, &.{vb.hdl}, &.{0});
        cmdb.bindIndexBuffer(ib.hdl, 0, .uint32);
        cmdb.drawIndexed(3, 1, 0, 0, 0);
        // cmdb.draw(3, 1, 0, 0);
        cmdb.endRenderingKHR();

        cmdb.pipelineBarrier(.{ .color_attachment_output_bit = true }, .{ .top_of_pipe_bit = true }, .{}, 0, null, 0, null, 1, &.{
            vk.ImageMemoryBarrier{
                .old_layout = .color_attachment_optimal,
                .new_layout = .present_src_khr,
                .src_access_mask = .{ .color_attachment_write_bit = true },
                .dst_access_mask = .{},
                .image = sc_next.image,
                .subresource_range = vkt.Utils.fullSubres(.{ .color_bit = true }),
                .src_queue_family_index = gq.fam.?,
                .dst_queue_family_index = gq.fam.?,
            },
        });

        cmdb.endCommandBuffer() catch unreachable;

        try gq.api.submit(1, &.{
            vk.SubmitInfo{
                .command_buffer_count = 1,
                .p_command_buffers = &.{cmdb.handle},
                .p_signal_semaphores = &.{clear_finished_sem},
                .signal_semaphore_count = 1,
                .p_wait_semaphores = &.{sc_img_acquired_sem},
                .wait_semaphore_count = 1,
                .p_wait_dst_stage_mask = &.{.{ .top_of_pipe_bit = true }},
            },
        }, cmdb_finished_fence);

        try ctx.sc.present(gq, clear_finished_sem);

        glfw.pollEvents();
    }

    // Wait for idle before any resource destruction
    dev.deviceWaitIdle() catch unreachable;
}
