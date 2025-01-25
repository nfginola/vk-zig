const std = @import("std");
const glfw = @import("mach-glfw");
const zimg = @import("zigimg");
const nvk = @import("vtx.zig");
const vk = @import("vulkan");
const memh = @import("memory_helpers.zig");
const utx = @import("vk_upload.zig");
const vkt = @import("vk_types.zig");
const vkds = @import("vk_ds.zig");

const WIDTH = 1600;
const HEIGHT = 900;
const MAX_FIF = 2;

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
    return glfw.Window.create(WIDTH, HEIGHT, "Graphics Application", null, null, .{
        .client_api = .no_api,
        .position_x = 700,
        .position_y = 300,
        .resizable = true,
    }) orelse {
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

    var upload = try utx.create(arena.ator(), ctx, 64_000);
    defer upload.deinit();

    // Top level VK resources lifetime arena
    var varena = try ctx.createArena(arena.ator());
    defer varena.deinit();

    const Vertex = packed struct {
        x: f32,
        y: f32,
        z: f32,
        r: f32,
        g: f32,
        b: f32,
    };
    const vb = try ctx.createBufferWithMemory(varena, .{
        .size = 32_000,
        .usage = .{ .vertex_buffer_bit = true, .transfer_dst_bit = true, .shader_device_address_bit = true },
        .mem_type = .gpu,
    });
    const ib = try ctx.createBufferWithMemory(varena, .{
        .size = 32_000,
        .usage = .{ .index_buffer_bit = true, .transfer_dst_bit = true, .shader_device_address_bit = true },
        .mem_type = .gpu,
    });
    // Upload tri and indices
    const vertices = [_]Vertex{
        .{ .x = 0.0, .y = -0.5, .z = 0.0, .r = 1.0, .g = 0.0, .b = 0.0 },
        .{ .x = -0.5, .y = 0.5, .z = 0.0, .r = 0.0, .g = 1.0, .b = 0.0 },
        .{ .x = 0.5, .y = 0.5, .z = 0.0, .r = 0.0, .g = 0.0, .b = 1.0 },
    };
    const indices = [_]u32{ 0, 1, 2 };
    try upload.copy_to_buffer(vb, .{ .vertex_shader_bit = true }, try upload.push(memh.byteSliceC(Vertex, vertices[0..]), 0));
    try upload.copy_to_buffer(ib, .{ .vertex_shader_bit = true }, try upload.push(memh.byteSliceC(u32, indices[0..]), 0));
    try upload.submit(.graphics, null);
    try upload.host_wait();

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
    //  x Stack allocator for per frame data
    //  x FIF
    //
    //  x Run shader with time based colors
    //      x Naive
    //          x PipelineLayout
    //          x Descriptor Pool
    //          x Update descriptor set
    //      x Descriptor indexing
    //
    //  x Refactor descriptor allocation API
    //
    //  x Support buffer device address
    //  x Change upload context to use device address buffers
    //  x Add helper for buffer/memory pair allocation
    //
    //  x Use vertex pulling instead of VB/IB bind
    //      x Remove vertex input state declaration
    //
    //  x Fix RenderDoc
    //      > Had to do with allocateMemory flags not being 'var'.
    //        Memory is modified inside of RenderDoc and since using
    //        const we only guarantee read-only storage, then it
    //        segfaulted on write (it appends replay flag for
    //        buffer device address if not present and using
    //        buffer device address)
    //
    //  x Use timeline semaphores
    //
    //  - Swapchain recreation on resize
    //      - Naive deviceWait preferred
    //      - Since this happens after vkWaitResetFences and we
    //        have yet to wait on any previous semaphores, we
    //        dont need to discard any of those resources and
    //        should resume again on the same PerFrame data once
    //        SC resources have been recreated.
    //        --> getNext() should have a while() or similar to recreate
    //
    //        This can fail on Present too.. at that point the sync items
    //        for the queue submit is still fine.
    //
    //        In other words, one getNext() and present(),
    //        ensure that SC is recreated inside before returning to
    //        normal ensuring normal PF stepping
    //
    //  - Make Gfx pipeline helpers
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

    var pf_stack = try vkds.Stack.init(varena, ctx, .{ .rr_block_size = 2048, .rr_blocks = MAX_FIF, .device_adr = true });
    const PerFrameData = packed struct {
        r: f32,
        g: f32,
        b: f32,
    };

    //
    // Make a Pool Creator:
    //  - Pool for bindless specifically
    //      - Used for Per-Material data!
    //      - Will be texture set + SSBO likely
    //
    //  - Pool for normal usage
    //      - Global data (static)          --> One UBO?
    //      - Per Frame data (dynamic)      --> One UBO?
    //      - Per Pass data (dynamic)       -->
    //
    //  Use #include support to have easy-access CPU/GPU mapping
    //  between global and per frame sets, which should stay
    //  consistent no matter the consumer
    //
    //  With the extra set number 4, we could leave it for any Per-Pass-Interframe data
    //  (temporal buffers and the likes)
    //

    // Make descriptor set layout
    const dlayout = try ctx.createDescSetLayout(arena.ator(), varena, .{
        .bindings = &[_]vkt.DescriptorSetLayoutBinding{
            vkt.DescriptorSetLayoutBinding{
                .binding = .{
                    .binding = 0,
                    .descriptor_count = 1000,
                    .descriptor_type = .uniform_buffer,
                    .stage_flags = .{ .vertex_bit = true },
                },
                .flags = .{
                    .update_after_bind_bit = true,
                    .update_unused_while_pending_bit = true,
                    // Allows shaders to access array elements that may not be explicitly populated with resources.
                    .partially_bound_bit = true,
                    .variable_descriptor_count_bit = true,
                },
            },
        },
        .flags = .{ .update_after_bind_pool_bit = true }, // DDI
    });

    // Make pool, allocate, write to descriptor
    var dpool = try ctx.createDescPool(varena, 1, &[_]vk.DescriptorPoolSize{
        .{
            .descriptor_count = 10_000,
            .type = .uniform_buffer,
        },
    }, .{ .update_after_bind_bit = true }); // DDI
    var dset = try dpool.alloc(.{
        .layout = dlayout,
        .variable_descriptors = 1000, // DDI
    });
    for (0..MAX_FIF) |frame| {
        dset.writeBuffer(.{
            .buf = pf_stack.buf,
            .buf_offset = pf_stack.getOffset(@intCast(frame)),
            .buf_range = pf_stack.getBlockSize(),
            .dst_binding = 0,
            .dst_array_el = @intCast(frame),
            .dst_type = .uniform_buffer,
        });
    }

    // Need packed to preserve memory ordering
    // 8 byte alignment of u64 causes gap..
    // sort by largest to smallest top-down to avoid having to pad
    const PushConstant = packed struct {
        adr: u64,
        pf_adr: u64,
        vb_adr: u64,
        ib_adr: u64,
    };

    const p_layout = try ctx.dev.createPipelineLayout(&vk.PipelineLayoutCreateInfo{
        .set_layout_count = 2,
        .p_set_layouts = &.{ dlayout, dlayout }, // TODO: Set order --> Global, Per Frame, Per Pass, <Flexible 4th slot>
        .push_constant_range_count = 1,
        .p_push_constant_ranges = &.{vk.PushConstantRange{
            .stage_flags = .{ .vertex_bit = true },
            .offset = 0,
            .size = @sizeOf(PushConstant),
        }},
    }, null);
    defer ctx.dev.destroyPipelineLayout(p_layout, null);

    // ======================== SETUP DDI end

    const target_details = vk.PipelineRenderingCreateInfoKHR{
        .color_attachment_count = 1,
        .p_color_attachment_formats = &.{ctx.sc.native.format.format},
        .view_mask = 0,
        .depth_attachment_format = .undefined,
        .stencil_attachment_format = .undefined,
    };

    const dyn_states = &[_]vk.DynamicState{
        // With count required if we want to pass ViewportStateCreateInfo with 0 counts
        // for pipeline --> which we needed since rasterizer_discard_enable = FALSE,
        // requires a VPci even if we dynamically set
        vk.DynamicState.viewport_with_count,
        vk.DynamicState.scissor_with_count,
    };

    const dyn_state_ci = vk.PipelineDynamicStateCreateInfo{
        .dynamic_state_count = dyn_states.len,
        .p_dynamic_states = dyn_states,
    };

    var pipes: [1]vk.Pipeline = undefined;
    _ = try ctx.dev.createGraphicsPipelines(.null_handle, pipes.len, &.{
        vk.GraphicsPipelineCreateInfo{
            .p_dynamic_state = &dyn_state_ci,
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
            .p_vertex_input_state = &vk.PipelineVertexInputStateCreateInfo{},
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
            // If rasterizer_discard_enable = FALSE, spec requires us setting a Viewport (even if dynamic)
            .p_viewport_state = &vk.PipelineViewportStateCreateInfo{},
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
            .layout = p_layout,
            .subpass = 0,
            .base_pipeline_handle = .null_handle,
            .base_pipeline_index = 0,
        },
    }, null, &pipes);
    defer ctx.dev.destroyPipeline(pipes[0], null);

    const PerFrame = struct {
        // cpu
        farena: memh.Arena, // frame arena

        // gpu
        cmdp: vkt.CommandPool,
        sem_img_acq: vk.Semaphore,
        sem_ready_present: vk.Semaphore, // for present queue TODO: consider using timeline semaphores

        sem_work_finished: vkt.Semaphore, // for future frames
        fence_work_finished: vk.Fence,

        worked: bool = false, // if this GPU frame was skipped
    };

    var pf: [MAX_FIF]PerFrame = undefined;
    for (&pf) |*f| {
        // cpu
        f.farena = memh.Arena.init(arena.ator());

        // gpu
        f.cmdp = try ctx.createCmdPool(varena, .graphics, .{ .transient_bit = true });
        f.sem_img_acq = try ctx.createSemaphore(varena);
        f.sem_ready_present = try ctx.createSemaphore(varena);
        f.sem_work_finished = try ctx.createSemaphoreT(varena);
        f.fence_work_finished = try ctx.createFence(varena, .{ .signaled_bit = true });
    }

    // Hack for semaphore waiting, consider using timeline semaphores
    // to be able to wait for an monotonically increasing counter..
    var first_frame: bool = true;

    var prev_pf: PerFrame = pf[MAX_FIF - 1];
    var curr_f: u32 = 0;

    var dt: f32 = 0.0; // in s
    const init_disp_interval = 0.01;
    var display_dt_interval: f32 = init_disp_interval; // in s
    var elapsed: f32 = 0.0; // in s

    // ======== Create buffer using buffer device address
    const bd = try ctx.createBufferWithMemory(varena, .{
        .size = 64_000,
        .mem_type = .cpu_to_gpu,
        .usage = .{ .shader_device_address_bit = true, .storage_buffer_bit = true },
    });
    const bd_map = try ctx.mapBuffer(f32, bd);
    bd_map[0] = 0.2;
    bd_map[1] = 0.8;
    bd_map[2] = 0.5;

    while (!window.shouldClose()) {
        glfw.pollEvents();

        defer first_frame = false;
        var curr_pf = &pf[curr_f];
        defer curr_f = (curr_f + 1) % MAX_FIF;
        defer prev_pf = pf[curr_f];
        defer _ = curr_pf.farena.arena.reset(.retain_capacity);

        const start_time = std.time.microTimestamp();
        defer {
            const end_time = std.time.microTimestamp();
            dt = @as(f32, @floatFromInt(end_time - start_time)) / 1_000_000.0;

            elapsed += dt;

            display_dt_interval -= dt;
            if (display_dt_interval < 0) {
                const title = std.fmt.allocPrintZ(
                    curr_pf.farena.ator(),
                    "Graphics Application - {d:.2}- {d:.3} ms ({d:.0} FPS)",
                    .{ elapsed, dt * 1000.0, 1.0 / dt },
                ) catch unreachable;
                window.setTitle(title);

                display_dt_interval = init_disp_interval;
            }
        }

        if (window.getKey(glfw.Key.escape) == glfw.Action.press) {
            break;
        }

        // ============================================== GPU

        var cmdp = curr_pf.cmdp;

        if (curr_pf.worked) // if last iteration of this pf worked, only then we should wait
            try ctx.waitForFences(&[_]vk.Fence{curr_pf.fence_work_finished});
        try ctx.resetFences(&[_]vk.Fence{curr_pf.fence_work_finished});
        if (window.getKey(glfw.Key.r) == glfw.Action.press) {
            curr_pf.worked = false;
            continue;
        }

        const sc_next = try ctx.sc.getNext(curr_pf.sem_img_acq, null) orelse {
            curr_pf.worked = false;
            continue;
        };
        curr_pf.worked = true;

        defer pf_stack.next_block();
        const dyn = try pf_stack.grab(PerFrameData, 0);
        dyn.r = @cos(elapsed + dt * 7) * 0.5 + 0.5;
        dyn.g = @sin(elapsed + dt * 2) * 0.5 + 0.5;
        dyn.b = @sin(elapsed + dt * 3) * 0.5 + 0.5;

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
            .render_area = .{
                .extent = .{
                    .width = ctx.sc.getExtent().width,
                    .height = ctx.sc.getExtent().height,
                },
                .offset = .{ .x = 0, .y = 0 },
            },
            .layer_count = 1,
            .color_attachment_count = 1,
            .p_color_attachments = &.{color_att},
            .view_mask = 0,
        };
        cmdb.bindDescriptorSets(.graphics, p_layout, 0, 1, &.{dset.hdl}, 0, null);
        cmdb.beginRenderingKHR(&rinfo);
        const vp = vk.Viewport{
            .x = 0,
            .y = 0,
            .width = @floatFromInt(ctx.sc.getExtent().width),
            .height = @floatFromInt(ctx.sc.getExtent().height),
            .min_depth = 0.0,
            .max_depth = 1.0,
        };
        const scissor = vk.Rect2D{
            .offset = .{
                .x = 0,
                .y = 0,
            },
            .extent = ctx.sc.getExtent(),
        };

        cmdb.setViewportWithCount(1, @ptrCast(&vp));
        cmdb.setScissorWithCount(1, @ptrCast(&scissor));

        cmdb.bindPipeline(.graphics, pipes[0]);
        const pc: PushConstant = .{
            .adr = bd.gpu_adr.?,
            .pf_adr = pf_stack.buf.gpu_adr.? + pf_stack.getOffset(curr_f),
            .vb_adr = vb.gpu_adr.?,
            .ib_adr = ib.gpu_adr.?,
        };
        cmdb.pushConstants(p_layout, .{ .vertex_bit = true }, 0, @sizeOf(PushConstant), &pc);

        // bind index buffer even with Vertex pulling to allow
        // GPU caching on vertex fetches
        // should profile when we have some more complex scene :)
        cmdb.bindIndexBuffer(ib.hdl, 0, .uint32);
        cmdb.drawIndexed(3, 1, 0, 0, 0);
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

        const timeline = vk.TimelineSemaphoreSubmitInfo{
            .signal_semaphore_value_count = 2,
            .p_signal_semaphore_values = &.{ curr_pf.sem_work_finished.next(), 0 },
            .wait_semaphore_value_count = if (prev_pf.worked) 2 else 1,
            .p_wait_semaphore_values = &.{ 0, prev_pf.sem_work_finished.value },
        };
        try gq.api.submit(1, &.{
            vk.SubmitInfo{
                .command_buffer_count = 1,
                .p_command_buffers = &.{cmdb.handle},
                .signal_semaphore_count = 2,
                .p_signal_semaphores = &.{ curr_pf.sem_work_finished.hdl, curr_pf.sem_ready_present },
                // Chain GPU frames (truly transient per frame resources)
                .wait_semaphore_count = if (prev_pf.worked) 2 else 1,
                .p_wait_semaphores = &.{ curr_pf.sem_img_acq, prev_pf.sem_work_finished.hdl },
                .p_wait_dst_stage_mask = &.{ .{ .top_of_pipe_bit = true }, .{ .top_of_pipe_bit = true } },
                .p_next = &timeline,
            },
        }, curr_pf.fence_work_finished);

        try ctx.sc.present(gq, curr_pf.sem_ready_present);
    }

    dev.deviceWaitIdle() catch unreachable;
}
