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
    var gq = ctx.getQueue(.graphics);
    defer ctx.deinit();

    var upload = try utx.create(arena.ator(), ctx, 256_000);
    defer upload.deinit();

    // Top level VK resources lifetime arena
    var varena = try ctx.createArena(arena.ator());
    defer varena.deinit();

    const Vertex = packed struct {
        x: f32,
        y: f32,
        z: f32,
        pad_1: f32 = 0.0,
        u: f32 = 0.0,
        v: f32 = 0.0,
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
    const vertices = [_]Vertex{ // ndc is (-1,-1) top left
        .{ .x = 0.0, .y = -0.5, .z = 0.0, .u = 0.5, .v = 0.0 }, // uv space is (0,0) top left
        .{ .x = -0.5, .y = 0.5, .z = 0.0, .u = 0.0, .v = 1.0 },
        .{ .x = 0.5, .y = 0.5, .z = 0.0, .u = 1.0, .v = 1.0 },
    };
    const indices = [_]u32{ 0, 1, 2 };
    try upload.copy_to_buffer(vb, try upload.push(memh.byteSliceC(Vertex, vertices[0..]), 0));
    try upload.copy_to_buffer(ib, try upload.push(memh.byteSliceC(u32, indices[0..]), 0));

    var image = try zimg.Image.fromFilePath(arena.ator(), "res/textures/vulkan.png");
    std.debug.assert(image.pixelFormat() == .rgba32);
    const img = try ctx.createImageWithMemory(varena, vkt.ImageInfo{
        .view_type = .@"2d",
        .type = .@"2d",
        .width = @intCast(image.width),
        .height = @intCast(image.height),
        .format = .r8g8b8a8_srgb,
        .usage = .{ .sampled_bit = true },
    });
    const img_rec = try upload.grab(image.imageByteSize(), 0);
    @memcpy(img_rec.memory.?, image.rawBytes());
    try upload.copy_to_image(img, .{
        .extent = .{
            .width = @intCast(image.width),
            .height = @intCast(image.height),
            .depth = 1,
        },
        .layout = .shader_read_only_optimal,
    }, img_rec);

    _ = try upload.submit(.graphics);
    try upload.host_wait();

    // Create sampler
    const samp = try ctx.dev.createSampler(&vk.SamplerCreateInfo{
        .min_filter = .linear,
        .mag_filter = .linear,
        .address_mode_u = .repeat,
        .address_mode_v = .repeat,
        .address_mode_w = .repeat,
        .anisotropy_enable = vk.TRUE,
        .max_anisotropy = 16.0,
        .border_color = vk.BorderColor.float_opaque_black,
        .unnormalized_coordinates = vk.FALSE,
        .compare_enable = vk.FALSE,
        .compare_op = .always,
        .mipmap_mode = .linear,
        .mip_lod_bias = 0.0,
        .min_lod = 0.0,
        .max_lod = vk.LOD_CLAMP_NONE,
    }, null);
    defer ctx.dev.destroySampler(samp, null);

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
    //  x Swapchain recreation on resize
    //
    //  x Load and read texture
    //
    //  x Generate mips helper
    //
    //  x Make Gfx pipeline helpers
    //
    //  - Add ImGUI
    //
    //  - Add tracy
    //
    //  - Make math helpers
    //      - Vector2,3,4
    //          - vadd, vsub, vdivFac, vmulFac, dot, cross
    //      - Mat2,3,4
    //          - store column-major for better grounding to math conventions
    //          - mat2 x v2
    //          - mat3 x v3
    //          - mat4 x v4
    //      - Use Right Handed coordinate system with Z up and Y into screen
    //
    //  - Add VMA support
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
            // Binding with variable descriptor count must be the highets binding within that set!
            vkt.DescriptorSetLayoutBinding{
                .binding = .{
                    .binding = 0,
                    .descriptor_count = 1,
                    .descriptor_type = .sampler,
                    .stage_flags = .{ .fragment_bit = true },
                    .p_immutable_samplers = &.{samp},
                },
                .flags = .{},
            },
            vkt.DescriptorSetLayoutBinding{
                .binding = .{
                    .binding = 1,
                    .descriptor_count = 1000,
                    .descriptor_type = .sampled_image,
                    .stage_flags = .{ .fragment_bit = true },
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
            .descriptor_count = 10,
            .type = .sampler,
        },
        .{
            .descriptor_count = 10_000,
            .type = .sampled_image,
        },
    }, .{ .update_after_bind_bit = true }); // DDI
    var dset = try dpool.alloc(.{
        .layout = dlayout,
        .variable_descriptors = 1000, // DDI
    });
    dset.writeImage(.{
        .dst_binding = 1,
        .dst_array_el = 0,
        .dst_type = .sampled_image,
        .layout = .shader_read_only_optimal,
        .view = img.view.?,
    });

    // Need packed to preserve memory ordering
    // 8 byte alignment of u64 causes gap..
    // sort by largest to smallest top-down to avoid having to pad
    const PushConstant = packed struct {
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

    const pipe = try ctx.createGraphicsPipeline(varena, .{
        .layout = p_layout,
        .shaders = &[_]vkt.ShaderInfo{
            vkt.ShaderInfo{
                .module = try ctx.createShaderModuleFromFile(arena.ator(), varena, "res/shaders/compiled/tri.spv"),
                .stage = .{ .vertex_bit = true },
            },
            vkt.ShaderInfo{
                .module = try ctx.createShaderModuleFromFile(arena.ator(), varena, "res/shaders/compiled/pass.spv"),
                .stage = .{ .fragment_bit = true },
            },
        },
        .output = .{
            .colors = &[_]vk.Format{
                ctx.sc.native.format.format,
            },
        },
    });

    const PerFrame = struct {
        // cpu
        farena: memh.Arena, // frame arena

        // gpu
        cmdp: vkt.CommandPool,
        sem_img_acq: vkt.Semaphore,
        sem_ready_present: vkt.Semaphore,
        sem_work_finished: vkt.Semaphore,
    };

    var pf: [MAX_FIF]PerFrame = undefined;
    for (&pf) |*f| {
        f.farena = memh.Arena.init(arena.ator());

        f.cmdp = try ctx.createCmdPool(varena, .graphics, .{ .transient_bit = true });
        f.sem_img_acq = try ctx.createSemaphoreB(varena);
        f.sem_ready_present = try ctx.createSemaphoreB(varena);
        f.sem_work_finished = try ctx.createSemaphore(varena);
    }

    var prev_pf: PerFrame = pf[MAX_FIF - 1];
    var curr_f: u32 = 0;

    var dt: f32 = 0.0; // in s
    const init_disp_interval = 0.01;
    var display_dt_interval: f32 = init_disp_interval; // in s
    var elapsed: f32 = 0.0; // in s

    while (!window.shouldClose()) {
        glfw.pollEvents();

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

        try ctx.waitSemaphores(&[_]vkt.Semaphore{curr_pf.sem_work_finished});

        const sc_next = try ctx.sc.getNext(curr_pf.sem_img_acq.hdl, null) orelse continue;

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
        cmdb.setViewportWithCount(1, @ptrCast(&vk.Viewport{
            .x = 0,
            .y = 0,
            .width = @floatFromInt(ctx.sc.getExtent().width),
            .height = @floatFromInt(ctx.sc.getExtent().height),
            .min_depth = 0.0,
            .max_depth = 1.0,
        }));
        cmdb.setScissorWithCount(1, @ptrCast(&vk.Rect2D{
            .offset = .{
                .x = 0,
                .y = 0,
            },
            .extent = ctx.sc.getExtent(),
        }));

        cmdb.bindPipeline(.graphics, pipe);
        const pc: PushConstant = .{
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

        try gq.submit(.{
            .cmdbs = &[_]vk.CommandBuffer{cmdb.handle},
            .waits = &[_]*vkt.Semaphore{ &curr_pf.sem_img_acq, &prev_pf.sem_work_finished },
            .signals = &[_]*vkt.Semaphore{ &curr_pf.sem_ready_present, &curr_pf.sem_work_finished },
        });

        try ctx.sc.present(gq, curr_pf.sem_ready_present.hdl);
    }

    ctx.dev.deviceWaitIdle() catch unreachable;
}
