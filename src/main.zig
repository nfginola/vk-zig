const std = @import("std");
const glfw = @import("mach-glfw");
const zimg = @import("zigimg");
const nvk = @import("vtx.zig");
const vk = @import("vulkan");
const memh = @import("memory_helpers.zig");
const utx = @import("vk_upload_context.zig");

const WIDTH = 1600;
const HEIGHT = 900;

/// Default GLFW error handling callback
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

    var dstack = memh.Dstack.init(gpa.allocator());
    defer dstack.deinit();

    var ctx = try nvk.Context.create(dstack.ator(), .{ .name = "Vulkan Engine", .window = window });
    defer ctx.deinit();

    var upload = try utx.UploadContext.create(dstack.ator(), ctx);
    defer upload.deinit();

    const dev = ctx.dev;
    const gq = ctx.getQueue(.graphics);
    var cmdp = try ctx.createCmdPool(.graphics, .{ .transient_bit = true });
    const clear_finished_sem = ctx.createSemaphore();
    const sc_img_acquired_sem = ctx.createSemaphore();
    const sc_img_acquired_fence = ctx.createFence(.{ .signaled_bit = true });
    const cmdb_finished_fence = ctx.createFence(.{ .signaled_bit = true });

    defer ctx.destroyFence(cmdb_finished_fence);
    defer ctx.destroyFence(sc_img_acquired_fence);
    defer ctx.destroySemaphore(sc_img_acquired_sem);
    defer ctx.destroySemaphore(clear_finished_sem);
    defer ctx.destroyCmdPool(cmdp);

    // TODO:
    //  - Grab glslc and compile shaders.
    //  - Make pipeline for triangle :skull:
    //

    // Grab buffer memory
    const vmem = try ctx.allocateMemory(.gpu, 64_000);
    const imem = try ctx.allocateMemory(.gpu, 64_000);
    defer ctx.freeMemory(vmem);
    defer ctx.freeMemory(imem);
    const vb = try ctx.createBuffer(64_000, .{ .vertex_buffer_bit = true, .transfer_dst_bit = true });
    const ib = try ctx.createBuffer(64_000, .{ .index_buffer_bit = true, .transfer_dst_bit = true });
    defer ctx.destroyBuffer(vb);
    defer ctx.destroyBuffer(ib);
    _ = try ctx.dev.bindBufferMemory(vb, vmem, 0);
    _ = try ctx.dev.bindBufferMemory(ib, imem, 0);

    {
        // Upload tri and index
        const Vertex = struct {
            x: f32,
            y: f32,
            z: f32,
            u: f32,
            v: f32,
        };
        const vertices = [_]Vertex{
            .{ .x = -0.5, .y = -0.5, .z = 0.5, .u = 0.0, .v = 0.0 },
            .{ .x = 0.5, .y = -0.5, .z = 0.5, .u = 0.0, .v = 1.0 },
            .{ .x = 0.0, .y = 0.5, .z = 0.5, .u = 1.0, .v = 1.0 },
        };
        const indices = [_]u32{ 0, 1, 2 };
        const r1 = try upload.push(memh.byteSliceC(Vertex, vertices[0..]), 4);
        const r2 = try upload.push(memh.byteSliceC(u32, indices[0..]), 4);

        const Payload = struct {
            dst_vb: vk.Buffer,
            dst_ib: vk.Buffer,
            vb_r: utx.Receipt,
            ib_r: utx.Receipt,
        };
        const my_payload = Payload{ .dst_vb = vb, .dst_ib = ib, .vb_r = r1, .ib_r = r2 };
        try upload.add_work(Payload, my_payload, struct {
            pub fn work(cmdb: nvk.CommandBuffer, src: vk.Buffer, payload: *const Payload) void {
                cmdb.copyBuffer(src, payload.dst_vb, 1, &.{vk.BufferCopy{ .dst_offset = 0, .src_offset = payload.vb_r.start, .size = payload.vb_r.size }});
                cmdb.copyBuffer(src, payload.dst_ib, 1, &.{vk.BufferCopy{ .dst_offset = 0, .src_offset = payload.ib_r.start, .size = payload.ib_r.size }});
            }
        }.work);
        try upload.host_wait();
    }

    while (!window.shouldClose()) {
        if (window.getKey(glfw.Key.escape) == glfw.Action.press) {
            break;
        }

        _ = try dev.waitForFences(2, &.{ sc_img_acquired_fence, cmdb_finished_fence }, vk.TRUE, std.math.maxInt(u64));
        try dev.resetFences(2, &.{ sc_img_acquired_fence, cmdb_finished_fence });
        const sc_next = try ctx.sc.getNext(sc_img_acquired_sem, sc_img_acquired_fence);

        try cmdp.reset(.{});
        var cmdb = try cmdp.alloc(.primary, 1);
        try cmdb.beginCommandBuffer(&.{ .flags = .{ .one_time_submit_bit = true } });

        cmdb.pipelineBarrier(.{ .bottom_of_pipe_bit = true }, .{ .color_attachment_output_bit = true }, .{ .by_region_bit = true }, 0, null, 0, null, 1, &.{
            vk.ImageMemoryBarrier{
                .old_layout = .undefined,
                .new_layout = .color_attachment_optimal,
                .src_access_mask = .{},
                .dst_access_mask = .{ .color_attachment_write_bit = true },
                .image = sc_next.image,
                .subresource_range = nvk.Def.full_subres(.{ .color_bit = true }),
                .src_queue_family_index = gq.fam,
                .dst_queue_family_index = gq.fam,
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
        cmdb.endRenderingKHR();

        cmdb.pipelineBarrier(.{ .color_attachment_output_bit = true }, .{ .top_of_pipe_bit = true }, .{ .by_region_bit = true }, 0, null, 0, null, 1, &.{
            vk.ImageMemoryBarrier{
                .old_layout = .color_attachment_optimal,
                .new_layout = .present_src_khr,
                .src_access_mask = .{ .color_attachment_write_bit = true },
                .dst_access_mask = .{},
                .image = sc_next.image,
                .subresource_range = nvk.Def.full_subres(.{ .color_bit = true }),
                .src_queue_family_index = gq.fam,
                .dst_queue_family_index = gq.fam,
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
