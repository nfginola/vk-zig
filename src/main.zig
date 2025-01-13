const std = @import("std");
const glfw = @import("mach-glfw");
const zimg = @import("zigimg");
const nvk = @import("vtx.zig");
const vk = @import("vulkan");
const memh = @import("memory_helpers.zig");

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

    const dev = ctx.dev;
    const gq = ctx.getQueue(.graphics);
    var cmdp = try ctx.createCmdPool(.{
        .queue_family_index = gq.fam,
        .flags = .{ .transient_bit = true },
    });

    const clear_finished_sem = ctx.createSemaphore();
    const sc_img_acquired_sem = ctx.createSemaphore();
    const sc_img_acquired_fence = ctx.createFence(.{ .signaled_bit = true });
    const cmdb_finished_fence = ctx.createFence(.{ .signaled_bit = true });

    defer ctx.destroyFence(cmdb_finished_fence);
    defer ctx.destroyFence(sc_img_acquired_fence);
    defer ctx.destroySemaphore(sc_img_acquired_sem);
    defer ctx.destroySemaphore(clear_finished_sem);
    defer ctx.destroyCmdPool(cmdp);

    // Wait for idle before any resource destruction
    defer dev.deviceWaitIdle() catch unreachable;

    //
    // TODO:
    //  - Enable dynamic rendering and use it
    //      - Make a renderpass that clears color with no Pipeline
    //
    //  - Make an upload context using transfer queue
    //      - Prep for VB/IB setup
    //
    //  - Grab glslc and compile shaders.
    //  - Make pipeline for triangle :skull:
    //
    //

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

        cmdb.pipelineBarrier(.{ .bottom_of_pipe_bit = true }, .{ .transfer_bit = true }, .{ .by_region_bit = true }, 0, null, 0, null, 1, &.{
            vk.ImageMemoryBarrier{
                .old_layout = .undefined,
                .new_layout = .transfer_dst_optimal,
                .src_access_mask = .{},
                .dst_access_mask = .{ .transfer_write_bit = true },
                .image = sc_next.image,
                .subresource_range = .{ .aspect_mask = .{ .color_bit = true }, .base_array_layer = 0, .base_mip_level = 0, .layer_count = 1, .level_count = 1 },
                .src_queue_family_index = gq.fam,
                .dst_queue_family_index = gq.fam,
            },
        });

        const clear = vk.ClearColorValue{ .float_32 = .{ 0.3, 0.0, 1.0, 1.0 } };
        cmdb.clearColorImage(sc_next.image, .transfer_dst_optimal, &clear, 1, &.{.{ .aspect_mask = .{ .color_bit = true }, .base_array_layer = 0, .base_mip_level = 0, .layer_count = 1, .level_count = 1 }});

        cmdb.pipelineBarrier(.{ .transfer_bit = true }, .{ .top_of_pipe_bit = true }, .{ .by_region_bit = true }, 0, null, 0, null, 1, &.{
            vk.ImageMemoryBarrier{
                .old_layout = .transfer_dst_optimal,
                .new_layout = .present_src_khr,
                .src_access_mask = .{ .transfer_write_bit = true },
                .dst_access_mask = .{},
                .image = sc_next.image,
                .subresource_range = .{ .aspect_mask = .{ .color_bit = true }, .base_array_layer = 0, .base_mip_level = 0, .layer_count = 1, .level_count = 1 },
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
}
