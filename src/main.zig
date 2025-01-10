const std = @import("std");
const glfw = @import("mach-glfw");
const zimg = @import("zigimg");
const vk = @import("vtx.zig");

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
    const ator = gpa.allocator();
    defer _ = gpa.deinit();

    var ctx = try vk.Context.init(ator, .{ .name = "Vulkan Engine", .window = window });
    defer ctx.deinit();

    while (!window.shouldClose()) {
        if (window.getKey(glfw.Key.escape) == glfw.Action.press) {
            break;
        }

        //window.swapBuffers();
        glfw.pollEvents();
    }
}
