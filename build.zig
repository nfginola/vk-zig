const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "vk-zig",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Expose mach-glfw as module for exe
    const glfw_dep = b.dependency("mach-glfw", .{
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("mach-glfw", glfw_dep.module("mach-glfw"));

    // Expose vulkan bindings
    const vulkan = b.dependency("vulkan_zig", .{
        .registry = b.path("ext/vk-1.3.296.0.xml"),
    }).module("vulkan-zig");
    exe.root_module.addImport("vulkan", vulkan);

    // Expose image reader (zigimg)
    const zigimg_dependency = b.dependency("zigimg", .{
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("zigimg", zigimg_dependency.module("zigimg"));

    // Add Vulkan options
    const vk_opts = b.addOptions();
    vk_opts.addOption(bool, "validation_layer", b.option(bool, "vl", "Vulkan Validation Layer") orelse true); // On by default
    exe.root_module.addOptions("VK_CONF", vk_opts);

    // Before final install TLS, compile shaders
    {
        const spath = "res/shaders";
        const output_ext = ".spv";

        var arena = std.heap.ArenaAllocator.init(b.allocator);
        defer arena.deinit();

        var sdir = try std.fs.cwd().openDir(spath, .{ .iterate = true });
        defer sdir.close();
        var sdir_it = sdir.iterate();

        while (try sdir_it.next()) |entry| {
            if (entry.kind == .file) {
                const ext = std.fs.path.extension(entry.name);
                const just_name = entry.name[0..(std.mem.lastIndexOf(u8, entry.name, ".") orelse 0)];
                if (std.mem.eql(u8, ext, output_ext))
                    continue;

                const in_sname = try std.mem.concat(arena.allocator(), u8, &.{ spath, "/", entry.name });
                const out_sname = try std.mem.concat(arena.allocator(), u8, &.{ spath, "/", "compiled/", just_name, output_ext });

                // One glslc invocation per shader
                const compile_shaders = b.addSystemCommand(&.{"glslc"});
                compile_shaders.addArgs(&.{
                    in_sname,
                    "-o",
                    out_sname,
                });

                b.getInstallStep().dependOn(&compile_shaders.step);
                _ = arena.reset(.retain_capacity);
            }
        }
    }

    b.installArtifact(exe);

    // Add convenience 'run' step
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
