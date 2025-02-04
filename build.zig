const std = @import("std");
const builtin = @import("builtin");

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
        .registry = b.path("ext/vk/vk-1.3.296.0.xml"),
    }).module("vulkan-zig");
    exe.root_module.addImport("vulkan", vulkan);

    // Expose image reader (zigimg)
    const zigimg_dependency = b.dependency("zigimg", .{
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("zigimg", zigimg_dependency.module("zigimg"));

    // Zgui (static lib)
    const zgui = b.dependency("zgui", .{
        .shared = false,
        .backend = .glfw_vulkan,
    });
    exe.root_module.addImport("zgui", zgui.module("root"));
    exe.linkLibrary(zgui.artifact("imgui"));

    // Ztracy
    const ztracy = b.dependency("ztracy", .{
        .enable_ztracy = true,
        .enable_fibers = true,
        .on_demand = true,
    });
    exe.root_module.addImport("ztracy", ztracy.module("root"));
    exe.linkLibrary(ztracy.artifact("tracy"));

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

    // Add convenience step for building tracy server
    try build_tracy_server(b, "btr", "Build tracy server for v0.11.1");

    // Add convenience 'run' step
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}

pub fn build_tracy_server(b: *std.Build, name: []const u8, description: []const u8) !void {
    const submodule_init = b.addSystemCommand(&.{"git"});
    submodule_init.addArgs(&.{
        "submodule",
        "update",
        "--init",
        "--recursive",
    });

    // cmake -B profiler/build -S profiler -DCMAKE_BUILD_TYPE=Release -DLEGACY=ON -DTBB_STRICT=OFF
    const setup = b.addSystemCommand(&.{"cmake"});
    setup.addArgs(&.{
        "-B",
        "ext/tracy/profiler/build",
        "-S",
        "ext/tracy/profiler",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLEGACY=ON",
        "-DTBB_STRICT=OFF",
    });

    //cmake --build profiler/build --config Release --parallel
    const compile = b.addSystemCommand(&.{"cmake"});
    compile.addArgs(&.{
        "--build",
        "ext/tracy/profiler/build",
        "--config",
        "Release",
        "--parallel",
    });

    const copy_server = b.addSystemCommand(&.{"cp"});
    copy_server.addArgs(&.{
        "ext/tracy/profiler/build/tracy-profiler",
        b.exe_dir,
    });

    setup.step.dependOn(&submodule_init.step);
    compile.step.dependOn(&setup.step);
    copy_server.step.dependOn(&compile.step);

    const build_cmd = b.step(name, description);
    build_cmd.dependOn(&copy_server.step);

    // Add convenience function to run tracy using 'zig build tr'
    {
        var arena = std.heap.ArenaAllocator.init(b.allocator);
        defer arena.deinit();

        const bin = try std.mem.concat(arena.allocator(), u8, &.{ b.exe_dir, "/", "tracy-profiler" });
        const run_tracy = b.addSystemCommand(&.{bin});
        const run_with_tracy = b.step("tr", "Run the tracy server");
        run_with_tracy.dependOn(&run_tracy.step);
    }

    // Add super convenience function to run both
    {
        const unix = builtin.os.tag == .linux or builtin.os.tag == .macos;
        // Use & to run the first command in the background, allowing the second to run.
        // Otherwise everything here is sequential
        const args = if (unix) &.{ "sh", "-c", "zig build run & zig build tr" } else &.{ "/C", "zig build run & zig build tr" };

        const run_both_cmd = b.addSystemCommand(args);

        const run_both = b.step("all", "Run the app and tracy server");
        run_both.dependOn(&run_both_cmd.step);
        run_both.dependOn(b.getInstallStep());
    }
}
