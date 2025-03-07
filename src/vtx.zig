const bconf = @import("VK_CONF"); // Build configs
const std = @import("std");
const Allocator = std.mem.Allocator;
const glfw = @import("mach-glfw");
const vk = @import("vulkan");
const vkt = @import("vk_types.zig");
const vkb = @import("vk_base.zig");
const memh = @import("memory_helpers.zig");
const zgui = @import("zgui");
const vkg = @import("vk_gui.zig");

const Self = @This();

// Graphics Arena Allocator
pub const Arena = memh.CallbackArena(Self);

pub const InitOptions = struct {
    name: [*:0]const u8 = "Default Name",
    window: glfw.Window,
};

// Per-context
pd: vk.PhysicalDevice,
pd_props: vk.PhysicalDeviceProperties,
mem_props: vk.PhysicalDeviceMemoryProperties,

devd: vkb.DeviceDispatch,
dev: vkt.Device,

// One queue of each type, no practical reason to have more.
queues: [@typeInfo(vkt.QueueType).@"enum".fields.len]vkt.Queue = undefined,

// Holds heap index per memory type
memory_heaps: [@typeInfo(vkt.MemoryType).@"enum".fields.len]u32,

sc: vkt.Swapchain,

var global_dev: vkt.Device = undefined;

pub fn create(
    ator: Allocator,
    opts: InitOptions,
) !*Self {
    const self = try ator.create(Self);

    const get_inst_fn = @as(vk.PfnGetInstanceProcAddr, @ptrCast(&glfw.getInstanceProcAddress));
    const glfw_exts = glfw.getRequiredInstanceExtensions() orelse {
        std.log.err("Failed to get required VK vkb.instance extensions for GLFW. Error = {s}", .{glfw.mustGetError().description});
        return error.MissingGlfwInstanceExtensions;
    };
    try vkb.init(ator, opts.name, get_inst_fn, glfw_exts);

    try self.getPhysicalDevice(ator);
    try self.createDevice(ator);
    try self.sc.native.init(ator, self.pd, self.dev, opts.window);

    try vkg.init(ator, self);

    // const pool = try self.createDescPool(null, 10, &[_]vk.DescriptorPoolSize{
    //     vk.DescriptorPoolSize{
    //         .descriptor_count = 10,
    //         .type = .combined_image_sampler,
    //     },
    // }, .{ .free_descriptor_set_bit = true });
    //
    // const prend: zgui.backend.VkPipelineRenderingCreateInfo = .{
    //     .s_type = @intFromEnum(vk.StructureType.pipeline_rendering_create_info),
    //     .color_attachment_count = 1,
    //     .p_color_attachment_formats = &[_]i32{@intFromEnum(self.sc.getFormat())},
    //     .view_mask = 0,
    //     .depth_attachment_format = 0,
    //     .stencil_attachment_format = 0,
    // };
    //
    // const init: zgui.backend.ImGui_ImplVulkan_InitInfo = .{
    //     .instance = @intFromEnum(vkb.inst.handle),
    //     .physical_device = @intFromEnum(self.pd),
    //     .device = @intFromEnum(self.dev.handle),
    //     .queue_family = self.getQueue(.graphics).fam.?,
    //     .queue = @intFromEnum(self.getQueue(.graphics).api.handle),
    //     .descriptor_pool = @intFromEnum(pool.hdl),
    //     .min_image_count = 2,
    //     .image_count = 2,
    //     .render_pass = 0,
    //     .use_dynamic_rendering = true,
    //     .pipeline_rendering_create_info = prend,
    // };
    //
    // const vk_loader = struct {
    //     pub fn load(function_name: [*:0]const u8, user_data: ?*anyopaque) callconv(.C) ?*anyopaque {
    //         if (user_data == null)
    //             std.debug.assert(false);
    //
    //         const instance: *vk.Instance = @ptrCast(@alignCast(user_data.?));
    //         return @constCast(@ptrCast(get_inst_fn(instance.*, function_name)));
    //     }
    // }.load;
    //
    // zgui.init(ator);
    // zgui.io.setConfigFlags(.{ .dock_enable = true });
    // const style = zgui.getStyle();
    // zgui.Style.setColorsBuiltin(style, .dark);
    //
    // _ = zgui.backend.loadFunctions(vk_loader, &vkb.inst.handle);
    // zgui.backend.init(init, opts.window.handle);

    return self;
}

pub fn deinit(self: *Self) void {
    self.dev.deviceWaitIdle() catch unreachable;
    vkg.deinit();
    self.sc.native.deinit();
    self.dev.destroyDevice(null);
    vkb.deinit();
}

pub fn createArena(self: *Self, ator: Allocator) !*Arena {
    return Arena.create(ator, self);
}

pub fn waitForFences(self: *Self, fences: []const vk.Fence) !void {
    _ = try self.dev.waitForFences(@intCast(fences.len), @ptrCast(fences), vk.TRUE, std.math.maxInt(u64));
}
pub fn resetFences(self: *Self, fences: []const vk.Fence) !void {
    _ = try self.dev.resetFences(@intCast(fences.len), @ptrCast(fences));
}
pub fn waitResetFences(self: *Self, fences: []const vk.Fence) !void {
    try self.waitForFences(fences);
    try self.resetFences(fences);
}

/// Each queue type should be treated as distinct queues from a distinct family.
/// As fallback, if no distinct family can be found, they will use graphics family.
pub fn getQueue(self: *Self, qtype: vkt.QueueType) vkt.Queue {
    return self.queues[@intFromEnum(qtype)];
}

pub fn createShaderModuleFromFile(self: *Self, ator: Allocator, maybe_varena: ?*Arena, fpath: []const u8) !vk.ShaderModule {
    const file = try std.fs.cwd().openFile(fpath, .{});
    defer file.close();

    // We must guarantee that the final binary is 4-byte aligned, that is
    // the requirement for the binary passed to create shader module
    const binary = try file.reader().readAllAlloc(ator, std.math.maxInt(usize));
    const bin_aligned = try ator.alignedAlloc(u8, @alignOf(u32), binary.len);
    @memcpy(bin_aligned, binary);

    return try self.createShaderModule(maybe_varena orelse null, bin_aligned);
}

pub fn createShaderModule(self: *Self, maybe_varena: ?*Arena, binary: []u8) !vk.ShaderModule {
    std.debug.assert(@intFromPtr(binary.ptr) % @sizeOf(u32) == 0);
    const mod = try self.dev.createShaderModule(&.{
        .code_size = binary.len,
        .p_code = @ptrCast(@alignCast(binary.ptr)),
    }, null);

    if (maybe_varena) |varena| {
        try varena.add(@TypeOf(mod), Self.destroyShaderModule, mod);
    }

    return mod;
}

pub fn destroyShaderModule(self: *Self, module: vk.ShaderModule) void {
    self.dev.destroyShaderModule(module, null);
}

pub fn createCmdPool(self: *Self, maybe_varena: ?*Arena, qtype: vkt.QueueType, flags: vk.CommandPoolCreateFlags) !vkt.CommandPool {
    const pool = vkt.CommandPool{
        .devd = self.devd,
        .dev = self.dev,
        .hdl = try self.dev.createCommandPool(&.{ .queue_family_index = self.queues[@intFromEnum(qtype)].fam.?, .flags = flags }, null),
    };

    if (maybe_varena) |varena| {
        try varena.add(@TypeOf(pool), Self.destroyCmdPool, pool);
    }

    return pool;
}

pub fn destroyCmdPool(self: *Self, pool: vkt.CommandPool) void {
    self.dev.destroyCommandPool(pool.hdl, null);
}

pub fn createFence(self: *Self, maybe_varena: ?*Arena, flags: vk.FenceCreateFlags) !vk.Fence {
    const f = try self.dev.createFence(&.{ .flags = flags }, null);

    if (maybe_varena) |varena| {
        try varena.add(@TypeOf(f), Self.destroyFence, f);
    }

    return f;
}

pub fn destroyFence(self: *Self, hdl: vk.Fence) void {
    self.dev.destroyFence(hdl, null);
}

/// Binary semaphore
pub fn createSemaphoreB(self: *Self, maybe_varena: ?*Arena) !vkt.Semaphore {
    const sem = try self.dev.createSemaphore(&.{}, null);

    const ret: vkt.Semaphore = .{
        .hdl = sem,
        .value = 0,
        .timeline = false,
    };

    if (maybe_varena) |varena| {
        try varena.add(@TypeOf(ret), Self.destroySemaphore, ret);
    }

    return ret;
}

pub fn destroySemaphore(self: *Self, sem: vkt.Semaphore) void {
    self.dev.destroySemaphore(sem.hdl, null);
}

pub fn createSemaphore(self: *Self, maybe_varena: ?*Arena) !vkt.Semaphore {
    const timeline_ci = vk.SemaphoreTypeCreateInfo{
        .initial_value = 0,
        .semaphore_type = .timeline,
    };

    const sem = vkt.Semaphore{
        .hdl = try self.dev.createSemaphore(&.{
            .p_next = &timeline_ci,
        }, null),
        .value = 0,
        .timeline = true,
    };

    if (maybe_varena) |varena| {
        try varena.add(@TypeOf(sem), Self.destroySemaphoreT, sem);
    }

    return sem;
}

pub fn destroySemaphoreT(self: *Self, sem: vkt.Semaphore) void {
    self.dev.destroySemaphore(sem.hdl, null);
}

pub fn waitSemaphores(self: *Self, sems: []const vkt.Semaphore) !void {
    var sems_raw: [8]vk.Semaphore = undefined;
    var values: [8]u64 = undefined;
    for (sems, 0..) |sem, i| {
        std.debug.assert(sem.timeline);
        sems_raw[i] = sem.hdl;
        values[i] = sem.value;
    }

    _ = try self.dev.waitSemaphores(
        &.{
            .semaphore_count = @intCast(sems.len),
            .p_semaphores = &sems_raw,
            .p_values = &values,
        },
        std.math.maxInt(u64),
    );
}

pub fn allocateMemory(self: *Self, maybe_varena: ?*Arena, inf: vkt.MemoryAllocateInfo) !vkt.DeviceMemory {
    // RenderDoc modifies this internally and we need to guarantee
    // accessible memory!
    // Check RDoc:vk_resource_funcs.cpp:727
    // When device address bit is set, it internally sets
    // device_address_capture_replay_bit
    var alloc_flags = vk.MemoryAllocateFlagsInfo{
        .flags = .{
            .device_address_bit = true,
        },
        .device_mask = 0, // unused
    };

    const mem = vkt.DeviceMemory{
        .hdl = try self.dev.allocateMemory(&vk.MemoryAllocateInfo{
            .allocation_size = inf.size,
            .memory_type_index = self.memory_heaps[@intFromEnum(inf.type)],
            .p_next = if (inf.device_adr) &alloc_flags else null,
        }, null),
        .dev_addressable = inf.device_adr,
    };

    if (maybe_varena) |varena| {
        try varena.add(@TypeOf(mem), Self.freeMemory, mem);
    }

    return mem;
}

pub fn freeMemory(self: *Self, mem: vkt.DeviceMemory) void {
    self.dev.freeMemory(mem.hdl, null);
}

pub fn createImage(self: *Self, maybe_varena: ?*Arena, inf: vkt.ImageInfo) !vkt.Image {
    const ret = vkt.Image{
        .hdl = try self.dev.createImage(&vk.ImageCreateInfo{
            .image_type = inf.type,
            .format = inf.format,
            .extent = .{ .width = inf.width, .height = inf.height, .depth = inf.depth },
            .mip_levels = inf.mips,
            .array_layers = inf.array_layers,
            .samples = inf.samples,
            .tiling = inf.tiling,
            .usage = vk.ImageUsageFlags.merge(inf.usage, .{ .transfer_dst_bit = true }),
            .sharing_mode = inf.sharing_mode,
            .initial_layout = .undefined,
        }, null),
    };

    if (maybe_varena) |varena| {
        try varena.add(@TypeOf(ret), Self.destroyImage, ret);
    }

    return ret;
}

pub fn createImageWithMemory(self: *Self, maybe_varena: ?*Arena, inf: vkt.ImageInfo) !vkt.Image {
    const mips = if (inf.mips == 0) vkt.Utils.getMipLevels(inf.width, inf.height) else inf.mips;
    const hdl =
        try self.dev.createImage(&vk.ImageCreateInfo{
        .image_type = inf.type,
        .format = inf.format,
        .extent = .{ .width = inf.width, .height = inf.height, .depth = inf.depth },
        .mip_levels = mips,
        .array_layers = inf.array_layers,
        .samples = inf.samples,
        .tiling = inf.tiling,
        .usage = vk.ImageUsageFlags.merge(inf.usage, .{ .transfer_dst_bit = true, .transfer_src_bit = true }),
        .sharing_mode = inf.sharing_mode,
        .initial_layout = .undefined,
    }, null);

    const mem_req = self.dev.getImageMemoryRequirements(hdl);

    // FUTURE: Often the required memory is larger than packed data size, if we
    //         track memory usage, we should track it properly here.
    const mem = try self.allocateMemory(maybe_varena, .{
        .size = mem_req.size,
        .type = .gpu,
    });
    // FUTURE: If suballocating, we need to guarantee that image ends up
    // at an aligned address!
    // std.debug.print("Alignment req image: {}\n", .{mem_req.alignment});
    try self.dev.bindImageMemory(hdl, mem.hdl, 0);

    var ret = vkt.Image{
        .hdl = hdl,
        .memory = mem,
        .mips = mips,
    };

    if (inf.view_type) |vtype| {
        ret.view = try self.dev.createImageView(&vk.ImageViewCreateInfo{
            .image = hdl,
            .view_type = vtype,
            .format = inf.format,
            .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
            .subresource_range = vkt.Utils.fullSubres(.{ .color_bit = true }),
        }, null);
    }

    if (maybe_varena) |varena| {
        try varena.add(@TypeOf(ret), Self.destroyImage, ret);
    }

    return ret;
}

pub fn destroyImage(self: *Self, img: vkt.Image) void {
    self.dev.destroyImage(img.hdl, null);
    if (img.view) |v| {
        self.dev.destroyImageView(v, null);
    }
}

pub fn createBuffer(self: *Self, maybe_varena: ?*Arena, size: u64, usage: vk.BufferUsageFlags) !vkt.Buffer {
    const buf = vkt.Buffer{
        .hdl = try self.dev.createBuffer(&vk.BufferCreateInfo{
            .sharing_mode = .exclusive,
            .size = size,
            .usage = usage,
        }, null),
        .size = size,
    };

    if (maybe_varena) |varena| {
        try varena.add(@TypeOf(buf), Self.destroyBuffer, buf);
    }

    return buf;
}

pub fn destroyBuffer(self: *Self, buffer: vkt.Buffer) void {
    self.dev.destroyBuffer(buffer.hdl, null);
}

/// Shorthand for paired buffer/memory
pub fn createBufferWithMemory(
    self: *Self,
    maybe_varena: ?*Arena,
    inf: vkt.MemoryBufferInfo,
) !vkt.Buffer {
    var buf = try self.createBuffer(maybe_varena, inf.size, inf.usage);
    const mem = try self.allocateMemory(maybe_varena, .{
        .size = inf.size,
        .device_adr = if (inf.usage.shader_device_address_bit == true) true else false,
        .type = inf.mem_type,
    });
    try self.bindBufferMemory(&buf, mem, 0);
    return buf;
}

pub fn bindBufferMemory(self: *Self, buffer: *vkt.Buffer, memory: vkt.DeviceMemory, offset: u32) !void {
    _ = try self.dev.bindBufferMemory(buffer.hdl, memory.hdl, offset);
    buffer.memory = memory;
    buffer.mem_offset = offset;
    if (memory.dev_addressable)
        buffer.gpu_adr = self.dev.getBufferDeviceAddress(&vk.BufferDeviceAddressInfo{
            .buffer = buffer.hdl,
        });
}

pub fn mapBuffer(self: *Self, comptime T: type, buffer: vkt.Buffer) ![]T {
    std.debug.assert(buffer.memory != null);
    if (try self.dev.mapMemory(buffer.memory.?.hdl, buffer.mem_offset.?, buffer.size, .{})) |p| {
        std.debug.assert(@intFromPtr(p) % @sizeOf(T) == 0);
        std.debug.assert(buffer.size % @sizeOf(T) == 0);
        return @as([*]T, @ptrCast(@alignCast(p)))[0..(buffer.size / @sizeOf(T))];
    } else {
        return error.MemoryMapFailed;
    }
}

pub fn createDescPool(
    self: *Self,
    maybe_varena: ?*Arena,
    max_sets: u32,
    pool_descs: []const vk.DescriptorPoolSize,
    flags: vk.DescriptorPoolCreateFlags,
) !vkt.DescriptorPool {
    const hdl = try self.dev.createDescriptorPool(&vk.DescriptorPoolCreateInfo{
        .max_sets = max_sets,
        .pool_size_count = @intCast(pool_descs.len),
        .p_pool_sizes = pool_descs.ptr,
        .flags = flags,
    }, null);

    const dpool: vkt.DescriptorPool = .{ .dev = self.dev, .hdl = hdl };

    if (maybe_varena) |varena| {
        try varena.add(@TypeOf(dpool), Self.destroyDescPool, dpool);
    }

    return dpool;
}

pub fn destroyDescPool(self: *Self, pool: vkt.DescriptorPool) void {
    self.dev.destroyDescriptorPool(pool.hdl, null);
}

pub fn createDescSetLayout(
    self: *Self,
    ator: Allocator,
    maybe_varena: ?*Arena,
    layout_info: vkt.DescriptorSetLayoutInfo,
) !vk.DescriptorSetLayout {
    // Per binding flags are all or nothing for simplicity (they map contiguously to bindings.. no gaps!)
    const flag_on = layout_info.bindings[0].flags != null;
    for (layout_info.bindings) |binding| {
        // Not all bindings have been provided flags. Either provide for all, or for none
        if (flag_on) {
            std.debug.assert(binding.flags != null);
        } else {
            std.debug.assert(binding.flags == null);
        }
    }

    // Unpack bindings and per-binding flags
    var bindings = try ator.alloc(vk.DescriptorSetLayoutBinding, layout_info.bindings.len);
    var binding_flags = try ator.alloc(vk.DescriptorBindingFlags, layout_info.bindings.len);
    for (layout_info.bindings, 0..) |binding, i| {
        bindings[i] = binding.binding;
        binding_flags[i] = binding.flags orelse .{};
    }

    const ddi_ci = vk.DescriptorSetLayoutBindingFlagsCreateInfo{
        .binding_count = @intCast(binding_flags.len),
        .p_binding_flags = binding_flags.ptr,
    };

    const dlayout = try self.dev.createDescriptorSetLayout(&vk.DescriptorSetLayoutCreateInfo{
        .binding_count = @intCast(bindings.len),
        .p_bindings = bindings.ptr,
        .flags = layout_info.flags,
        .p_next = &ddi_ci,
    }, null);

    if (maybe_varena) |varena| {
        try varena.add(@TypeOf(dlayout), Self.destroyDescSetLayout, dlayout);
    }

    return dlayout;
}

pub fn destroyDescSetLayout(self: *Self, layout: vk.DescriptorSetLayout) void {
    self.dev.destroyDescriptorSetLayout(layout, null);
}

pub fn createGraphicsPipeline(self: *Self, maybe_varena: ?*Arena, inf: vkt.GraphicsPipelineInfo) !vk.Pipeline {
    var pinfo = vkt.Utils.base_pipe;
    pinfo.layout = inf.layout;

    // Unpack
    var shaders: [5]vk.PipelineShaderStageCreateInfo = undefined;
    for (inf.shaders, 0..) |shader, i| {
        shaders[i] = vk.PipelineShaderStageCreateInfo{
            .stage = shader.stage,
            .p_name = shader.name,
            .module = shader.module,
        };
    }
    pinfo.stage_count = @intCast(inf.shaders.len);
    pinfo.p_stages = @ptrCast(&shaders);

    var output = vk.PipelineRenderingCreateInfoKHR{
        .color_attachment_count = @intCast(inf.output.colors.len),
        .p_color_attachment_formats = @ptrCast(inf.output.colors.ptr),
        .view_mask = 0,
        .depth_attachment_format = inf.output.depth,
        .stencil_attachment_format = inf.output.stencil,
    };
    pinfo.p_next = &output;

    var pipe: vk.Pipeline = undefined;
    _ = try self.dev.createGraphicsPipelines(.null_handle, 1, &.{pinfo}, null, @ptrCast(&pipe));

    if (maybe_varena) |varena| {
        try varena.add(@TypeOf(pipe), Self.destroyPipeline, pipe);
    }

    return pipe;
}

pub fn destroyPipeline(self: *Self, hdl: vk.Pipeline) void {
    self.dev.destroyPipeline(hdl, null);
}

fn getPhysicalDevice(self: *Self, ator: Allocator) !void {
    const pds = try vkb.inst.enumeratePhysicalDevicesAlloc(ator);
    for (pds) |pd| {
        const props = vkb.inst.getPhysicalDeviceProperties(pd);
        if (props.device_type == .discrete_gpu) {
            self.pd = pd;
            self.pd_props = props;
            break;
        }
    }

    // Get relevant memory heaps
    self.mem_props = vkb.inst.getPhysicalDeviceMemoryProperties(self.pd);

    // NOTE: Assuming that device always has coherent memory so that we
    // dont need manual cache flushing/invalidation on app side for
    // CPU-GPU or GPU-CPU
    //
    // TODO MISC:
    // We could always have a helper 'sync_write()' and 'sync_read()'
    // to flush/invalidate if needed, and do so based on MemoryType and
    // whether it supported coherency or not (early out if memory has coherency)
    //
    for (0..self.mem_props.memory_type_count) |i| {
        const mem = self.mem_props.memory_types[i];
        const flags = mem.property_flags;
        if (flags.device_local_bit == true) {
            self.memory_heaps[@intFromEnum(vkt.MemoryType.gpu)] = @intCast(i);
            std.debug.print("Found GPU memory! (device local) ({})\n", .{i});
            break;
        }
    }

    for (0..self.mem_props.memory_type_count) |i| {
        const mem = self.mem_props.memory_types[i];
        const flags = mem.property_flags;
        if (flags.device_local_bit == true and flags.host_visible_bit == true and flags.host_coherent_bit == true) {
            self.memory_heaps[@intFromEnum(vkt.MemoryType.cpu_to_gpu)] = @intCast(i);
            std.debug.print("Found host-visible and host-coherent GPU memory! (staging) ({})\n", .{i});
        }
    }

    for (0..self.mem_props.memory_type_count) |i| {
        const mem = self.mem_props.memory_types[i];
        const flags = mem.property_flags;
        if (flags.host_visible_bit == true and flags.host_cached_bit == true and flags.host_coherent_bit == true) {
            self.memory_heaps[@intFromEnum(vkt.MemoryType.gpu_to_cpu)] = @intCast(i);
            std.debug.print("Found host-visible, host-cached and host-coherent GPU memory! (readback) ({})\n", .{i});
        }
    }

    std.debug.print("Using physical device: {s}\n", .{self.pd_props.device_name});
}

fn createDevice(self: *Self, ator: Allocator) !void {
    var arena = memh.Arena.init(ator);
    defer arena.deinit();

    self.queues = .{.{}} ** @typeInfo(vkt.QueueType).@"enum".fields.len;

    // Ensure unique set of queue families
    var set = std.AutoHashMap(u32, u32).init(arena.ator()); // (qf_idx, num_queues)

    // Find suitable queue families
    // Use distinct queue family for each (Graphics, Compute, Transfer)
    const qf_props = try vkb.inst.getPhysicalDeviceQueueFamilyPropertiesAlloc(self.pd, arena.ator());
    for (qf_props, 0..) |qf, qf_idx| {
        if (set.contains(@intCast(qf_idx)))
            continue;

        const gfx = &self.queues[@intFromEnum(vkt.QueueType.graphics)];
        const compute = &self.queues[@intFromEnum(vkt.QueueType.compute)];
        const transfer = &self.queues[@intFromEnum(vkt.QueueType.transfer)];
        const tr_graphics = &self.queues[@intFromEnum(vkt.QueueType.transfer_graphics)];
        const tr_compute = &self.queues[@intFromEnum(vkt.QueueType.transfer_compute)];

        if (gfx.fam == null and qf.queue_flags.graphics_bit == true) {
            gfx.*.fam = @intCast(qf_idx);
            gfx.*.id = 0;
            tr_graphics.*.fam = @intCast(qf_idx);
            tr_graphics.*.id = 1;
            std.debug.print("Found distinct queue family ({}) Graphics: {any}\n", .{ qf_idx, qf.queue_flags });
            try set.put(@intCast(qf_idx), 2);
        } else if (compute.fam == null and qf.queue_flags.compute_bit == true) {
            compute.*.fam = @intCast(qf_idx);
            compute.*.id = 0;
            tr_compute.*.fam = @intCast(qf_idx);
            tr_compute.*.id = 1;
            std.debug.print("Found distinct queue family ({}) Compute: {any}\n", .{ qf_idx, qf.queue_flags });
            try set.put(@intCast(qf_idx), 2);
        } else if (transfer.fam == null and qf.queue_flags.transfer_bit == true) {
            transfer.*.fam = @intCast(qf_idx);
            std.debug.print("Found distinct queue family ({}) Transfer: {any}\n", .{ qf_idx, qf.queue_flags });
            try set.put(@intCast(qf_idx), 1);
        }
    }

    // If distinct queue family was not found, use graphics family as fallback
    // and create a distinct queue, preserving any potential async nature by
    // calling operations
    const gfx_fam = self.queues[@intFromEnum(vkt.QueueType.graphics)].fam.?;
    for (&self.queues, 0..) |*q, i| {
        const kind: vkt.QueueType = @enumFromInt(i);
        if (q.fam == null) {
            q.fam = gfx_fam;
            std.debug.print("Distinct queue family was not found for {}, using family ({}) as fallback\n", .{ kind, q.fam.? });
            const num_queues = set.getPtr(gfx_fam).?;
            q.id = num_queues.*;
            num_queues.* += 1;
        }
    }

    // Setup queue create infos per unique queue family
    var q_cinfos = std.ArrayList(vk.DeviceQueueCreateInfo).init(arena.ator());
    var it = set.iterator();
    while (it.next()) |entry| {
        try q_cinfos.append(.{
            .queue_count = entry.value_ptr.*,
            .queue_family_index = entry.key_ptr.*,
            .p_queue_priorities = &.{1.0},
        });
    }

    // Query descriptor indexing features:
    // Pass DDI feats with proper sType to PhysDev pNext and it gets filled for us.
    // Assume pNext is matched with all possible available sType (for this query)
    // and that it recursively checks any following pNext inside that structure (linked list of pNext)
    var ddi_feats: vk.PhysicalDeviceDescriptorIndexingFeatures = .{};

    // Buffer device address
    var bda_feats: vk.PhysicalDeviceBufferDeviceAddressFeatures = .{};
    bda_feats.buffer_device_address = vk.TRUE;
    bda_feats.buffer_device_address_capture_replay = vk.TRUE;
    ddi_feats.p_next = &bda_feats; // Chain pNext

    var time_sem = vk.PhysicalDeviceTimelineSemaphoreFeatures{};
    bda_feats.p_next = &time_sem;

    var sync2 = vk.PhysicalDeviceSynchronization2Features{};
    sync2.synchronization_2 = vk.TRUE;
    time_sem.p_next = &sync2;

    var feats: vk.PhysicalDeviceFeatures2 = .{ .p_next = &ddi_feats, .features = undefined };

    // Turn on/off features available in the physical device
    vkb.inst.getPhysicalDeviceFeatures2(self.pd, &feats);

    // TODO: Just assume all supported and pass on to device creation
    //       Return no error for now if something missing
    // if (ddi_feats.descriptor_binding_uniform_buffer_update_after_bind == vk.TRUE) {
    //     std.debug.print("yay\n", .{});
    // } else {
    //     std.debug.print("nay\n", .{});
    // }

    const exts = &[_][*:0]const u8{
        vk.extensions.khr_swapchain.name,
        vk.extensions.khr_dynamic_rendering.name,
        vk.extensions.ext_descriptor_indexing.name,
        // Works around a validation layer bug with descriptor pool allocation with VARIABLE_COUNT.
        // See: https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/2350.
        vk.extensions.khr_maintenance_1.name,
        vk.extensions.khr_buffer_device_address.name,
        vk.extensions.khr_synchronization_2.name, // QueueSubmit2
    };

    // Queue capabilities are dependent on queue family identified by index
    const dev = try vkb.inst.createDevice(self.pd, &vk.DeviceCreateInfo{
        .queue_create_info_count = @intCast(q_cinfos.items.len),
        .p_queue_create_infos = @ptrCast(q_cinfos.items),
        .enabled_extension_count = exts.len,
        .pp_enabled_extension_names = exts,
        .p_enabled_features = &feats.features,
        // Enable dynamic rendering
        // Enable DDI (additional)
        .p_next = &vk.PhysicalDeviceDynamicRenderingFeaturesKHR{ .dynamic_rendering = vk.TRUE, .p_next = &ddi_feats },
    }, null);

    self.devd = try vkb.DeviceDispatch.load(dev, vkb.vki.dispatch.vkGetDeviceProcAddr);
    self.dev = vkt.Device.init(dev, &self.devd);

    // Retrieve queues. Queue proxy needs Device dispatch table because
    // the QueueSubmit and similar functions are loaded into the table there
    for (&self.queues) |*q| {
        q.api = vkb.Queue.init(self.dev.getDeviceQueue(q.fam.?, q.id), &self.devd);
    }
}
