const std = @import("std");
const glfw = @import("mach-glfw");
const vk = @import("vulkan");
const Allocator = std.mem.Allocator;
const memh = @import("memory_helpers.zig");
const vkb = @import("vk_base.zig");

pub const Swapchain = struct {
    const Self = @This();
    pub const Next = struct {
        idx: u32,
        image: vk.Image,
        view: vk.ImageView,
    };

    ator: Allocator,
    pd: vk.PhysicalDevice,
    win: glfw.Window,

    surf: vk.SurfaceKHR,
    hdl: vk.SwapchainKHR,
    format: vk.SurfaceFormatKHR,
    images: std.ArrayList(vk.Image),
    views: std.ArrayList(vk.ImageView),
    dev: vkb.Device,
    extent: vk.Extent2D,

    next: Next,

    fn recreate(self: *Self) !void {
        try self.dev.deviceWaitIdle();
        try self.init_(self.ator, self.pd, self.dev, self.win, true);
    }

    pub fn getNext(self: *Self, img_acq_sem: vk.Semaphore, img_acq_fence: ?vk.Fence) !?Next {
        const res = self.dev.acquireNextImageKHR(
            self.hdl,
            std.math.maxInt(u64),
            img_acq_sem,
            if (img_acq_fence) |f| f else .null_handle,
        ) catch |err| {
            switch (err) {
                error.OutOfDateKHR => {
                    try self.recreate();
                    return null; // Skip this frame
                },
                else => {
                    return error.SwapchainAcquireFailed;
                },
            }
        };

        self.next = .{
            .image = self.images.items[res.image_index],
            .idx = res.image_index,
            .view = self.views.items[res.image_index],
        };
        return self.next;
    }

    pub fn present(self: *Self, present_queue: vkb.Queue, wait_sem: vk.Semaphore) !void {
        _ = present_queue.presentKHR(&.{
            .p_image_indices = &.{self.next.idx},
            .p_swapchains = &.{self.hdl},
            .swapchain_count = 1,
            .p_wait_semaphores = &.{wait_sem},
            .wait_semaphore_count = 1,
        }) catch |err| {
            switch (err) {
                error.OutOfDateKHR => {
                    try self.recreate();
                    // Recreate resources and continue without presenting
                },
                else => {
                    return error.SwapchainPresentFailed;
                },
            }
        };
    }

    pub fn init_(self: *Self, ator: Allocator, pd: vk.PhysicalDevice, dev: vkb.Device, win: glfw.Window, recr: bool) !void {
        var arena = memh.Arena.init(ator);
        defer arena.deinit();

        if (!recr) {
            if (glfw.createWindowSurface(vkb.inst.handle, win, null, &self.surf) != @intFromEnum(vk.Result.success))
                return error.SurfaceInitFailed;
        }

        self.format = blk: {
            const surf_fmts = try vkb.inst.getPhysicalDeviceSurfaceFormatsAllocKHR(pd, self.surf, arena.ator());
            for (surf_fmts) |fmt| {
                if (fmt.format == .b8g8r8a8_srgb and fmt.color_space == .srgb_nonlinear_khr) {
                    break :blk fmt;
                }
            }
            return error.SurfFormatNotFound;
        };

        const sel_pmode: vk.PresentModeKHR = blk: {
            const present_modes = try vkb.inst.getPhysicalDeviceSurfacePresentModesAllocKHR(pd, self.surf, arena.ator());
            for (present_modes) |pmode| {
                if (pmode == .mailbox_khr)
                    break :blk pmode;
            }

            // Fallback
            // FIFO: Show on next vertical blank (vsync) - guaranteed
            // Immediate: May cause tearing (No vsync)
            break :blk vk.PresentModeKHR.fifo_khr;
        };

        self.extent = blk: {
            const surf_caps = try vkb.inst.getPhysicalDeviceSurfaceCapabilitiesKHR(pd, self.surf);
            // Spec: (0xFFFFFFFF, 0xFFFFFFFF) indicates that the surface size will be determined by the extent of a swapchain
            // We assign the window client dimensions.
            if (surf_caps.current_extent.width == std.math.maxInt(u32)) {
                const draw_area = win.getFramebufferSize();
                break :blk .{
                    .width = std.math.clamp(draw_area.width, surf_caps.current_extent.width, surf_caps.current_extent.width),
                    .height = std.math.clamp(draw_area.height, surf_caps.current_extent.height, surf_caps.current_extent.height),
                };
            }

            break :blk surf_caps.current_extent;
        };

        if (recr) {
            // Associated resources in addition to surface/sc released
            // before new swapchain creation
            for (self.views.items) |v| {
                self.dev.destroyImageView(v, null);
            }
        }

        const old_sc = self.hdl;
        self.hdl = try dev.createSwapchainKHR(&vk.SwapchainCreateInfoKHR{
            .surface = self.surf,
            .image_format = self.format.format,
            .image_extent = self.extent,
            .image_array_layers = 1,
            .image_usage = .{ .color_attachment_bit = true, .transfer_dst_bit = true }, // allow clear color with transfer dst
            .image_color_space = .srgb_nonlinear_khr,
            .min_image_count = 3,
            .present_mode = sel_pmode,
            .image_sharing_mode = .exclusive,
            .pre_transform = .{ .identity_bit_khr = true },
            .composite_alpha = .{ .opaque_bit_khr = true },
            .clipped = vk.TRUE,
            .old_swapchain = if (old_sc != .null_handle) old_sc else .null_handle,
        }, null);

        if (recr) {
            // Swapchain destroyed after creation
            // Surface is identical, it appears internally managed
            // (Old surface must be same as new surface)
            self.dev.destroySwapchainKHR(old_sc, null);
            self.views.deinit();
            self.images.deinit();
        }

        const images = try dev.getSwapchainImagesAllocKHR(self.hdl, arena.ator());
        self.images = try std.ArrayList(vk.Image).initCapacity(ator, images.len);
        self.views = try std.ArrayList(vk.ImageView).initCapacity(ator, images.len);
        try self.images.appendSlice(images);

        for (self.images.items) |img| {
            const ci = vk.ImageViewCreateInfo{ .image = img, .view_type = .@"2d", .format = self.format.format, .components = .{
                .r = .identity,
                .g = .identity,
                .b = .identity,
                .a = .identity,
            }, .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_array_layer = 0,
                .base_mip_level = 0,
                .layer_count = 1,
                .level_count = 1,
            } };
            const v = try self.dev.createImageView(&ci, null);
            try self.views.append(v);
        }
    }

    pub fn init(self: *Self, ator: Allocator, pd: vk.PhysicalDevice, dev: vkb.Device, win: glfw.Window) !void {
        self.dev = dev;
        self.ator = ator;
        self.pd = pd;
        self.win = win;
        self.hdl = .null_handle;

        try self.init_(ator, pd, dev, win, false);
    }

    pub fn deinit(self: *Self) void {
        for (self.views.items) |v| {
            self.dev.destroyImageView(v, null);
        }
        self.dev.destroySwapchainKHR(self.hdl, null);
        vkb.inst.destroySurfaceKHR(self.surf, null);
    }
};
