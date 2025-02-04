# Zig VK

Zigging through VK.  
Heavy WIP.  

Current focus:  
* Fleshing out API  

## Zig commands
* zig build
    * Build the application
* zig build run
    * Build and run the application
* zig build btr
    * Build tracy server 
* zig build tr
    * Build and run tracy server 
* zig build all
    * Build and run both app and tracy server simultaneously

## Progress List
* Autocompile shaders with build.zig
* Arena allocator for Vulkan resources
* Vertex Pulling
* Staging upload context with queue ownership transfer
* Load texture and generate mips
* Resizing and swapchain recreation
* Buffer access with buffer device address
* Image access with descriptor indexing
* ImGUI
* Tracy
    * Client zig-bindings using ztracy
    * Server using submodule tracking v.0.11.1 


## Used Vulkan Features/Extensions
* Vulkan 1.3  
* Dynamic Rendering  
* Descriptor Indexing  
* Buffer Device Address 
* Timeline Semaphores

