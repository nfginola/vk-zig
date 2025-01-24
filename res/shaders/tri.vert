#version 450
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference : require

// Unsized arrays, requires extension to be turned on!
// https://docs.vulkan.org/samples/latest/samples/extensions/descriptor_indexing/README.html
layout(set = 0, binding = 0) uniform UBO {
    vec3 rgb;
// } ubos[1000];
} ubos[];


// Pointer alignment, buffer_ref_align
layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer SomeData
{
    float floats[];
};

// Appears like there's no alignment requirement here,
// it just interprets the memory straight.
// (We can use packed struct in Zig and match it here)
struct Vertex {
    vec3 position;
    vec3 color;
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertexData
{
    Vertex data[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer IndexData
{
    uint data[];
};


layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer PerFrame
{
    vec3 rgb;
};


// Push constant don't need 
// nonuniformEXT qualifier as the value is 
// the same across the whole invocation group (a draw call in gfx case).
// nonuniformEXT needs to be used if value can be different within a
// draw call.
//
// An expression is considered dynamically uniform if all invocations in an invocation group have the same value.
// Draw call -> Single invocation group
//  If Draw call grabs PerDrawData (like materials) then they do not need nonuniformEXT()
// 
layout(push_constant) uniform constants
{
    SomeData some_data;
    PerFrame per_frame;
    VertexData verts;
    IndexData indices;
} regs;


layout(location = 0) out vec3 fragColor;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    uint id = regs.indices.data[gl_VertexIndex];
    gl_Position = vec4(regs.verts.data[id].position.xyz, 1.0);
    fragColor = vec3(regs.per_frame.rgb);

}
