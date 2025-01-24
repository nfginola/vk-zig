#version 450
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference : require

out gl_PerVertex {
    vec4 gl_Position;
};

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

// Pointer alignment, buffer_ref_align
layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer PositionData
{
    vec3 positions[];
};


// Pointer alignment, buffer_ref_align
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
    PositionData positions;
} push_constants;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

vec3 positions[3] = vec3[](
    vec3(0.0, -0.5, 0.0),
    vec3(-0.5, 0.5, 0.0),
    vec3(0.5, 0.5, 0.0)
);

vec3 colors[3] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);

void main() {
    // gl_Position = vec4(positions[gl_VertexIndex], 1.0);
    // fragColor = colors[gl_VertexIndex];
    gl_Position = vec4(inPosition, 1.0);
    fragColor = vec3(push_constants.per_frame.rgb);

}
