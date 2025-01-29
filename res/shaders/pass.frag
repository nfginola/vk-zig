#version 450
#extension GL_EXT_nonuniform_qualifier : require
        
// layout(location = 0) in vec3 fragColor;
layout(location = 0) in vec2 fragUv;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler imm_samp;

// Unsized arrays, requires extension to be turned on!
// https://docs.vulkan.org/samples/latest/samples/extensions/descriptor_indexing/README.html
layout(set = 0, binding = 1) uniform texture2D texs[];


void main() {
    vec4 color = texture(sampler2D(texs[nonuniformEXT(0)], imm_samp), vec2(fragUv));
    outColor = vec4(color.rgb, 1.0);
}
