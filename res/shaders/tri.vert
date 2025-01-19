#version 450

out gl_PerVertex {
    vec4 gl_Position;
};

layout(set = 0, binding = 0) uniform UBO {
    vec3 rgb;
} ubos[];

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
    // fragColor = vec3(1.0);
    fragColor = inColor;
    vec3 b1 = ubos[0].rgb;      // f0 resource
    vec3 b2 = ubos[1].rgb;      // f1 resource
    vec3 b3 = ubos[99].rgb;     // no resource, zeroed
    vec3 col = b1 + b2 + b3;
    fragColor = col;

}
