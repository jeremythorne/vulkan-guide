#version 450

layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec3 vColor;

layout (location = 0) out vec3 outColor;
layout (location = 1) out vec3 outNormal;

layout (set = 0, binding = 0) uniform CameraBuffer {
    mat4 view;
    mat4 proj;
    mat4 viewproj;
} cameraData;

layout (push_constant) uniform Constants {
    vec4 data;
    mat4 render_matrix;
} pushConstants;

void main() {
    mat4 transformMatrix = cameraData.viewproj * pushConstants.render_matrix;
    gl_Position = transformMatrix * vec4(vPosition, 1.0f);
    outColor = vColor;
    outNormal = mat3(pushConstants.render_matrix) * vNormal;
}
