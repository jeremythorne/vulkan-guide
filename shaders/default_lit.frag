#version 450

layout (location = 0) in vec3 inColor;
layout (location = 1) in vec3 inNormal;
layout (location = 0) out vec4 outFragColor;

layout (set = 0, binding = 1) uniform SceneData {
    vec4 fogColor;
    vec4 fogDistances;
    vec4 ambientColor;
    vec4 sunlightDirection;
    vec4 sunlightColor;
} sceneData;

void main() {
    vec3 lambert = clamp(
            dot(normalize(inNormal), sceneData.sunlightDirection.xyz),
            0.0, 1.0) *
            sceneData.sunlightColor.rgb;
    outFragColor = vec4(lambert + sceneData.ambientColor.rgb, 1.0f);
}
