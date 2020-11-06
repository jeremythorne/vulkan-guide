#version 450
layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec3 vColor;
layout (location = 3) in vec2 vTexCoord;

layout (location = 0) out vec3 outColor;
layout (location = 1) out vec2 texCoord;
layout (location = 2) out vec3 outNormal;

layout(set = 0, binding = 0) uniform  CameraBuffer{   
    mat4 view;
    mat4 proj;
	mat4 viewproj;
} cameraData;

struct ObjectData{
	mat4 model;
}; 

//all object matrices
layout(std140,set = 1, binding = 0) readonly buffer ObjectBuffer{   

	ObjectData objects[];
} objectBuffer;

//all object indices
layout(set = 1, binding = 1) readonly buffer InstanceBuffer{   

	int IDs[];
} instanceBuffer;

void main() 
{	
	int index = instanceBuffer.IDs[gl_InstanceIndex];
	
	mat4 modelMatrix = objectBuffer.objects[index].model;
	mat4 transformMatrix = (cameraData.viewproj * modelMatrix);
	gl_Position = transformMatrix * vec4(vPosition, 1.0f);
	outNormal = (modelMatrix * vec4(vNormal,0.f)).xyz;
	outColor = vColor;
	texCoord = vTexCoord;
}